"""训练 BERT 意图分类器（大规模版）
数据格式：data/train.txt / test.txt / dev.txt，每行 `文本\t标签ID`
用法：python scripts/train_intent.py --epochs 3 --bs 64
"""
import sys, argparse, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from loguru import logger

DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints" / "intent_bert"


def load_txt(path):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" in line:
                text, lbl = line.rsplit("\t", 1)
                texts.append(text)
                labels.append(int(lbl))
    return texts, labels


class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=48):
        self.texts, self.labels = texts, labels
        self.tok, self.max_len = tokenizer, max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[i], dtype=torch.long),
        }


def evaluate(model, dl, device):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for batch in dl:
            logits = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device)).logits
            correct += (logits.argmax(-1).cpu() == batch["label"]).sum().item()
            total += len(batch["label"])
    return correct / max(total, 1)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = ROOT / args.model
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    # 读 class.txt 确定类别数
    class_path = DATA_DIR / "class.txt"
    num_labels = len(class_path.read_text(encoding="utf-8").strip().split("\n"))
    logger.info(f"类别数: {num_labels}")

    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), num_labels=num_labels).to(device)

    train_t, train_l = load_txt(DATA_DIR / "train.txt")
    dev_t, dev_l = load_txt(DATA_DIR / "dev.txt")
    test_t, test_l = load_txt(DATA_DIR / "test.txt")
    logger.info(f"train={len(train_t)} dev={len(dev_t)} test={len(test_t)}")

    train_dl = DataLoader(IntentDataset(train_t, train_l, tokenizer),
                          batch_size=args.bs, shuffle=True)
    dev_dl = DataLoader(IntentDataset(dev_t, dev_l, tokenizer), batch_size=args.bs)
    test_dl = DataLoader(IntentDataset(test_t, test_l, tokenizer), batch_size=args.bs)

    opt = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_acc = 0
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss = 0
        for step, batch in enumerate(train_dl):
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=torch.bfloat16):
                out = model(input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                            labels=batch["label"].to(device))
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            total_loss += loss.item()
            if step % 100 == 0:
                logger.info(f"Epoch {ep+1} step {step}/{len(train_dl)} loss={loss.item():.4f}")

        dev_acc = evaluate(model, dev_dl, device)
        logger.info(f"Epoch {ep+1}/{args.epochs}  loss={total_loss/len(train_dl):.4f}  "
                    f"dev_acc={dev_acc:.4f}  {time.time()-t0:.0f}s")
        if dev_acc > best_acc:
            best_acc = dev_acc
            torch.save(model.state_dict(), CKPT_DIR / "best.pt")
            logger.info(f"保存最佳模型 acc={dev_acc:.4f}")

    test_acc = evaluate(model, test_dl, device)
    logger.info(f"训练完成 best_dev_acc={best_acc:.4f}  test_acc={test_acc:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/bert-base-chinese")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    main(ap.parse_args())
