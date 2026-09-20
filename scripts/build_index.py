"""
构建 ChromaDB 向量索引
用法: python scripts/build_index.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chromadb
import torch
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

KB_ROOT = ROOT / "data" / "knowledge_base"
INDEX_DIR = ROOT / "data" / "chroma"
EMBED_MODEL = ROOT / "models" / "bge-small-zh-v1.5"
COLLECTION_NAME = "knowledge_base"

# 分类映射: 文件夹名 -> BERT 类别
FOLDER_TO_LABEL = {
    "policy": "policy_attendance",
    "it": "it_vpn",
    "finance": "finance_expense",
    "hr": "hr_onboarding",
    "admin": "admin_logistics",
    "engineering": "engineering_eq",
    "legal": "legal_contract",
    "sales": "sales_marketing",
}


def chunk_text(text, max_len=400):
    """按段落分块，超长段落按句子切。"""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for p in paragraphs:
        if len(p) <= max_len:
            chunks.append(p)
        else:
            # 按句子切
            sentences = p.replace("。", "。\n").split("\n")
            cur = ""
            for s in sentences:
                if len(cur) + len(s) <= max_len:
                    cur += s
                else:
                    if cur:
                        chunks.append(cur)
                    cur = s
            if cur:
                chunks.append(cur)
    return chunks


def main():
    print("Loading embedding model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = SentenceTransformer(str(EMBED_MODEL), device=device)

    # 清空旧索引
    if INDEX_DIR.exists():
        import shutil
        shutil.rmtree(INDEX_DIR)
    INDEX_DIR.mkdir(parents=True)

    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    col = client.get_or_create_collection(COLLECTION_NAME)

    all_docs, all_metas, all_ids = [], [], []
    doc_count = 0
    chunk_count = 0

    for folder in sorted(KB_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        label = FOLDER_TO_LABEL.get(folder.name, folder.name)
        for md_file in sorted(folder.glob("*.md")):
            doc_count += 1
            text = md_file.read_text(encoding="utf-8")
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                chunk_count += 1
                all_docs.append(chunk)
                all_metas.append({
                    "label": label,
                    "source": md_file.name,
                    "category_folder": folder.name,
                    "chunk_index": i,
                })
                all_ids.append(f"{folder.name}_{md_file.stem}_{i}")

    print(f"Found {doc_count} docs, {chunk_count} chunks")

    print("Embedding...")
    embeddings = model.encode(all_docs, show_progress_bar=True, normalize_embeddings=True).tolist()

    print("Adding to ChromaDB...")
    col.add(ids=all_ids, documents=all_docs, embeddings=embeddings, metadatas=all_metas)
    print(f"Done! Collection has {col.count()} chunks")


if __name__ == "__main__":
    main()
