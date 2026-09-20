"""ChatEngine：可复用问答核心（BERT 分类 + Chroma 检索 + Ollama 生成）。

从 webui.py 提取，供 FastAPI 并发接口复用，保证与 Streamlit UI 行为一致。
线程安全：BERT 推理加锁串行（毫秒级），LLM 调用不加锁（IO 等待期并发）。
"""
import sys
import json
import threading
import requests
import torch
import torch.nn.functional as F
import chromadb
from datetime import datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LABELS = [
    "policy_attendance", "it_vpn", "finance_expense", "hr_onboarding",
    "admin_logistics", "engineering_eq", "legal_contract", "sales_marketing",
    "other_chitchat",   # 第 9 类（与 data/class.txt 顺序一致）
]
OTHER_INTENT = "other_chitchat"
OTHER_THRESHOLD = 0.55            # 兜底：BERT 最高置信度低于此值 → 判为"其他"
LABEL_CN = {
    "policy_attendance": ("考勤制度", "上下班/打卡/请假/年假"),
    "it_vpn": ("IT 支持", "VPN/密码/电脑/打印机"),
    "finance_expense": ("财务报销", "报销/发票/工资/差旅"),
    "hr_onboarding": ("人力资源", "入职/离职/社保/福利"),
    "admin_logistics": ("行政后勤", "食堂/班车/宿舍/会议室"),
    "engineering_eq": ("工程设备", "产线/维修/点检/备件"),
    "legal_contract": ("法务合同", "合同/公章/保密/知识产权"),
    "sales_marketing": ("销售市场", "报价/订单/客户/展会"),
    "other_chitchat": ("其他", "闲聊/业务外问题"),
}
CKPT = ROOT / "checkpoints" / "intent_bert" / "best.pt"
MODEL_DIR = ROOT / "models" / "bert-base-chinese"
EMBED_DIR = ROOT / "models" / "bge-small-zh-v1.5"
CHROMA_DIR = ROOT / "data" / "chroma"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "qwen3.5:9b"

# BERT 分类与 Embedding 推理放 CPU：延迟毫秒级，避免多 worker 各自占用 GPU 显存。
# GPU 显存只留给 qwen3.5:9b，避免 12GB 小显存卡被占满。
CLASSIFY_DEVICE = "cpu"

CHAT_SYSTEM = """你是企业内部智能客服，当前为自由闲聊模式，用户的问题与公司制度、业务无关。
规则：
1. 友好自然地回答，80字以内
2. 不要编造任何公司制度信息
3. 结尾可温和引导用户提问业务问题（考勤/报销/IT/人事等）"""


def _now_zh():
    now = datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return f"{now:%Y年%m月%d日 %H:%M} {weekdays[now.weekday()]}"


class ChatEngine:
    def __init__(self):
        device = CLASSIFY_DEVICE
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        bert = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR), num_labels=len(LABELS))
        state = torch.load(str(CKPT), map_location=device, weights_only=True)
        bert.load_state_dict(state)
        bert.to(device).eval()
        self.bert = bert
        self.emb = SentenceTransformer(str(EMBED_DIR), device=device)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.col = client.get_collection("knowledge_base")
        self._bert_lock = threading.Lock()

    # ---------- 意图分类 ----------
    def classify(self, query):
        enc = self.tok(query, return_tensors="pt", truncation=True, max_length=48, padding=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self._bert_lock, torch.no_grad():
            probs = F.softmax(self.bert(**enc).logits, -1).cpu().numpy()[0]
        return probs

    # ---------- 向量检索 ----------
    def retrieve(self, category, query, top_k=3):
        q_emb = self.emb.encode([query], normalize_embeddings=True).tolist()
        res = self.col.query(query_embeddings=q_emb, n_results=top_k,
                             where={"label": category})
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        return docs, metas

    # ---------- Prompt 构建 ----------
    def _build_qa_messages(self, query, category, docs, metas):
        cn, _ = LABEL_CN[category]
        ctx_parts, src_labels = [], []
        seen = set()
        for d, m in zip(docs, metas):
            src = m.get("source", "")
            seg = m.get("chunk_index", 0) + 1
            ctx_parts.append(f"[{src} 第{seg}段]\n{d}")
            label = f"{src} 第{seg}段"
            if label not in seen:
                seen.add(label)
                src_labels.append(label)
        context = "\n---\n".join(ctx_parts)
        sys = f"""你是企业内部智能客服。根据以下制度片段回答员工问题。
规则：
1. 仅基于片段回答，片段中没有就说"未在制度中找到相关内容"，禁止编造
2. 简洁友好，150字以内
3. 末尾标注出处，格式：【来源：{cn} · 文件名 第N段】，引用多个片段时全部列出
======
{context}"""
        return [{"role": "system", "content": sys},
                {"role": "user", "content": query}], src_labels

    def _call_llm(self, messages, num_predict=400, temperature=0.3, stream=False):
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": stream,
            "think": False,
            "options": {"num_predict": num_predict, "temperature": temperature},
        }
        return requests.post(OLLAMA_URL, json=payload, stream=stream, timeout=120)

    # ---------- 非流式问答（供 /api/chat） ----------
    def answer(self, query):
        q = query.strip()
        if not q:
            return {"answer": "", "intent": "", "tags": [], "sources": []}

        # 时间查询直答
        if is_time_query(q):
            return {"answer": f"当前时间是：{_now_zh()}", "intent": "system_time",
                    "tags": [("系统时间", "src")], "sources": []}

        probs = self.classify(q)
        top3 = probs.argsort()[::-1][:3]
        top_label = LABELS[top3[0]]

        if top_label == OTHER_INTENT or probs[top3[0]] < OTHER_THRESHOLD:
            r = self._call_llm(
                [{"role": "system", "content": CHAT_SYSTEM},
                 {"role": "user", "content": q}],
                num_predict=200, temperature=0.7)
            d = r.json()
            answer = d.get("message", {}).get("content", "").strip()
            if not answer:
                answer = "这个问题我不太确定呢，可以换个问法，或问问我考勤、报销、IT 之类的业务问题~"
            return {"answer": answer, "intent": OTHER_INTENT,
                    "tags": [("其他 · 闲聊", "first")], "sources": []}

        cat = top_label
        tags = [(f"{LABEL_CN[LABELS[i]][0]} {probs[i]:.0%}",
                 "first" if i == top3[0] else "") for i in top3]
        docs, metas = self.retrieve(cat, q)
        if not docs:
            return {"answer": "抱歉，这个问题我暂时没有把握，已为您转接人工客服或提交工单。",
                    "intent": cat, "tags": tags, "sources": []}
        messages, src_labels = self._build_qa_messages(q, cat, docs, metas)
        r = self._call_llm(messages)
        d = r.json()
        answer = d.get("message", {}).get("content", "").strip()
        if not answer:
            answer = "抱歉，暂时无法生成回答，请稍后重试。"
        return {"answer": answer, "intent": cat, "tags": tags,
                "sources": [{"file": s.split(" ")[0], "label": s} for s in src_labels]}

    # ---------- 流式问答（供 /api/chat/stream） ----------
    def stream_answer(self, query):
        """yield (delta_text, tags, sources, done)"""
        q = query.strip()
        if not q:
            yield "", [], [], True
            return

        if is_time_query(q):
            yield f"当前时间是：{_now_zh()}", [("系统时间", "src")], [], True
            return

        probs = self.classify(q)
        top3 = probs.argsort()[::-1][:3]
        top_label = LABELS[top3[0]]

        if top_label == OTHER_INTENT or probs[top3[0]] < OTHER_THRESHOLD:
            r = self._call_llm(
                [{"role": "system", "content": CHAT_SYSTEM},
                 {"role": "user", "content": q}],
                num_predict=200, temperature=0.7, stream=True)
            full = ""
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                ch = obj.get("message", {}).get("content", "")
                if ch:
                    full += ch
                    yield ch, [("其他 · 闲聊", "first")], [], False
                if obj.get("done"):
                    break
            if not full:
                yield "这个问题我不太确定呢，可以换个问法，或问问我考勤、报销、IT 之类的业务问题~", [], [], False
            yield "", [], [], True
            return

        cat = top_label
        tags = [(f"{LABEL_CN[LABELS[i]][0]} {probs[i]:.0%}",
                 "first" if i == top3[0] else "") for i in top3]
        docs, metas = self.retrieve(cat, q)
        if not docs:
            yield "抱歉，这个问题我暂时没有把握，已为您转接人工客服或提交工单。", tags, [], True
            return
        messages, src_labels = self._build_qa_messages(q, cat, docs, metas)
        r = self._call_llm(messages, stream=True)
        full = ""
        for line in r.iter_lines():
            if not line:
                continue
            try:
                obj = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            ch = obj.get("message", {}).get("content", "")
            if ch:
                full += ch
                yield ch, tags, src_labels, False
            if obj.get("done"):
                break
        if not full:
            yield "抱歉，暂时无法生成回答，请稍后重试。", [], [], False
        yield "", [], [], True


def is_time_query(q):
    q = q.strip()
    time_patterns = ["现在几点", "今天几号", "当前时间", "现在时间", "现在日期",
                     "今天星期", "现在星期"]
    if any(p in q for p in time_patterns):
        return True
    if "几点" in q and "上班" not in q and "下班" not in q and "考勤" not in q:
        return True
    return False


_engine = None
_engine_lock = threading.Lock()


def get_engine():
    """进程内单例（多 worker 各自持有一份，避免跨进程共享模型）。"""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = ChatEngine()
    return _engine


# noqa: E402
