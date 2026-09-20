"""企业内部智能客服 - Streamlit UI
微信风格聊天气泡 · Light / Dark 双主题切换
"""
import sys, requests
from html import escape
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import chromadb
from sentence_transformers import SentenceTransformer
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = [
    "policy_attendance", "it_vpn", "finance_expense", "hr_onboarding",
    "admin_logistics", "engineering_eq", "legal_contract", "sales_marketing",
    "other_chitchat",   # 第 9 类（与 data/class.txt 顺序一致）
]
OTHER_INTENT = "other_chitchat"   # 其他/闲聊：无知识库，不检索，LLM 自由回答
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
import socket
def get_lan_ip():
    """获取局域网 IPv4：排除虚拟网卡/PPP 拨号/APIPA，优先 192.168/10.x 段。"""
    try:
        cands = []
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip.startswith(("127.", "169.254.")):
                continue
            cands.append(ip)
        for ip in cands:
            if ip.startswith(("192.168.", "10.", "172.")) and not ip.startswith(("10.254.", "172.28.", "192.168.88.")):
                return ip
        return cands[0] if cands else "127.0.0.1"
    except:
        return "127.0.0.1"

LAN_IP = get_lan_ip()
CKPT = ROOT / "checkpoints" / "intent_bert" / "best.pt"
MODEL_DIR = ROOT / "models" / "bert-base-chinese"
EMBED_DIR = ROOT / "models" / "bge-small-zh-v1.5"
CHROMA_DIR = ROOT / "data" / "chroma"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "qwen3.5:9b"

st.set_page_config(page_title="企业智能客服", page_icon="💬", layout="wide")

# ---------- 双主题配色（参考微信聊天） ----------
THEMES = {
    "light": {
        "page_bg": "#ededed", "text": "#191919", "scroll": "rgba(0,0,0,.18)",
        "header_bg": "rgba(237,237,237,.85)", "header_border": "rgba(0,0,0,.07)",
        "title": "#191919", "sub": "#9a9a9a",
        "online_bg": "rgba(7,193,96,.12)", "online_text": "#0a8f4d",
        "bot_bg": "#ffffff", "bot_border": "#e3e3e3", "bot_text": "#191919",
        "user_bg": "#95ec69", "user_border": "#86d95c", "user_text": "#1a1a1a",
        "bubble_shadow": "0 1px 2px rgba(0,0,0,.05)",
        "tag_bg": "#f0f4ff", "tag_text": "#576b95", "tag_border": "#dfe8fb",
        "tag_first_bg": "#576b95", "tag_first_text": "#ffffff", "tag_first_border": "#576b95",
        "src_bg": "rgba(7,193,96,.10)", "src_text": "#0a8f4d", "src_border": "rgba(7,193,96,.25)",
        "time_color": "#b0b0b0",
        "sidebar_bg": "#f7f7f7", "sidebar_text": "#191919", "divider": "rgba(0,0,0,.08)",
        "arch_bg": "#ffffff", "arch_border": "#e8e8e8", "arch_text": "#777777",
        "cat_hover": "rgba(0,0,0,.045)", "cat_name": "#191919", "cat_desc": "#9a9a9a",
        "input_bg": "#ffffff", "input_border": "#d9d9d9", "input_text": "#191919",
    },
    "dark": {
        "page_bg": "#111111", "text": "#e8e8e8", "scroll": "rgba(255,255,255,.15)",
        "header_bg": "rgba(17,17,17,.82)", "header_border": "rgba(255,255,255,.07)",
        "title": "#ededed", "sub": "#6f6f6f",
        "online_bg": "rgba(7,193,96,.15)", "online_text": "#34d17b",
        "bot_bg": "#2b2b2b", "bot_border": "#3a3a3a", "bot_text": "#e8e8e8",
        "user_bg": "#07c160", "user_border": "#06a854", "user_text": "#ffffff",
        "bubble_shadow": "0 1px 2px rgba(0,0,0,.35)",
        "tag_bg": "rgba(139,92,246,.16)", "tag_text": "#c4b5fd", "tag_border": "rgba(139,92,246,.30)",
        "tag_first_bg": "#8b5cf6", "tag_first_text": "#ffffff", "tag_first_border": "#8b5cf6",
        "src_bg": "rgba(7,193,96,.14)", "src_text": "#34d17b", "src_border": "rgba(7,193,96,.30)",
        "time_color": "#5f5f5f",
        "sidebar_bg": "#1b1b1b", "sidebar_text": "#e8e8e8", "divider": "rgba(255,255,255,.08)",
        "arch_bg": "#232323", "arch_border": "#333333", "arch_text": "#9a9a9a",
        "cat_hover": "rgba(255,255,255,.06)", "cat_name": "#e8e8e8", "cat_desc": "#777777",
        "input_bg": "#262626", "input_border": "#3a3a3a", "input_text": "#e8e8e8",
    },
}

# ---------- SVG 图标 ----------
LOGO_ICON = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none">
<path d="M4 6.8C4 5.2 5.3 4 6.9 4h10.2c1.6 0 2.9 1.2 2.9 2.8v6.4c0 1.6-1.3 2.8-2.9 2.8H13l-3.6 3.2c-.5.4-1.3.1-1.3-.6V16H6.9C5.3 16 4 14.8 4 13.2V6.8Z" stroke="#fff" stroke-width="1.6" stroke-linejoin="round"/>
<circle cx="9" cy="10" r="1" fill="#fff"/><circle cx="12.5" cy="10" r="1" fill="#fff"/><circle cx="16" cy="10" r="1" fill="#fff"/>
</svg>"""
BOT_ICON = """<svg width="21" height="21" viewBox="0 0 24 24" fill="none">
<rect x="4.5" y="7.5" width="15" height="12" rx="3.5" stroke="#fff" stroke-width="1.6"/>
<path d="M12 7.5V4.5" stroke="#fff" stroke-width="1.6" stroke-linecap="round"/>
<circle cx="12" cy="3.6" r="1.1" fill="#fff"/>
<circle cx="9.2" cy="12.4" r="1.15" fill="#fff"/><circle cx="14.8" cy="12.4" r="1.15" fill="#fff"/>
<path d="M9.6 16h4.8" stroke="#fff" stroke-width="1.6" stroke-linecap="round"/>
<path d="M2.8 11.5v3.6M21.2 11.5v3.6" stroke="#fff" stroke-width="1.6" stroke-linecap="round"/>
</svg>"""
USER_ICON = """<svg width="21" height="21" viewBox="0 0 24 24" fill="none">
<circle cx="12" cy="8.2" r="3.4" stroke="#fff" stroke-width="1.6"/>
<path d="M5.2 19.2c.9-3.2 3.6-4.9 6.8-4.9s5.9 1.7 6.8 4.9" stroke="#fff" stroke-width="1.6" stroke-linecap="round"/>
</svg>"""

# 侧边栏分类图标：emoji + 底色
CAT_ICON = {
    "policy_attendance": ("🕐", "#dbeafe"), "it_vpn": ("💻", "#e0e7ff"),
    "finance_expense": ("💰", "#fef3c7"), "hr_onboarding": ("👥", "#fce7f3"),
    "admin_logistics": ("🍱", "#ffedd5"), "engineering_eq": ("🔧", "#e2e8f0"),
    "legal_contract": ("📜", "#ede9fe"), "sales_marketing": ("📈", "#dcfce7"),
    "other_chitchat": ("💬", "#f3e8ff"),
}

if "theme" not in st.session_state:
    st.session_state.theme = "light"
C = THEMES[st.session_state.theme]

def _sync_theme():
    st.session_state.theme = "dark" if st.session_state.dark_toggle else "light"

st.markdown(f"""
<style>
html, body, .stApp, [data-testid="stAppViewContainer"] {{
    background: {C['page_bg']} !important; color: {C['text']};
    font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB",
                 "Microsoft YaHei", "Segoe UI", sans-serif;
}}
header[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"],
[data-testid="stBottom"] > div {{
    background: {C['page_bg']} !important;
}}
::selection {{ background: rgba(139,92,246,.28); }}
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-thumb {{ background: {C['scroll']}; border-radius: 4px; }}
::-webkit-scrollbar-track {{ background: transparent; }}

/* ===== 顶栏 ===== */
.header-bar {{ position: fixed; top: 0; left: 0; right: 0; z-index: 999;
    background: {C['header_bg']}; backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid {C['header_border']}; padding: 10px 20px; }}
.header-inner {{ max-width: 780px; margin: 0 auto; display: flex; align-items: center; gap: 12px; }}
.logo {{ width: 40px; height: 40px; border-radius: 10px; flex-shrink: 0;
    background: linear-gradient(135deg, #667eea, #8b5cf6);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 12px rgba(102,126,234,.35); }}
.logo.sm {{ width: 34px; height: 34px; border-radius: 8px; }}
.brand-title {{ font-size: 17px; font-weight: 700; color: {C['title']}; letter-spacing: .3px; }}
.brand-sub {{ font-size: 11.5px; color: {C['sub']}; margin-top: 1px; }}
.online {{ margin-left: auto; font-size: 11px; font-weight: 500; color: {C['online_text']};
    background: {C['online_bg']}; padding: 4px 11px; border-radius: 20px;
    display: flex; align-items: center; gap: 5px; }}
.online::before {{ content: ""; width: 6px; height: 6px; border-radius: 50%;
    background: #07c160; box-shadow: 0 0 6px #07c160; }}

/* ===== 聊天区（微信风格） ===== */
.chat-area {{ max-width: 780px; margin: 0 auto; padding: 84px 20px 16px; }}
.row {{ display: flex; margin: 14px 0; align-items: flex-start; animation: msgIn .28s ease both; }}
@keyframes msgIn {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: none; }} }}
.row.user {{ justify-content: flex-end; }}
.avatar {{ width: 40px; height: 40px; border-radius: 8px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 2px 6px rgba(0,0,0,.12); }}
.avatar.bot {{ background: linear-gradient(135deg, #667eea, #8b5cf6); }}
.avatar.user {{ background: linear-gradient(135deg, #5b8def, #3b6fd4); }}
.msg-wrap {{ max-width: 76%; display: flex; flex-direction: column; }}
.row.user .msg-wrap {{ align-items: flex-end; }}
.bubble {{ position: relative; padding: 10px 14px; border-radius: 10px;
    font-size: 14.5px; line-height: 1.7; white-space: pre-wrap; word-break: break-word; }}
.row.bot .bubble {{ margin-left: 12px; border-top-left-radius: 3px;
    background: {C['bot_bg']}; border: 1px solid {C['bot_border']};
    color: {C['bot_text']}; box-shadow: {C['bubble_shadow']}; }}
.row.user .bubble {{ margin-right: 12px; border-top-right-radius: 3px;
    background: {C['user_bg']}; border: 1px solid {C['user_border']};
    color: {C['user_text']}; box-shadow: {C['bubble_shadow']}; }}
.row.bot .bubble::before {{ content: ""; position: absolute; top: 14px; left: -5px;
    width: 9px; height: 9px; transform: rotate(45deg); background: {C['bot_bg']};
    border-left: 1px solid {C['bot_border']}; border-bottom: 1px solid {C['bot_border']}; }}
.row.user .bubble::before {{ content: ""; position: absolute; top: 14px; right: -5px;
    width: 9px; height: 9px; transform: rotate(45deg); background: {C['user_bg']};
    border-top: 1px solid {C['user_border']}; border-right: 1px solid {C['user_border']}; }}
.tags {{ margin: 6px 0 0 12px; display: flex; gap: 6px; flex-wrap: wrap; }}
.tag {{ font-size: 11px; padding: 3px 10px; border-radius: 20px; font-weight: 500;
    background: {C['tag_bg']}; color: {C['tag_text']}; border: 1px solid {C['tag_border']};
    display: inline-flex; align-items: center; gap: 5px; }}
.tag::before {{ content: ""; width: 5px; height: 5px; border-radius: 50%;
    background: currentColor; opacity: .65; }}
.tag.first {{ background: {C['tag_first_bg']}; color: {C['tag_first_text']};
    border-color: {C['tag_first_border']}; }}
.tag.src {{ background: {C['src_bg']}; color: {C['src_text']};
    border-color: {C['src_border']}; }}
.time-divider {{ text-align: center; font-size: 11.5px; margin: 18px 0 6px;
    font-weight: 500; color: {C['time_color']}; }}

/* ===== 侧边栏 ===== */
[data-testid="stSidebar"] {{ background: {C['sidebar_bg']} !important;
    border-right: 1px solid {C['header_border']}; }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] span {{
    color: {C['sidebar_text']} !important; }}
[data-testid="stSidebar"] hr {{ border-color: {C['divider']} !important; }}
.side-brand {{ display: flex; align-items: center; gap: 10px; margin: 2px 0 4px; }}
.side-title {{ font-size: 16px; font-weight: 700; color: {C['title']}; }}
.side-sub {{ font-size: 11px; color: {C['sub']}; margin-top: 1px; }}
.sec-title {{ font-size: 13px; font-weight: 700; color: {C['title']}; margin: 10px 0 6px; }}
.arch {{ border-radius: 12px; padding: 12px 14px; font-size: 12px; line-height: 2;
    background: {C['arch_bg']}; border: 1px solid {C['arch_border']}; color: {C['arch_text']}; }}
.arch b {{ color: {C['title']}; }}
.cat-item {{ display: flex; align-items: center; gap: 10px; padding: 7px 8px;
    border-radius: 10px; transition: background .15s; }}
.cat-item:hover {{ background: {C['cat_hover']}; }}
.cat-ico {{ width: 32px; height: 32px; border-radius: 8px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 15px; }}
.cat-name {{ font-size: 13.5px; font-weight: 600; color: {C['cat_name']}; }}
.cat-desc {{ font-size: 11px; color: {C['cat_desc']}; margin-top: 1px; }}

/* ===== 输入框 ===== */
.stChatInputContainer, [data-testid="stChatInput"],
[data-testid="stChatInput"] div, [data-testid="stChatInputTextArea"] {{
    background: {C['input_bg']} !important;
}}
[data-testid="stChatInput"], [data-testid="stChatInput"] > div:first-child {{
    border-color: {C['input_border']} !important; border-radius: 12px !important; }}
[data-testid="stChatInput"]:focus-within,
[data-testid="stChatInput"] > div:first-child:focus-within {{
    border-color: #8b5cf6 !important; box-shadow: 0 0 0 3px rgba(139,92,246,.18); }}
[data-testid="stChatInput"] textarea, [data-testid="stChatInputTextArea"] {{
    color: {C['input_text']} !important; }}
[data-testid="stChatInput"] button {{ color: {C['tag_text']} !important; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """加载 BERT 分类器 + Embedding 模型 + ChromaDB。
    推理放 CPU（毫秒级），GPU 显存只留给 qwen3.5:9b，避免显存被多进程模型副本占满。"""
    device = "cpu"
    # BERT 分类器
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    bert = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR), num_labels=len(LABELS))
    state = torch.load(str(CKPT), map_location=device, weights_only=True)
    bert.load_state_dict(state)
    bert.to(device).eval()
    # Embedding
    emb = SentenceTransformer(str(EMBED_DIR), device=device)
    # ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_collection("knowledge_base")
    return tok, bert, device, emb, col

@st.cache_resource
def warm_up_llm():
    try:
        requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL, "messages": [{"role": "user", "content": "hi"}],
            "stream": False, "think": False,
            "options": {"num_predict": 5}
        }, timeout=120)
    except Exception as e:
        print("warm up failed:", e)

tok, bert_model, device, embed_model, chroma_col = load_models()
warm_up_llm()


def classify(query):
    enc = tok(query, return_tensors="pt", truncation=True, max_length=48, padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        probs = F.softmax(bert_model(**enc).logits, -1).cpu().numpy()[0]
    return probs


def retrieve(category, query, top_k=3):
    """向量检索：在指定类别下找 Top-K 相关片段，返回 (片段列表, 元数据列表)。"""
    q_emb = embed_model.encode([query], normalize_embeddings=True).tolist()
    res = chroma_col.query(
        query_embeddings=q_emb,
        n_results=top_k,
        where={"label": category}
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return docs, metas


def ask_llm(query, category, docs, metas):
    """RAG 回答：上下文携带来源（文件名+段号），要求 LLM 标注出处。
    返回 (回答, 来源标签列表)。"""
    cn, _ = LABEL_CN[category]
    ctx_parts, src_labels = [], []
    for d, m in zip(docs, metas):
        src = m.get("source", "")
        seg = m.get("chunk_index", 0) + 1
        ctx_parts.append(f"[{src} 第{seg}段]\n{d}")
        label = f"{src} 第{seg}段"
        if label not in src_labels:
            src_labels.append(label)
    context = "\n---\n".join(ctx_parts)
    sys = f"""你是企业内部智能客服。根据以下制度片段回答员工问题。
规则：
1. 仅基于片段回答，片段中没有就说"未在制度中找到相关内容"，禁止编造
2. 简洁友好，150字以内
3. 末尾标注出处，格式：【来源：{cn} · 文件名 第N段】，引用多个片段时全部列出
======
{context}"""
    r = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": sys}, {"role": "user", "content": query}],
        "stream": False, "think": False,
        "options": {"num_predict": 400, "temperature": 0.3}
    }, timeout=60)
    d = r.json()
    return d.get("message", {}).get("content", "").strip(), src_labels


def stream_ask_llm(query, cat, docs, metas):
    """业务问题：流式生成回答，yield (chunk_text, src_labels)。"""
    cn = LABEL_CN[cat][0]
    ctx_parts, src_labels = [], []
    seen = set()
    for d, m in zip(docs, metas):
        src = m.get("source", "")
        seg = m.get("chunk_index", 0) + 1
        ctx_parts.append(f"[{src} 第{seg}段]\n{d}")
        label = f"{src} 第{seg}段"
        if label not in seen:
            seen.add(label); src_labels.append(label)
    context = "\n---\n".join(ctx_parts)
    sys = f"""你是企业内部智能客服。根据以下制度片段回答员工问题。
规则：
1. 仅基于片段回答，片段中没有就说"未在制度中找到相关内容"，禁止编造
2. 简洁友好，150字以内
3. 末尾标注出处，格式：【来源：{cn} · 文件名 第N段】，引用多个片段时全部列出
======
{context}"""
    r = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": sys}, {"role": "user", "content": query}],
        "stream": True, "think": False,
        "options": {"num_predict": 400, "temperature": 0.3}
    }, stream=True, timeout=120)
    for line in r.iter_lines():
        if not line: continue
        try:
            import json as _json
            obj = _json.loads(line.decode("utf-8"))
            ch = obj.get("message", {}).get("content", "")
            if ch: yield ch, src_labels
            if obj.get("done"): break
        except: pass


def stream_chat_llm(query):
    """闲聊：流式生成，yield chunk。"""
    sys = """你是企业内部智能客服，当前为自由闲聊模式，用户的问题与公司制度、业务无关。
规则：
1. 友好自然地回答，80字以内
2. 不要编造任何公司制度信息
3. 结尾可温和引导用户提问业务问题（考勤/报销/IT/人事等）"""
    r = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": sys}, {"role": "user", "content": query}],
        "stream": True, "think": False,
        "options": {"num_predict": 200, "temperature": 0.7}
    }, stream=True, timeout=120)
    for line in r.iter_lines():
        if not line: continue
        try:
            import json as _json
            obj = _json.loads(line.decode("utf-8"))
            ch = obj.get("message", {}).get("content", "")
            if ch: yield ch
            if obj.get("done"): break
        except: pass


def is_time_query(q):
    q = q.strip()
    time_patterns = ["现在几点", "今天几号", "当前时间", "现在时间", "现在日期", "今天星期", "现在星期"]
    if any(p in q for p in time_patterns):
        return True
    if "几点" in q and "上班" not in q and "下班" not in q and "考勤" not in q:
        return True
    return False


# 侧边栏
with st.sidebar:
    st.markdown(f"""
<div class="side-brand">
  <div class="logo sm">{LOGO_ICON}</div>
  <div>
    <div class="side-title">智能客服</div>
    <div class="side-sub">BERT 分类 × RAG 检索 × LLM 生成</div>
  </div>
</div>
""", unsafe_allow_html=True)
    st.toggle("🌙 深色模式", key="dark_toggle",
              value=(st.session_state.theme == "dark"), on_change=_sync_theme)
    st.markdown(f"""
<div class="arch">
<b>🖥️ 系统配置</b><br>
🧠 BERT · bert-base-chinese<br>
🔢 Embedding · bge-small-zh<br>
🦙 LLM · {OLLAMA_MODEL}<br>
🌐 地址 · http://{LAN_IP}:8501
</div>
""", unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="sec-title">📂 支持类别</div>', unsafe_allow_html=True)
    for label in LABELS:
        cn, desc = LABEL_CN[label]
        ico, ico_bg = CAT_ICON[label]
        st.markdown(f"""
<div class="cat-item">
  <div class="cat-ico" style="background:{ico_bg};">{ico}</div>
  <div>
    <div class="cat-name">{cn}</div>
    <div class="cat-desc">{desc}</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="header-bar">
  <div class="header-inner">
    <div class="logo">{LOGO_ICON}</div>
    <div>
      <div class="brand-title">企业智能客服</div>
      <div class="brand-sub">BERT 意图分类 · RAG 检索 · {OLLAMA_MODEL} 生成</div>
    </div>
    <div class="online">在线</div>
  </div>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "content": "您好！我是企业智能客服，请问有什么可以帮您？", "meta": None, "time": datetime.now().strftime("%H:%M")}
    ]

# 渲染历史消息（包含刚发的 user 消息，流式 bot 回复在下方 placeholder）
_rendering_unanswered = bool(st.session_state.messages and st.session_state.messages[-1]["role"] == "user")
_hist = st.session_state.messages  # 全部渲染，user 消息立即可见

parts = ['<div class="chat-area">']
for msg in _hist:
    if msg.get("time"):
        parts.append(f'<div class="time-divider">{escape(msg["time"])}</div>')
    content = escape(msg["content"])
    if msg["role"] == "user":
        parts.append(
            f'<div class="row user"><div class="msg-wrap"><div class="bubble">{content}</div></div>'
            f'<div class="avatar user">{USER_ICON}</div></div>')
    else:
        meta_html = ""
        if msg.get("meta"):
            tags = "".join(
                f'<span class="tag {cls}">{escape(t)}</span>'
                for t, cls in msg["meta"])
            meta_html = f'<div class="tags">{tags}</div>'
        parts.append(
            f'<div class="row bot"><div class="avatar bot">{BOT_ICON}</div>'
            f'<div class="msg-wrap"><div class="bubble">{content}</div>{meta_html}</div></div>')
parts.append('</div>')
st.markdown("".join(parts), unsafe_allow_html=True)


def render_bot_streaming(text, tags_html=""):
    """把流式内容渲染成 bot 气泡 HTML。"""
    return (f'<div class="chat-area"><div class="row bot">'
            f'<div class="avatar bot">{BOT_ICON}</div>'
            f'<div class="msg-wrap"><div class="bubble">{escape(text)}</div>{tags_html}</div></div></div>')


q = st.chat_input("请输入您的问题...")
if q:
    now = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({"role": "user", "content": q, "meta": None, "time": now})
    st.rerun()

# 处理未回答的用户消息（流式）
if _rendering_unanswered:
    last = st.session_state.messages[-1]
    placeholder = st.empty()

    if is_time_query(last["content"]):
        now = datetime.now().strftime("%Y年%m月%d日 %H:%M")
        weekdays = ["星期一","星期二","星期三","星期四","星期五","星期六","星期日"]
        wd = weekdays[datetime.now().weekday()]
        answer = f"当前时间是：{now} {wd}"
        tags = [("系统时间", "src")]
        full = answer
        placeholder.markdown(render_bot_streaming(full,
            '<div class="tags"><span class="tag src">系统时间</span></div>'),
            unsafe_allow_html=True)
    else:
        with st.spinner("思考中..."):
            probs = classify(last["content"])
            top3 = probs.argsort()[::-1][:3]

        if LABELS[top3[0]] == OTHER_INTENT or probs[top3[0]] < OTHER_THRESHOLD:
            tags = [("其他 · 闲聊", "first")]
            full = ""
            try:
                for chunk in stream_chat_llm(last["content"]):
                    full += chunk
                    placeholder.markdown(render_bot_streaming(full), unsafe_allow_html=True)
                if not full:
                    full = "这个问题我不太确定呢，可以换个问法，或问问我考勤、报销、IT 之类的业务问题~"
            except Exception as e:
                full = f"⚠️ LLM 调用失败：{e}"
        else:
            cat = LABELS[top3[0]]
            tags = [(f"{LABEL_CN[LABELS[i]][0]} {probs[i]:.0%}",
                     "first" if i == top3[0] else "") for i in top3]
            docs, metas = retrieve(cat, last["content"])
            full = ""
            srcs = []
            try:
                for chunk, src_labels in stream_ask_llm(last["content"], cat, docs, metas):
                    full += chunk
                    srcs = src_labels
                    tag_html = '<div class="tags">' + "".join(
                        f'<span class="tag {c}">{escape(t)}</span>'
                        for t, c in tags + [(f"📄 {s}", "src") for s in srcs]
                    ) + '</div>'
                    placeholder.markdown(render_bot_streaming(full, tag_html), unsafe_allow_html=True)
                if not full:
                    full = "抱歉，暂时无法生成回答，请稍后重试。"
                tags += [(f"📄 {s}", "src") for s in srcs]
            except Exception as e:
                full = f"⚠️ LLM 调用失败：{e}"

        placeholder.markdown(render_bot_streaming(full,
            '<div class="tags">' + "".join(
                f'<span class="tag {c}">{escape(t)}</span>' for t, c in tags) + '</div>'),
            unsafe_allow_html=True)

    st.session_state.messages.append({"role": "bot", "content": full, "meta": tags,
                                       "time": datetime.now().strftime("%H:%M")})
