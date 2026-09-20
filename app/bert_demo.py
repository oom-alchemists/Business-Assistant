"""BERT 意图分类 Demo（Streamlit）
启动: streamlit run app/bert_demo.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 4 个意图
LABELS = ["policy_attendance", "it_vpn", "finance_expense", "hr_onboarding"]
LABEL_DESC = {
    "policy_attendance": "考勤/上下班/请假/年假",
    "it_vpn": "IT/VPN/密码/设备",
    "finance_expense": "财务/报销/发票/工资",
    "hr_onboarding": "HR/入职/离职/社保",
}
CKPT = ROOT / "checkpoints" / "intent_bert" / "best.pt"
MODEL_DIR = ROOT / "models" / "bert-base-chinese"

st.set_page_config(page_title="BERT 意图分类 Demo", page_icon="🏷️", layout="centered")
st.title("🏷️ BERT 企业客服意图分类")
st.caption("bert-base-chinese 微调 · 4 类意图 · 本地 GPU 推理")


@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), num_labels=len(LABELS))
    state = torch.load(str(CKPT), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return tok, model, device


tok, model, device = load_model()

query = st.text_input("请输入员工问题，例如：公司上下班时间是几点？", "公司上下班时间是几点？")

if st.button("开始分类", type="primary"):
    if not query.strip():
        st.warning("请输入问题")
    else:
        enc = tok(query, return_tensors="pt", truncation=True, max_length=64, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

        pred_idx = int(probs.argmax())
        pred_label = LABELS[pred_idx]
        pred_conf = float(probs[pred_idx])

        st.divider()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("预测类别", pred_label)
        with col2:
            st.metric("置信度", f"{pred_conf:.1%}")

        st.write("**类别说明**：", LABEL_DESC[pred_label])

        st.write("**各类别置信度分布**：")
        chart_data = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
        st.bar_chart(chart_data, use_container_width=True)

        # 详细概率表
        st.write("**详细概率**：")
        for i, label in enumerate(LABELS):
            st.write(f"- `{label}`: {probs[i]:.2%}")
