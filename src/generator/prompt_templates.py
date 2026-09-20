"""Prompt 模板：RAG 回答生成 + 闲聊回答。"""

# 强制模型"仅基于上下文回答"，并要求引用来源（精确到文件与段号）
QA_SYSTEM = """你是公司内部智能客服助手。请严格遵循以下规则回答员工问题：

1. 【仅基于给定上下文】回答必须完全基于下方提供的知识库片段。如果上下文没有答案，直接说"未在公司制度中找到相关内容"，禁止编造、禁止使用你自己的知识。
2. 【引用来源】回答末尾用【来源：文件名 第N段】标注出处，引用多个片段时全部列出。
3. 【简洁友好】用中文口语化回答，200 字以内，不要重复用户问题。
4. 【越权拒答】如果问题涉及他人薪资、公司机密、未公开数据，礼貌拒绝。

==================== 知识库上下文 ===================={context}
"""

# 其他/闲聊模式：不基于知识库，自由友好回答
CHAT_SYSTEM = """你是企业内部智能客服，当前为自由闲聊模式，用户的问题与公司制度、业务无关。
规则：
1. 友好自然地回答，80 字以内
2. 不要编造任何公司制度信息
3. 结尾可温和引导用户提问业务问题（考勤/报销/IT/人事等）"""


def build_qa_prompt(query: str, contexts: list[dict]) -> list[dict]:
    """组装对话消息，片段来源精确到 知识库/文件名 第N段。"""
    ctx_lines = []
    for i, c in enumerate(contexts, 1):
        m = c.get("metadata", {})
        src = m.get("source", c.get("kb", "?"))
        idx = m.get("chunk_index")
        loc = f"{src} 第{idx + 1}段" if isinstance(idx, int) else src
        ctx_lines.append(f"[片段{i} | 来源: {m.get('kb', '')}/{loc}]\n{c['text']}")
    ctx_text = "\n\n---\n\n".join(ctx_lines)
    return [
        {"role": "system", "content": QA_SYSTEM.format(context=ctx_text)},
        {"role": "user", "content": query},
    ]


def build_chat_prompt(query: str) -> list[dict]:
    """闲聊消息（不携带知识库上下文）。"""
    return [
        {"role": "system", "content": CHAT_SYSTEM},
        {"role": "user", "content": query},
    ]
