"""第 3 层路由：LLM 兜底意图理解。
用 Qwen3-8B 做意图分类，强制 JSON 输出。
"""
import json
from typing import Optional, Dict
from openai import OpenAI

from .rule_router import RouteResult


SYSTEM_PROMPT = """你是企业内部客服的意图识别模块。根据用户问题，从下面意图列表中选择最合适的一个，并输出严格 JSON。

可选意图（name: desc）：
{intent_list}

规则：
1. 只输出 JSON，不要任何多余文字。
2. 格式：{{"intent": "意图name", "confidence": 0.0~1.0, "needs_human": true/false}}
3. 若问题涉及越权/机密（如询问他人工资、公司财报），intent 填 "out_of_scope"；与业务无关的普通闲聊/问候，intent 填 "other_chitchat"。
4. confidence < 0.6 表示模糊，系统会走多库并行检索。
"""


class LLMRouter:
    def __init__(self,
                 client: OpenAI,
                 model: str,
                 intents: Dict[str, str],
                 confidence_threshold: float = 0.60,
                 temperature: float = 0.1,
                 kb_map: Optional[Dict[str, str]] = None):
        self.client = client
        self.model = model
        self.intents = intents          # {name: desc}
        self.conf_threshold = confidence_threshold
        self.temperature = temperature
        # 由上层（intents.yaml）注入，未注入时退化为仅识别越权/未知
        self.kb_map = kb_map or {}

    def route(self, query: str) -> Optional[RouteResult]:
        intent_list = "\n".join(f"- {k}: {v}" for k, v in self.intents.items())
        sys = SYSTEM_PROMPT.format(intent_list=intent_list)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": query},
                ],
                temperature=self.temperature,
                max_tokens=128,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
        except Exception as e:
            print(f"[LLMRouter] 解析失败: {e}")
            return None

        intent = data.get("intent", "")
        conf = float(data.get("confidence", 0.0))
        if intent == "out_of_scope" or intent not in self.kb_map:
            return RouteResult(intent=intent or "unknown", kb_dir="",
                               source="llm", confidence=conf)
        return RouteResult(intent=intent, kb_dir=self.kb_map[intent],
                           source="llm", confidence=conf)
