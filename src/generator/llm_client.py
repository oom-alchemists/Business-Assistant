"""LLM 客户端封装：统一调用 OpenAI 兼容接口（vLLM 起的本地服务）。"""
from openai import OpenAI


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str,
                 max_tokens: int = 512, temperature: float = 0.3):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat(self, messages: list[dict], **kwargs) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
        )
        return resp.choices[0].message.content
