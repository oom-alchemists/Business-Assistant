"""第 1 层路由：规则关键词匹配。
命中即返回对应意图，零模型调用，毫秒级。
"""
from dataclasses import dataclass
from typing import Optional, Dict, List
import yaml


@dataclass
class RouteResult:
    intent: str                 # 意图名
    kb_dir: str                 # 目标知识库目录
    source: str                 # 命中来源: rule / bert / llm
    confidence: float           # 规则恒为 1.0
    matched_keyword: str = ""


class RuleRouter:
    def __init__(self, intents_config: str):
        with open(intents_config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        # 构建关键词 -> (intent, kb_dir) 倒排表
        self.kw_index: Dict[str, tuple] = {}
        self.intent_desc: Dict[str, str] = {}
        for it in cfg["intents"]:
            self.intent_desc[it["name"]] = it["desc"]
            for kw in it.get("keywords", []):
                self.kw_index[kw.lower()] = (it["name"], it["kb_dir"])

    def route(self, query: str) -> Optional[RouteResult]:
        """命中任一关键词即返回，按最长关键词优先（避免短词误命中）。"""
        q = query.lower().strip()
        if not q:
            return None
        # 按关键词长度降序，优先匹配更具体的词
        for kw in sorted(self.kw_index.keys(), key=len, reverse=True):
            if kw in q:
                intent, kb_dir = self.kw_index[kw]
                return RouteResult(intent=intent, kb_dir=kb_dir,
                                   source="rule", confidence=1.0,
                                   matched_keyword=kw)
        return None


if __name__ == "__main__":
    r = RuleRouter("config/intents.yaml")
    print(r.route("公司上下班时间是几点"))
    print(r.route("VPN 连不上"))
    print(r.route("报销发票怎么贴"))
