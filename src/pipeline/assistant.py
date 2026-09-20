"""主 Pipeline：串联 意图路由 → 检索 → 重排 → 生成 → 兜底。

调用顺序：
  1. RuleRouter    (关键词，0 成本)
  2. BertRouter    (置信度 > 阈值直接用)
  3. LLMRouter     (模糊问题兜底)
  4. VectorStore   (按路由结果检索；置信度低则跨库)
  5. Reranker
  6. LLMClient     (生成答案)
  7. Fallback      (无答案 → 转人工)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import yaml
from loguru import logger

from src.router import RuleRouter, BertRouter, LLMRouter
from src.retrieval import VectorStore, Reranker
from src.generator import LLMClient, build_qa_prompt, build_chat_prompt

OTHER_INTENT = "other_chitchat"   # 其他/闲聊：无知识库，不检索，LLM 自由回答


@dataclass
class AnswerResult:
    answer: str
    intent: str = ""
    route_source: str = ""
    confidence: float = 0.0
    contexts: List[Dict] = field(default_factory=list)
    need_human: bool = False


class AssistantPipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        # --- 意图路由 ---
        self.rule_router = RuleRouter("config/intents.yaml")

        # 加载意图描述给 LLM 路由用
        with open("config/intents.yaml", "r", encoding="utf-8") as f:
            intents_cfg = yaml.safe_load(f)
        self.intent_desc = {it["name"]: it["desc"] for it in intents_cfg["intents"]}
        # 意图 -> 知识库目录（仅业务类，other_chitchat 无知识库不参与检索）
        self.kb_map = {it["name"]: it["kb_dir"] for it in intents_cfg["intents"] if it.get("kb_dir")}
        self.all_kb_dirs = sorted(set(self.kb_map.values()))
        # BERT 分类头覆盖全部 9 类（含 other_chitchat），与 checkpoints/intent_bert 一致
        self.all_intents = [it["name"] for it in intents_cfg["intents"]]

        # BERT 路由（未训练则传空 ckpt，自动跳过）
        rt = self.cfg["router"]["bert"]
        self.bert_router = BertRouter(
            model_name=rt["model_name"],
            ckpt_path=rt.get("classifier_ckpt", ""),
            label_list=self.all_intents,
            kb_map=self.kb_map,
            device=rt.get("device", "cuda"),
            confidence_threshold=rt["confidence_threshold"],
        ) if rt["enabled"] else None

        # LLM 客户端（同时给路由和生成用）
        gen = self.cfg["generator"]
        g = gen[gen["backend"]]
        self.llm = LLMClient(
            base_url=g["base_url"], api_key=g.get("api_key", "EMPTY"),
            model=g["model"], max_tokens=g.get("max_tokens", 512),
            temperature=g.get("temperature", 0.3),
        )
        self.llm_router = LLMRouter(
            client=self.llm.client, model=g["model"],
            intents=self.intent_desc,
            confidence_threshold=self.cfg["router"]["llm"]["confidence_threshold"],
            temperature=self.cfg["router"]["llm"]["temperature"],
            kb_map=self.kb_map,
        )

        # --- 检索 ---
        rs = self.cfg["retrieval"]
        self.vs = VectorStore(
            persist_dir=self.cfg["paths"]["vector_db"],
            embedding_model=rs["embedding_model"],
            ollama_base_url=self.cfg["generator"]["openai_compatible"]["base_url"],
        )
        self.reranker = Reranker(rs["rerank"]["model"]) if rs["rerank"]["enabled"] else None
        self.top_k = rs["top_k"]
        self.sim_threshold = rs["similarity_threshold"]
        self.rerank_top_n = rs["rerank"]["top_n"]

        # --- 兜底 ---
        self.fb = self.cfg["fallback"]
        logger.info("AssistantPipeline 初始化完成")

    # ---------- 1. 意图路由 ----------
    def _route(self, query: str):
        # 第 1 层：规则
        r = self.rule_router.route(query)
        if r:
            logger.debug(f"[路由] 规则命中: {r.intent} (kw={r.matched_keyword})")
            return r
        # 第 2 层：BERT
        if self.bert_router:
            r = self.bert_router.route(query)
            if r:
                logger.debug(f"[路由] BERT 命中: {r.intent} conf={r.confidence:.2f}")
                return r
        # 第 3 层：LLM
        r = self.llm_router.route(query)
        if r:
            logger.debug(f"[路由] LLM 命中: {r.intent} conf={r.confidence:.2f}")
        return r

    # ---------- 2. 检索 ----------
    def _retrieve(self, query: str, route) -> List[Dict]:
        if route is None or not route.kb_dir:
            # 无路由结果，全库搜
            return self.vs.search_all(query, self.all_kb_dirs, top_k=self.top_k)

        hits = self.vs.search(route.kb_dir, query, top_k=self.top_k)
        # 单库最高分低于阈值 → 跨库回退
        if self.fb["cross_library_search"] and (not hits or hits[0]["score"] < self.sim_threshold):
            logger.debug(f"[检索] 单库最高分 {hits[0]['score'] if hits else 0:.2f} < {self.sim_threshold}, 跨库回退")
            hits = self.vs.search_all(query, self.all_kb_dirs, top_k=self.top_k)
        return hits

    # ---------- 3. 生成 ----------
    def answer(self, query: str) -> AnswerResult:
        route = self._route(query)

        # out_of_scope 直接拒答
        if route and route.intent == "out_of_scope":
            return AnswerResult(
                answer="抱歉，您的问题涉及公司内部信息，我无法回答，已为您转接人工。",
                intent="out_of_scope", route_source="llm",
                need_human=self.fb["transfer_to_human"],
            )

        # 其他/闲聊：不检索知识库，LLM 自由回答
        if route and route.intent == OTHER_INTENT:
            return AnswerResult(
                answer=self.llm.chat(build_chat_prompt(query)),
                intent=OTHER_INTENT,
                route_source=route.source,
                confidence=route.confidence,
            )

        hits = self._retrieve(query, route)
        if not hits or hits[0]["score"] < self.sim_threshold:
            return AnswerResult(
                answer=self.fb["human_reply"],
                intent=route.intent if route else "",
                route_source=route.source if route else "",
                confidence=route.confidence if route else 0.0,
                need_human=self.fb["transfer_to_human"],
            )

        # rerank
        if self.reranker:
            hits = self.reranker.rerank(query, hits, top_n=self.rerank_top_n)

        # 生成
        messages = build_qa_prompt(query, hits)
        ans = self.llm.chat(messages)
        return AnswerResult(
            answer=ans,
            intent=route.intent if route else "",
            route_source=route.source if route else "",
            confidence=route.confidence if route else 0.0,
            contexts=hits,
        )
