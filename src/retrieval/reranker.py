"""重排模块：bge-reranker 对初筛结果精排。"""
from typing import List, Dict
from FlagEmbedding import FlagReranker


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.reranker = FlagReranker(model_name, use_fp16=True)

    def rerank(self, query: str, candidates: List[Dict],
               top_n: int = 3) -> List[Dict]:
        """对 candidates 按相关性重排，保留 top_n。"""
        if not candidates:
            return []
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.reranker.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            scores = [scores]
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_n]
