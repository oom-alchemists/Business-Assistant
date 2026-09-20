"""评估指标工具：意图分类准确率、检索 Recall@K、答案好评率统计。"""
from collections import defaultdict
from typing import List, Dict


def intent_accuracy(preds: List[str], golds: List[str]) -> Dict[str, float]:
    """意图分类准确率。"""
    assert len(preds) == len(golds)
    total = len(golds)
    correct = sum(p == g for p, g in zip(preds, golds))
    return {"total": total, "correct": correct, "accuracy": correct / max(total, 1)}


def recall_at_k(retrieved: List[List[str]], golds: List[str], k: int = 5) -> float:
    """检索 Recall@K：标准答案是否在召回的前 K 条 chunk 中。
    retrieved 中每个元素是该 query 命中的 chunk_id 列表；golds 为正确 chunk_id。
    这里简化为字符串包含判断，真实项目建议按 chunk_id 比对。
    """
    hit = 0
    for rec, gold in zip(retrieved, golds):
        if any(gold in chunk for chunk in rec[:k]):
            hit += 1
    return hit / max(len(golds), 1)


def feedback_stats(feedback: List[Dict]) -> Dict[str, float]:
    """用户反馈统计：[{"query":..., "rating": 1/-1}, ...]"""
    if not feedback:
        return {"total": 0, "good": 0, "bad": 0, "good_rate": 0.0}
    good = sum(1 for f in feedback if f.get("rating", 0) > 0)
    total = len(feedback)
    return {"total": total, "good": good, "bad": total - good,
            "good_rate": good / total}
