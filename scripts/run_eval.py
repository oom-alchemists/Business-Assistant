"""跑评估集：读取 data/eval/test_queries.jsonl，批量跑 pipeline，统计意图准确率与延迟。
用法：python scripts/run_eval.py
"""
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import AssistantPipeline
from src.utils.metrics import intent_accuracy
from src.utils.logger import setup_logger

logger = setup_logger()


def main():
    bot = AssistantPipeline("config/config.yaml")
    queries = [json.loads(l) for l in
               open(ROOT / "data/eval/test_queries.jsonl", encoding="utf-8")]

    preds, golds, lats = [], [], []
    for q in queries:
        t0 = time.time()
        res = bot.answer(q["query"])
        lat = (time.time() - t0) * 1000
        lats.append(lat)
        preds.append(res.intent)
        golds.append(q["intent"])
        logger.info(f"Q: {q['query']} | pred={res.intent} gold={q['intent']} | {lat:.0f}ms")

    acc = intent_accuracy(preds, golds)
    avg_lat = sum(lats) / len(lats)
    logger.info(f"===== 评估结果 =====")
    logger.info(f"意图准确率: {acc['accuracy']:.2%} ({acc['correct']}/{acc['total']})")
    logger.info(f"平均延迟: {avg_lat:.0f} ms")


if __name__ == "__main__":
    main()
