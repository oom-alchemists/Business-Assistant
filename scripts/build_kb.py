"""构建知识库向量索引：扫描 data/knowledge_base 下所有 .md 文件，按段落切块，写入 Chroma。
用法：python scripts/build_kb.py
"""
import sys
from pathlib import Path

# 把项目根目录加入 sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
from src.retrieval.vector_store import VectorStore
from src.utils.logger import setup_logger

logger = setup_logger()


def split_markdown(text: str, chunk_size: int = 300, overlap: int = 50):
    """简单按段落切块：按 \n\n 分块，超长再按字符切。"""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) < chunk_size:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p[-overlap:] + p
    if buf:
        chunks.append(buf)
    return chunks


def main():
    with open(ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    kb_root = ROOT / cfg["paths"]["kb_root"]
    vs = VectorStore(
        persist_dir=str(ROOT / cfg["paths"]["vector_db"]),
        embedding_model=cfg["retrieval"]["embedding_model"],
        ollama_base_url=cfg["generator"]["openai_compatible"]["base_url"],
    )

    # 每个子目录对应一个 collection
    for sub in sorted(kb_root.iterdir()):
        if not sub.is_dir():
            continue
        collection = sub.name
        chunks = []
        for md in sub.glob("*.md"):
            text = md.read_text(encoding="utf-8")
            for i, c in enumerate(split_markdown(text)):
                chunks.append({
                    "id": f"{md.stem}_{i}",
                    "text": c,
                    "metadata": {"source": md.name, "kb": collection, "chunk_index": i},
                })
        vs.add_documents(collection, chunks)
        logger.info(f"[KB] {collection}: 写入 {len(chunks)} 个 chunk")

    logger.info("知识库构建完成 ✅")


if __name__ == "__main__":
    main()
