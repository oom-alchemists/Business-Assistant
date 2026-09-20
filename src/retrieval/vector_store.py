"""向量检索模块：基于 Chroma + Ollama embedding API。
每个知识库子目录对应一个 Chroma collection。
embedding 通过 Ollama OpenAI 兼容接口调用，无需本地加载模型。
"""
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from openai import OpenAI


class OllamaEmbedding:
    """通过 Ollama /v1/embeddings 接口计算向量。"""
    def __init__(self, base_url: str = "http://127.0.0.1:11434/v1",
                 model: str = "bge-m3"):
        self.client = OpenAI(base_url=base_url, api_key="ollama")
        self.model = model

    def encode(self, texts: List[str]) -> List[List[float]]:
        """批量编码。"""
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def encode_query(self, text: str) -> List[float]:
        return self.encode([text])[0]


class VectorStore:
    def __init__(self, persist_dir: str, embedding_model: str = "bge-m3",
                 ollama_base_url: str = "http://127.0.0.1:11434/v1"):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.encoder = OllamaEmbedding(ollama_base_url, embedding_model)

    def add_documents(self, collection_name: str, chunks: List[Dict]) -> None:
        """写入 chunks: [{"id":..., "text":..., "metadata":{...}}]"""
        col = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        if chunks:
            embeddings = self.encoder.encode([c["text"] for c in chunks])
            col.upsert(
                ids=[c["id"] for c in chunks],
                documents=[c["text"] for c in chunks],
                embeddings=embeddings,
                metadatas=[c.get("metadata", {}) for c in chunks],
            )

    def search(self, collection_name: str, query: str,
               top_k: int = 5) -> List[Dict]:
        """在指定库检索 Top-K。"""
        try:
            col = self.client.get_collection(collection_name)
        except Exception:
            return []
        q_emb = self.encoder.encode_query(query)
        res = col.query(query_embeddings=[q_emb], n_results=top_k)
        hits = []
        for doc, dist, meta in zip(res["documents"][0],
                                    res["distances"][0],
                                    res["metadatas"][0]):
            hits.append({
                "text": doc,
                "score": 1 - dist,
                "metadata": meta,
                "kb": collection_name,
            })
        return hits

    def search_all(self, query: str, kb_dirs: List[str],
                   top_k: int = 5) -> List[Dict]:
        """跨库并行检索（用于单库置信度低时的回退）。"""
        merged = []
        for kb in kb_dirs:
            merged.extend(self.search(kb, query, top_k=top_k))
        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged[:top_k * 2]
