"""FastAPI 接口层（并发版）：复用 ChatEngine，与 Streamlit UI 行为一致。

启动（多 worker 并发）：
    uvicorn app.api:app --host 0.0.0.0 --port 8001 --workers 2

说明：worker 数取 2。分类/检索模型在 CPU 推理（毫秒级），GPU 只服务 qwen3.5:9b，
多 worker 仅提升请求并发处理能力；LLM 生成瓶颈在 Ollama 单卡，2 worker 已足够。

接口：
    GET  /                → 微信风格聊天页面
    POST /api/chat        → 非流式问答（JSON）
    POST /api/chat/stream → 流式问答（SSE，text/event-stream）
    GET  /api/health      → 健康检查
"""
import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.chat_engine import get_engine

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="企业智能客服 API", version="2.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    query: str


@app.on_event("startup")
def startup():
    get_engine()  # 预热：加载 BERT/Embedding/Chroma，避免首个请求慢


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/chat")
def chat(req: ChatRequest):
    """非流式问答：多用户并发（sync def 自动跑在线程池）。"""
    try:
        res = get_engine().answer(req.query)
        res["ok"] = True
        return res
    except Exception as e:
        return {"ok": False, "error": str(e), "answer": f"⚠️ 服务异常：{e}"}


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    """流式问答：SSE 逐段推送（NDJSON 行）。"""
    def gen():
        try:
            for delta, tags, sources, done in get_engine().stream_answer(req.query):
                yield json.dumps({"delta": delta, "tags": tags,
                                  "sources": sources, "done": done},
                                 ensure_ascii=False) + "\n"
        except Exception as e:
            yield json.dumps({"delta": f"⚠️ 服务异常：{e}", "tags": [],
                              "sources": [], "done": True},
                             ensure_ascii=False) + "\n"
    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "enterprise-assistant", "version": "2.0.0"}
