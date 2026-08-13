# ============================================================
# FastAPI 服务 - 将 LangGraph RAG Agent 封装成 API
# 基于 src/ 核心库
# 功能：
#   1. 启动时加载 RAG Agent（只加载一次）
#   2. 提供 /ask 接口，接收问题，返回答案
#   3. 提供 /health 接口，检查服务状态
#   4. 提供 /docs 接口，查看 API 文档
# ============================================================
#运行命令   python 工程化和部署/api.py

import os
import sys
import io
from typing import List

# 把项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 离线开关：本地默认离线；云端设 HF_OFFLINE=0 联网下载。
# 必须在所有 langchain/transformers 导入前设置；config.py 也会同步这些值。
os.environ.setdefault("HF_OFFLINE", "1")
os.environ["HF_HUB_OFFLINE"] = os.environ["HF_OFFLINE"]
os.environ["TRANSFORMERS_OFFLINE"] = os.environ["HF_OFFLINE"]
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")  # 尊重环境变量覆盖；默认官方源
print(f"[启动诊断] HF_OFFLINE={os.environ.get('HF_OFFLINE')} HF_ENDPOINT={os.environ.get('HF_ENDPOINT')} DEEPSEEK_API_KEY={'已设置' if os.environ.get('DEEPSEEK_API_KEY') else '未设置'}")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from langchain_openai import ChatOpenAI

import config
from src.rag.retriever import load_retrieval_components
from src.agent.state import AgentState
from src.agent.graph import build_agent

# ---------- 强制 UTF-8 编码 ----------
# line_buffering=True：保证服务日志实时刷新（否则 stdout 块缓冲，日志会憋着不显示）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)
os.environ["PYTHONIOENCODING"] = "utf-8"


# ============================================================
# 第1部分：FastAPI 应用
# ============================================================

# ---------- 定义请求/响应模型 ----------
class AskRequest(BaseModel):
    """API 请求格式"""
    question: str


class AskResponse(BaseModel):
    """API 响应格式"""
    answer: str
    sources: List[str]
    used_tool: bool


# ---------- 创建 FastAPI 应用 ----------
VERSION = "2.0.0"

app = FastAPI(
    title="RAG Agent API",
    description="基于 LangGraph 的 RAG 智能问答系统 API",
    version=VERSION
)

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- 启动时加载 Agent ----------
agent = None


@app.on_event("startup")
async def startup_event():
    """服务启动时加载 RAG Agent（只加载一次）"""
    global agent

    print("=" * 60)
    print("🚀 正在启动 RAG Agent API 服务...")
    print("=" * 60)

    # 一键加载检索组件（FAISS + Rerank + BM25）
    vector_store, faiss_retriever, bm25_retriever, reranker = load_retrieval_components()

    # 构建 Agent（多路召回：FAISS + BM25）
    agent = build_agent(faiss_retriever, bm25_retriever, reranker)

    print("=" * 60)
    print("✅ RAG Agent API 服务启动成功！")
    print(f"📁 知识库：{vector_store.index.ntotal} 个向量")
    print("🌐 访问 http://localhost:8000/docs 查看 API 文档")
    print("=" * 60)


# ---------- API 端点 ----------
@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "RAG Agent API",
        "version": VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy", "agent_loaded": agent is not None}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    问答接口

    接收用户问题，返回 RAG Agent 的回答和引用来源。

    示例：
        POST /ask
        {
            "question": "错误码 DBS.200026 表示什么？怎么处理？"
        }

        返回：
        {
            "answer": "根据参考资料，DBS.200026 表示数据库连接超时...（含错误描述、可能原因、解决建议）",
            "sources": ["GaussDB V2.0-25.860.0 参考指南(...).pdf"],
            "used_tool": true
        }
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        # 初始化 State
        initial_state: AgentState = {
            "question": request.question,
            "messages": [],
            "retrieved_docs": "",
            "sources": [],
            "answer": "",
            "prompt": "",
            "rewritten_queries": [],
        }

        # 执行 Agent
        result = agent.invoke(initial_state)

        # 关键修复：走工具路径时 Agent 只生成 prompt，需在此调用 LLM 补全答案
        answer = result.get("answer") or ""
        if not answer and result.get("prompt"):
            llm = ChatOpenAI(
                api_key=config.DEEPSEEK_API_KEY,
                base_url=config.LLM_BASE_URL,
                model=config.LLM_MODEL,
                temperature=config.TEMPERATURE,
            )
            response = llm.invoke(result["prompt"])
            answer = response.content if hasattr(response, "content") else str(response)

        # 返回结果
        return AskResponse(
            answer=answer,
            sources=result.get("sources", []),
            used_tool=bool(result.get("retrieved_docs"))
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时出错：{str(e)}")


# ============================================================
# 第2部分：程序入口
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
    )
