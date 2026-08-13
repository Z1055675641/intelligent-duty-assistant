# ============================================================
# LangGraph 图构建
# ============================================================
# 功能：构建完整的 LangGraph Agent（支持多路召回 + 查询重写）
# ============================================================

from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

import config
from .state import AgentState
from .nodes import analyze_question, execute_tools, generate_answer, rewrite_query_node
from .tools import create_rag_tool, get_weather, get_current_time
from ..rag.retriever import create_ensemble_retriever


def build_agent(faiss_retriever, bm25_retriever, reranker):
    """
    构建 LangGraph Agent（支持多路召回 + 查询重写）
    
    流程图：
        analyze（判断是否需要工具）
            ↓
        【条件边】根据 tool_calls 分支
            ├── 有 tool_calls → rewrite_query（查询重写）→ execute_tools（执行工具）
            │                                                ↓
            │                                             generate（生成答案）
            │                                                ↓
            └── 无 tool_calls → generate（直接生成答案）
                                  ↓
                                 END
    
    参数：
        faiss_retriever: FAISS 语义检索器
        bm25_retriever: BM25 关键词检索器
        reranker: Rerank 模型
    
    返回：
        编译好的 Agent
    """
    
    # ---------- 创建多路召回检索器 ----------
    ensemble_retriever = create_ensemble_retriever(faiss_retriever, bm25_retriever)
    
    # ---------- 创建所有工具 ----------
    # 两种检索工具：
    #   rag_tool       : 单查询用，候选数默认 TOP_K_FIRST
    #   rag_tool_multi : 多查询用，每个改写查询只取 QUERY_REWRITE_TOP_K 个候选
    rag_tool = create_rag_tool(ensemble_retriever, reranker)
    rag_tool_multi = create_rag_tool(ensemble_retriever, reranker, top_k=config.QUERY_REWRITE_TOP_K)
    weather_tool = get_weather
    time_tool = get_current_time
    tools = [rag_tool, weather_tool, time_tool]
    
    # ---------- 创建 LLM ----------
    # 注意：需要两个 LLM 实例
    # 1. llm_with_tools：用于 analyze 节点（需要工具绑定）
    # 2. llm_no_tools：用于 rewrite_query 节点（纯文本改写，不需要工具）
    llm_with_tools = ChatOpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.LLM_BASE_URL,
        model=config.LLM_MODEL,
        temperature=config.TEMPERATURE,
    ).bind_tools(tools)

    llm_no_tools = ChatOpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.LLM_BASE_URL,
        model=config.LLM_MODEL,
        temperature=config.QUERY_REWRITE_TEMPERATURE,  # 改写用低温度，更稳定
    )
    
    # ---------- 创建图 ----------
    graph = StateGraph(AgentState)
    
    # ---------- 添加节点 ----------
    graph.add_node(
        "analyze",
        lambda state: analyze_question(state, llm_with_tools)
    )
    
    # 🆕 新增查询重写节点
    graph.add_node(
        "rewrite_query",
        lambda state: rewrite_query_node(state, llm_no_tools)
    )
    
    graph.add_node(
        "execute_tools",
        lambda state: execute_tools(state, rag_tool, rag_tool_multi, weather_tool, time_tool)
    )
    
    graph.add_node(
        "generate",
        lambda state: generate_answer(state, llm_no_tools)
    )
    
    # ---------- 设置入口 ----------
    graph.set_entry_point("analyze")
    
    # ---------- 条件边：analyze 后的路由 ----------
    def route_after_analyze(state: AgentState) -> Literal["rewrite_query", "generate"]:
        """
        根据 analyze 的结果决定走哪条路
        
        返回：
            "rewrite_query": 需要调用工具 → 先改写查询，再执行工具
            "generate": 不需要工具 → 直接生成答案
        """
        messages = state.get("messages", [])
        if messages:
            last = messages[-1]
            if hasattr(last, 'tool_calls') and last.tool_calls:
                return "rewrite_query"  # ← 改：先去 rewrite_query，再去 execute_tools
        return "generate"
    
    graph.add_conditional_edges(
        "analyze",
        route_after_analyze,
        {
            "rewrite_query": "rewrite_query",  # 需要工具 → 先重写
            "generate": "generate"              # 不需要 → 直接生成
        }
    )
    
    # ---------- 普通边 ----------
    graph.add_edge("rewrite_query", "execute_tools")  # 🆕 重写 → 执行
    graph.add_edge("execute_tools", "generate")        # 执行 → 生成
    graph.add_edge("generate", END)                    # 生成 → 结束
    
    return graph.compile()