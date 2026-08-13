# ============================================================
# Agent State 定义
# ============================================================
# 功能：定义 LangGraph 的状态结构
# ============================================================

from typing import TypedDict, List, Dict, Any


class AgentState(TypedDict):
    """
    Agent 的状态：在节点之间传递的数据
    
    字段说明：
        question: 当前用户的问题（字符串）
        messages: 完整的对话历史（列表），用于多轮对话记忆
        retrieved_docs: 检索到的文档内容（字符串）
        sources: 文档来源列表（List[str]）
        answer: 最终答案（字符串）
        prompt: 传给 UI 层做流式生成的 prompt
        rewritten_queries: 查询重写生成的多个检索查询（List[str]）
    """
    question: str
    messages: List[Dict[str, Any]]
    retrieved_docs: str
    sources: List[str]
    answer: str
    prompt: str  # 传给 UI 层做流式生成的 prompt
    rewritten_queries: List[str]  # 🆕 查询重写生成的多个检索查询