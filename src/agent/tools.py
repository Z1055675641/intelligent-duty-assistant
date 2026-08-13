# ============================================================
# Agent 工具定义
# ============================================================
# 功能：定义 Agent 可调用的工具（RAG检索 + 天气 + 时间）
# ============================================================

import os
import requests
from datetime import datetime
from langchain_core.tools import tool

import config
from ..rag.retriever import retrieve_with_rerank, format_docs


def create_rag_tool(retriever, reranker, top_k=None):
    """
    创建 RAG 检索工具

    参数：
        retriever: FAISS 检索器
        reranker: Rerank 模型
        top_k: 每次检索参与精排的候选数上限（None = 用全局 TOP_K_FIRST）
              多查询场景传 QUERY_REWRITE_TOP_K，每个改写查询只取更少的候选

    返回：
        rag_search: 工具函数
    """
    # 候选数上限在创建时闭包绑定，工具函数签名保持只有 query（不给 LLM 暴露多余参数）
    if top_k is None:
        top_k = config.TOP_K_FIRST

    @tool
    def rag_search(query: str) -> str:
        """
        从文档知识库中检索相关信息。
        
        适用场景：
            - 用户询问华为云产品（如 GaussDB）错误码的含义与处理
            - 用户询问日志路径、日志文件位置
            - 用户询问操作指南、配置方法、备份策略等
            - 用户询问华为云文档中的任何具体内容
        
        参数：
            query: 用户的问题或检索关键词
        
        返回：
            检索到的相关文档内容（包含来源信息）
        """
        docs = retrieve_with_rerank(query, retriever, reranker, top_k=top_k)
        
        if not docs:
            return "未找到相关信息。"
        
        # 为每个文档标注来源
        context_parts = []
        all_sources = []
        
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "未知")
            page = doc.metadata.get("page", "N/A")
            source_str = f"{os.path.basename(source)}（第{page}页）"
            all_sources.append(source_str)
            context_parts.append(f"【文档{i+1} 来源：{source_str}】\n{doc.page_content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        result = f"找到 {len(docs)} 个相关文档：\n\n{context}\n\n【所有来源】\n"
        for s in set(all_sources):
            result += f"  - {s}\n"
        
        return result
    
    return rag_search


# ---------- 天气工具 ----------
@tool
def get_weather(city: str) -> str:
    """
    查询指定城市的实时天气。
    
    适用场景：
        - 用户询问某个城市的天气
        - 用户问"今天热不热"、"要不要带伞"
    
    参数：
        city: 城市名称，如"上海"、"北京"
    
    返回：
        天气信息字符串
    """
    try:
        url = f"https://wttr.in/{city}?format=%C+%t&lang=zh"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return f"{city}天气：{response.text.strip()}"
        else:
            return f"{city}天气：多云，22-28°C（模拟数据）"
    except:
        return f"{city}天气：多云，22-28°C（模拟数据）"


# ---------- 时间工具 ----------
@tool
def get_current_time() -> str:
    """
    获取当前日期和时间。
    
    适用场景：
        - 用户问"今天几号"
        - 用户问"现在几点"
    
    返回：
        日期时间字符串
    """
    now = datetime.now()
    return f"现在是 {now.strftime('%Y年%m月%d日 %H:%M:%S')}"