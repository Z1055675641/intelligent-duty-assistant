# ============================================================
# 检索器 + Rerank
# ============================================================
# 功能：
#   1. 加载 FAISS 向量库
#   2. 执行 FAISS 粗筛 + Rerank 精排
#   3. 格式化检索结果
# ============================================================

import os
import re
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
# EnsembleRetriever 位置因 langchain 版本不同而异
# langchain 0.3.x: langchain.retrievers
# langchain 1.x:   langchain_classic.retrievers
try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:
    from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

import config
from .embeddings import get_embeddings, probe_and_report
from .loader import load_all_documents
from .splitter import get_text_splitter


def load_vectorstore():
    """
    从本地加载 FAISS 向量数据库
    
    返回：
        vector_store: FAISS 向量库对象
    """
    print("📂 正在从本地加载嵌入模型...")
    embeddings = get_embeddings()
    print("✅ 嵌入模型加载完成")
    
    if not os.path.exists(config.INDEX_SAVE_PATH):
        print(f"❌ 未找到向量库：{config.INDEX_SAVE_PATH}")
        raise FileNotFoundError(f"未找到向量库：{config.INDEX_SAVE_PATH}")
    
    print(f"📂 正在加载向量库：{config.INDEX_SAVE_PATH}")
    vector_store = FAISS.load_local(
        str(config.INDEX_SAVE_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"✅ 向量库加载完成，共 {vector_store.index.ntotal} 个向量")
    return vector_store

def chinese_tokenizer(text: str) -> List[str]:
    """
    BM25 中文分词预处理函数。

    langchain_community 的 BM25Retriever 默认用 text.split()（按空格分词），
    对无空格的中文整句会退化为"整句一个词"，关键词检索完全失效。
    这里用 jieba 做中文分词，并额外保留错误码/参数名这类字母数字串（如 DBS.200026）。
    """
    tokens: List[str] = []
    try:
        import jieba
        tokens = [t for t in jieba.cut_for_search(text) if t.strip()]
    except ImportError:
        tokens = text.split()
    # 补充保留整段代码串，提升错误码等精确关键词的匹配能力
    tokens += re.findall(r'[A-Za-z0-9_.\-]+', text)
    return tokens


def create_bm25_retriever(chunks):
    """
    从文档块创建 BM25 关键词检索器

    参数：
        chunks: 文档块列表（Document 对象列表）

    返回：
        bm25_retriever: BM25 检索器对象

    BM25 原理：
        - 基于词频（TF）和逆文档频率（IDF）计算相关性
        - 适合精确关键词匹配（如错误码 "DBS.200026"）
        - 与语义检索互补
    """
    # 使用 LangChain 的 BM25Retriever，并显式传入中文分词预处理函数
    # （否则默认按空格分词，中文检索会失效）
    bm25_retriever = BM25Retriever.from_documents(
        chunks,
        preprocess_func=chinese_tokenizer,
    )

    # 设置返回的文档数量
    bm25_retriever.k = config.BM25_K

    return bm25_retriever

def create_ensemble_retriever(faiss_retriever, bm25_retriever):
    """
    创建多路召回检索器（FAISS + BM25）
    
    参数：
        faiss_retriever: FAISS 语义检索器
        bm25_retriever: BM25 关键词检索器
    
    返回：
        ensemble_retriever: 合并后的检索器
    
    多路召回原理：
        1. 同时用两种方式检索
        2. 合并结果，去重
        3. 按权重重新排序
    """
    # 创建合并检索器
    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=config.ENSEMBLE_WEIGHTS
    )
    
    return ensemble_retriever

def load_reranker():
    """
    加载 Rerank 模型

    返回：
        reranker: CrossEncoder 实例

    说明：
        local_files_only 跟随离线开关 config.HF_OFFLINE：
        - 本机默认离线（HF_OFFLINE=1）：只从本地缓存加载
        - 云端部署（HF_OFFLINE=0）：允许联网下载模型
    """
    print("📂 正在加载 Rerank 模型...")
    probe_and_report(config.RERANK_MODEL)
    try:
        reranker = CrossEncoder(config.RERANK_MODEL, local_files_only=(config.HF_OFFLINE == "1"))
        print("✅ Rerank 模型加载完成")
        return reranker
    except Exception as e:
        import traceback
        print(f"❌ Rerank 模型加载失败：{type(e).__name__}: {e!r}")
        traceback.print_exc()
        raise


def retrieve_with_rerank(query, retriever, reranker, top_k=None):
    """
    完整的检索流程：多路召回 → Rerank 精排

    参数：
        query: 用户问题
        retriever: 检索器（可以是 FAISS、BM25 或 EnsembleRetriever）
        reranker: Rerank 模型
        top_k: 参与精排的候选数上限（None = 使用全局 TOP_K_FIRST）

    返回：
        最相关的 top_k_final 个文档
    """
    # 第1步：检索（多路或单路召回）
    candidates = retriever.invoke(query)

    # 候选数上限：超过则截断（如多查询场景，每个查询只取 QUERY_REWRITE_TOP_K 个）
    if top_k and len(candidates) > top_k:
        candidates = candidates[:top_k]

    if len(candidates) <= config.TOP_K_FINAL:
        return candidates
    
    # 第2步：Rerank 精排
    pairs = [[query, doc.page_content] for doc in candidates]
    scores = reranker.predict(pairs)
    
    # 按得分排序，取前 top_k_final 个
    sorted_pairs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in sorted_pairs[:config.TOP_K_FINAL]]


def format_docs(docs):
    """
    把文档列表拼接成上下文字符串

    参数：
        docs: Document 列表

    返回：
        拼接后的字符串（用 --- 分隔）
    """
    return "\n\n---\n\n".join([doc.page_content for doc in docs])


def load_retrieval_components():
    """
    一键加载整套检索组件（多个入口共用，避免重复代码）。

    流程：加载 FAISS 向量库 → 创建语义检索器 → 加载 Rerank 模型 → 加载文档构建 BM25。
    各入口（Streamlit 界面 / FastAPI / 测试脚本）只需调用本函数并自行 build_agent。

    返回：
        (vector_store, faiss_retriever, bm25_retriever, reranker)
    """
    vector_store = load_vectorstore()
    reranker = load_reranker()
    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": config.TOP_K_FIRST})
    documents = load_all_documents(str(config.DOCS_DIR))
    chunks = get_text_splitter().split_documents(documents)
    bm25_retriever = create_bm25_retriever(chunks)
    print(f"✅ 检索组件加载完成，BM25 共 {len(chunks)} 个文本块")
    return vector_store, faiss_retriever, bm25_retriever, reranker