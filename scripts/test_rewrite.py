# ============================================================
# 查询重写功能测试脚本（直接调用 Agent，不走 Web 界面）
# 用法：python scripts/test_rewrite.py
# ============================================================

import sys
import os
import io

# 解决 Windows GBK 终端 emoji 打印问题（line_buffering=True：保证进度实时刷新）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

# 项目根目录加入路径
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 离线开关：本地默认离线；云端设 HF_OFFLINE=0 联网下载。
# 必须在所有 langchain/transformers 导入前设置；config.py 也会同步这些值。
os.environ.setdefault("HF_OFFLINE", "1")
os.environ["HF_HUB_OFFLINE"] = os.environ["HF_OFFLINE"]
os.environ["TRANSFORMERS_OFFLINE"] = os.environ["HF_OFFLINE"]
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")  # 尊重环境变量覆盖；默认官方源
print(f"[启动诊断] HF_OFFLINE={os.environ.get('HF_OFFLINE')} HF_ENDPOINT={os.environ.get('HF_ENDPOINT')} DEEPSEEK_API_KEY={'已设置' if os.environ.get('DEEPSEEK_API_KEY') else '未设置'}")

import config
from src.rag.retriever import load_retrieval_components
from src.agent.graph import build_agent


def main():
    print("1. 加载检索组件（FAISS + Rerank + BM25）...")
    _, faiss_retriever, bm25_retriever, reranker = load_retrieval_components()

    print("2. 构建 Agent...")
    agent = build_agent(faiss_retriever, bm25_retriever, reranker)

    # 测试问题：模糊、口语化，应触发 rag_search
    question = "值班时遇到错误码 DBS.200026 怎么处理？"

    state = {
        "question": question,
        "messages": [],
        "retrieved_docs": "",
        "sources": [],
        "answer": "",
        "prompt": "",
        "rewritten_queries": [],
    }

    print(f"\n3. 提问：{question}")
    result = agent.invoke(state)

    print("\n===== 测试结果 =====")
    print("改写后的查询：", result.get("rewritten_queries"))
    print("来源数量：", len(result.get("sources", [])))
    print("来源：", result.get("sources"))
    prompt = result.get("prompt", "")
    print("prompt 长度：", len(prompt))
    print("prompt 前 200 字：", prompt[:200])


if __name__ == "__main__":
    main()
