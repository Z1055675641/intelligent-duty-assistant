# ============================================================
# Streamlit Web 界面
# ============================================================
# 功能：提供 Web 界面与 LangGraph RAG Agent 交互
# ============================================================
# 运行命令	streamlit run src/ui/streamlit_app.py
# 浏览器打开 http://localhost:8501

import sys
import os

# 修复：不要在 Streamlit 里用 io.TextIOWrapper 替换 sys.stdout/stderr。
# Streamlit 每次 rerun 都会从头重新执行本脚本；每次新建 wrapper 后，
# 旧的 wrapper 被 CPython 回收时会 close 掉底层 buffer，导致后续
# print()（如 agent 检索节点）抛 "ValueError: I/O operation on closed file"。
# 改为原位 reconfigure：不创建新对象、不触发旧对象回收；
# errors='replace' 保证 emoji 在 GBK 控制台也不抛 UnicodeEncodeError。
try:
    sys.stdout.reconfigure(errors='replace')
    sys.stderr.reconfigure(errors='replace')
except (AttributeError, ValueError, OSError):
    pass  # 非 TextIOWrapper 流则跳过，不影响使用

# 把项目根目录加入 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 离线开关：本地默认离线（用本地缓存）；云端 Secrets 注入 HF_OFFLINE=0 即联网下载。
# 必须在所有 langchain/transformers 导入前设置；config.py 也会同步这些值。
os.environ.setdefault("HF_OFFLINE", "1")
os.environ["HF_HUB_OFFLINE"] = os.environ["HF_OFFLINE"]
os.environ["TRANSFORMERS_OFFLINE"] = os.environ["HF_OFFLINE"]
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")  # 尊重 Secrets 覆盖；默认官方源（境外服务器连不上 hf-mirror）
# 启动诊断：Logs 里可一眼确认运行环境（云端应显示 HF_OFFLINE=0 HF_ENDPOINT=https://huggingface.co）
print(f"[启动诊断] HF_OFFLINE={os.environ.get('HF_OFFLINE')} HF_ENDPOINT={os.environ.get('HF_ENDPOINT')} DEEPSEEK_API_KEY={'已设置' if os.environ.get('DEEPSEEK_API_KEY') else '未设置'}")

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
import streamlit as st

import config
from src.rag.retriever import load_retrieval_components
from src.agent.graph import build_agent


def init_session_state():
    """初始化 Streamlit 会话状态"""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "agent_state" not in st.session_state:
        st.session_state["agent_state"] = {
            "question": "",
            "messages": [],
            "retrieved_docs": "",
            "sources": [],
            "answer": "",
            "prompt": "",
            "rewritten_queries": [],
        }


def load_rag_system():
    """
    加载 RAG 系统（支持多路召回：FAISS + BM25）

    返回：
        agent: LangGraph Agent 实例
        vector_store: FAISS 向量库对象
    """
    with st.spinner("🚀 正在加载 RAG 系统..."):
        # 一键加载检索组件（FAISS + Rerank + BM25）
        vector_store, faiss_retriever, bm25_retriever, reranker = load_retrieval_components()

        # 构建 Agent（传入两个检索器，内部创建多路召回）
        agent = build_agent(faiss_retriever, bm25_retriever, reranker)

    return agent, vector_store


def main():
    # ---------- 页面配置 ----------
    st.set_page_config(
        page_title="智能值班助手",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 智能值班助手")
    st.markdown(f"基于 LangGraph Agent + 多路召回 + `{config.LLM_MODEL}`，帮你快速检索值班相关知识")
    st.caption("语义检索 + 关键词检索 | 多轮对话记忆 | 流式输出")
    st.markdown("---")

    # ---------- 侧边栏 ----------
    with st.sidebar:
        st.header("⚙️ 系统信息")
        st.markdown("🏗️ **架构**：LangGraph Agent")
        st.markdown("🔧 **能力**：知识库检索 + 天气 + 时间")
        st.markdown("🧠 **记忆**：多轮对话")
        st.markdown("---")
        st.markdown("**检索配置**：")
        st.markdown(f"📊 语义检索：{config.TOP_K_FIRST} 个")
        st.markdown(f"🔑 关键词检索：{config.BM25_K} 个")
        st.markdown(f"🎯 精排：{config.TOP_K_FINAL} 个")
        st.markdown("---")
        st.markdown("**工作流程**：")
        st.markdown("1️⃣ 理解你的问题")
        st.markdown("2️⃣ 从知识库检索相关内容")
        st.markdown("3️⃣ 整合信息生成回答")

        if st.button("🔄 重新加载"):
            st.cache_resource.clear()
            st.rerun()

    # ---------- 加载系统 ----------
    @st.cache_resource
    def cached_load():
        return load_rag_system()

    agent, vector_store = cached_load()

    st.success(f"✅ Agent 已就绪，知识库共 {vector_store.index.ntotal} 个向量")

    # ---------- 初始化会话 ----------
    init_session_state()

    # ---------- 显示聊天历史 ----------
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])

    # ---------- 用户输入 ----------
    query = st.chat_input("输入你的问题...")

    if query:
        # 显示用户消息
        st.chat_message("user").write(query)
        st.session_state["messages"].append({"role": "user", "content": query})
        st.session_state["agent_state"]["question"] = query

        with st.spinner("🧠 Agent 正在思考..."):
            try:
                # --- Agent 负责决策 + 检索 + 准备 prompt ---
                result = agent.invoke(st.session_state["agent_state"])
                st.session_state["agent_state"] = result

                # --- UI 层负责流式生成 ---
                prompt = result.get("prompt", "")
                answer = result.get("answer", "")
                if prompt and not answer:
                    llm = ChatOpenAI(
                        api_key=config.DEEPSEEK_API_KEY,
                        base_url=config.LLM_BASE_URL,
                        model=config.LLM_MODEL,
                        temperature=config.TEMPERATURE,
                        streaming=True,
                    )

                    answer_box = st.empty()
                    full = ""
                    for chunk in llm.stream(prompt):
                        if chunk.content:
                            full += chunk.content
                            answer_box.markdown(full + "▌")
                    answer_box.markdown(full)

                    result["answer"] = full
                    st.session_state["agent_state"]["answer"] = full
                    # 关键修复：把真实回答也写入 agent 的对话历史，
                    # 否则下一轮 LLM 只能看到空的工具调用消息，多轮记忆失效
                    st.session_state["agent_state"]["messages"].append(
                        AIMessage(content=full)
                    )
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": full}
                    )
                else:
                    # 不走工具：analyze 已直接生成答案，直接用（避免二次调用 LLM）
                    full = answer if answer else ""
                    st.chat_message("assistant").write(full)
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": full}
                    )

                # 显示来源
                if result.get("sources"):
                    with st.expander("📄 查看引用来源"):
                        for source in result["sources"]:
                            st.markdown(f"- {source}")

                # 调试信息
                with st.sidebar.expander("🧠 调试信息", expanded=False):
                    # 显示查询重写结果
                    if result.get("rewritten_queries"):
                        st.markdown("**🔄 查询重写：**")
                        for i, rq in enumerate(result["rewritten_queries"], 1):
                            st.markdown(f"{i}. `{rq}`")
                    st.json({
                        "是否使用工具": bool(result.get("retrieved_docs")),
                        "来源数量": len(result.get("sources", [])),
                        "来源列表": result.get("sources", []),
                        "历史消息数": len(result.get("messages", [])),
                    })

            except Exception as e:
                st.error(f"❌ 出错了：{type(e).__name__}: {e}")
                name = type(e).__name__.lower()
                if any(k in name for k in ("timeout", "connection", "http", "api", "requests")):
                    st.info("💡 请检查 API Key 是否正确，或网络是否通畅")
                else:
                    st.info("💡 这是本地运行错误，请在启动 Streamlit 的终端查看详细日志")

    # ---------- 清空对话 ----------
    if st.sidebar.button("🗑️ 清空对话"):
        st.session_state["messages"] = []
        st.session_state["agent_state"] = {
            "question": "",
            "messages": [],
            "retrieved_docs": "",
            "sources": [],
            "answer": "",
            "prompt": "",
            "rewritten_queries": [],
        }
        st.rerun()


if __name__ == "__main__":
    main()
