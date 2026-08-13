# ============================================================
# RAG Web 应用（基于 Streamlit）
#
# 功能说明：
#   1. 加载本地 FAISS 索引（docs/ 文件夹内的所有文档）
#   2. 用户通过网页输入问题
#   3. 系统执行：FAISS 粗筛 → Rerank 精排 → DeepSeek 生成
#   4. 在网页上显示回答 + 检索到的参考资料
# ============================================================
# 运行命令  streamlit run 工程化和部署/app.py

import sys
import os

# 把项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 离线开关：本地默认离线；云端设 HF_OFFLINE=0 联网下载。
# 必须在所有 langchain/transformers 导入前设置；config.py 也会同步这些值。
os.environ.setdefault("HF_OFFLINE", "1")
os.environ["HF_HUB_OFFLINE"] = os.environ["HF_OFFLINE"]
os.environ["TRANSFORMERS_OFFLINE"] = os.environ["HF_OFFLINE"]
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")  # 尊重环境变量覆盖；默认官方源
print(f"[启动诊断] HF_OFFLINE={os.environ.get('HF_OFFLINE')} HF_ENDPOINT={os.environ.get('HF_ENDPOINT')} DEEPSEEK_API_KEY={'已设置' if os.environ.get('DEEPSEEK_API_KEY') else '未设置'}")

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import time

import config
from src.rag.retriever import (
    load_vectorstore,
    load_reranker,
    retrieve_with_rerank,
    format_docs,
)


@st.cache_resource
def load_llm():
    """加载 DeepSeek LLM（带缓存，只加载一次）"""
    return ChatOpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.LLM_BASE_URL,
        model=config.LLM_MODEL,
        temperature=config.TEMPERATURE,
    )


# ============================================================
# 主程序
# ============================================================

def main():
    # ---------- 页面配置 ----------
    st.set_page_config(
        page_title="智能值班助手",
        page_icon="🤖",
        layout="wide"
    )

    # ---------- 标题 ----------
    st.title("🤖 智能值班助手")
    st.markdown("基于多文档检索 + Rerank 精排 + DeepSeek 生成")
    st.markdown("---")

    # ---------- 侧边栏 ----------
    with st.sidebar:
        st.header("⚙️ 系统信息")
        st.markdown(f"📁 **文档目录**：`{config.DOCS_DIR}`")
        st.markdown(f"📊 **嵌入模型**：`{config.EMBEDDING_MODEL.split('/')[-1]}`")
        st.markdown(f"🎯 **Rerank 模型**：`{config.RERANK_MODEL.split('/')[-1]}`")
        st.markdown(f"🤖 **生成模型**：`{config.LLM_MODEL}`")
        st.markdown("---")
        st.markdown("**工作流程**：")
        st.markdown(f"1️⃣ FAISS 粗筛（{config.TOP_K_FIRST} 个候选）")
        st.markdown(f"2️⃣ Rerank 精排（取 {config.TOP_K_FINAL} 个）")
        st.markdown(f"3️⃣ `{config.LLM_MODEL}` 生成答案")

        if st.button("🔄 重新加载索引"):
            st.cache_resource.clear()
            st.rerun()

    # ---------- 加载系统 ----------
    with st.spinner("🚀 正在加载系统..."):
        vector_store = load_vectorstore()
        reranker = load_reranker()
        llm = load_llm()
        retriever = vector_store.as_retriever(search_kwargs={"k": config.TOP_K_FIRST})

    # ---------- 检查索引向量数 ----------
    vector_count = vector_store.index.ntotal
    st.success(f"✅ 系统已就绪，向量数据库共 {vector_count} 个文本块")

    # ---------- 创建提示词模板 ----------
    prompt_template = ChatPromptTemplate.from_template("""
请根据以下参考资料回答用户的问题。

参考资料：
{context}

用户问题：{input}

请基于参考资料给出准确、简洁的回答。如果参考资料中找不到相关信息，请明确告知。
""")

    # ---------- 创建 RAG 链 ----------
    def rag_chain(query):
        # 1. 检索 + Rerank
        docs = retrieve_with_rerank(query, retriever, reranker)
        context = format_docs(docs)

        # 2. 构建提示词
        prompt = prompt_template.format(context=context, input=query)

        # 3. 调用 LLM
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)

        return answer, docs

    # ---------- 用户输入 ----------
    st.markdown("### 💬 你的问题")

    # 输入框
    query = st.text_input(
        "输入你的问题：",
        value=st.session_state.get("query", ""),
        placeholder="例如：值班时遇到错误码 DBS.200026 怎么处理？",
        label_visibility="collapsed"
    )

    # ---------- 提交按钮 ----------
    col1, col2 = st.columns([1, 5])
    with col1:
        submit = st.button("🔍 提交", type="primary", use_container_width=True)

    # ---------- 执行问答 ----------
    if submit and query:
        with st.spinner("⏳ 正在检索并生成答案..."):
            try:
                start_time = time.time()
                answer, source_docs = rag_chain(query)
                elapsed_time = time.time() - start_time

                # 显示回答
                st.markdown("### 📝 回答")
                st.markdown(f"⏱️ 用时：{elapsed_time:.2f} 秒")

                # 用容器显示回答（带背景色）
                with st.container():
                    st.markdown(f'<div style="background-color:#f0f2f6;padding:20px;border-radius:10px;">{answer}</div>', unsafe_allow_html=True)

                # 显示参考资料
                if source_docs:
                    st.markdown("### 📚 参考资料")
                    for i, doc in enumerate(source_docs, 1):
                        source = doc.metadata.get("source", "未知")
                        page = doc.metadata.get("page", "N/A")
                        with st.expander(f"📄 来源 {i}：{os.path.basename(source)}（第 {page} 页）"):
                            st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))

                # 保存到历史
                if "history" not in st.session_state:
                    st.session_state["history"] = []
                st.session_state["history"].append({
                    "query": query,
                    "answer": answer,
                    "sources": [doc.metadata.get("source", "未知") for doc in source_docs]
                })

            except Exception as e:
                st.error(f"❌ 出错了：{e}")
                st.info("💡 请检查 API Key 是否正确，或网络是否通畅")

    # ---------- 历史记录 ----------
    if "history" in st.session_state and st.session_state["history"]:
        with st.expander("📜 历史对话"):
            for i, item in enumerate(reversed(st.session_state["history"]), 1):
                st.markdown(f"**{i}. Q：{item['query']}**")
                st.text(item["answer"][:200] + ("..." if len(item["answer"]) > 200 else ""))
                st.caption(f"📄 来源：{', '.join(set(item['sources']))}")
                st.markdown("---")


if __name__ == "__main__":
    main()
