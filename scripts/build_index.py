import sys
import os
import shutil
from pathlib import Path

# 把项目根目录加入 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.vectorstores import FAISS

import config
from src.rag.loader import load_all_documents
from src.rag.splitter import get_text_splitter
from src.rag.embeddings import get_embeddings


def main():
    print("=" * 60)
    print("📚 索引构建工具")
    print("=" * 60)
    
    print(f"\n📁 扫描文档目录：{config.DOCS_DIR}")
    documents = load_all_documents(str(config.DOCS_DIR))
    
    if not documents:
        print("❌ 没有加载到任何文档")
        return
    
    print(f"\n✅ 共加载 {len(documents)} 个文档片段")
    
    print("\n✂️ 正在切分文档...")
    splitter = get_text_splitter()
    chunks = splitter.split_documents(documents)
    print(f"✅ 切分为 {len(chunks)} 个文本块")
    
    print("\n🧮 正在向量化...")
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    print(f"✅ 向量化完成，共 {vector_store.index.ntotal} 个向量")
    
    # ---------- 保存索引（确保目录存在） ----------
    save_path = str(config.INDEX_SAVE_PATH)
    os.makedirs(save_path, exist_ok=True)  # ← 关键修复
    print(f"\n💾 正在保存索引到 {save_path}...")
    vector_store.save_local(save_path)
    print("✅ 索引已保存")

    # 🆕 --sync-repo：同时把索引复制到仓库内 data/faiss_index（云端加载用，提交推送后生效）
    # 用 shutil 复制而非 vector_store.save_local：本机仓库路径可能含中文，FAISS 直接写会报错
    if "--sync-repo" in sys.argv:
        repo_index = config.BASE_DIR / "data" / "faiss_index"
        repo_index.mkdir(parents=True, exist_ok=True)
        for name in ("index.faiss", "index.pkl"):
            src = Path(save_path) / name
            if src.exists():
                shutil.copy2(src, repo_index / name)
        print(f"✅ 已同步到仓库内 {repo_index}（提交推送后云端生效）")

    print("\n" + "=" * 60)
    print("🎉 索引构建完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()