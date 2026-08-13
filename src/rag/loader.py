# ============================================================
# 文档加载器
# ============================================================
# 功能：遍历 docs/ 文件夹，自动识别并加载 PDF、Word、TXT
# ============================================================

import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document


def load_all_documents(docs_dir: str) -> List[Document]:
    """
    加载 docs/ 文件夹内所有支持的文档（.pdf, .docx, .txt）
    
    参数：
        docs_dir: 文件夹路径
    
    返回：
        all_documents: 所有文档的 Document 列表
    """
    
    # 检查文件夹是否存在
    if not os.path.exists(docs_dir):
        print(f"⚠️ 文件夹不存在：{docs_dir}，正在创建...")
        os.makedirs(docs_dir)
        print(f"✅ 已创建文件夹：{docs_dir}")
        return []
    
    # 支持的文件格式及对应的加载器
    extensions = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": TextLoader,
    }
    
    all_documents = []
    
    # 遍历文件夹内所有文件
    for file_path in Path(docs_dir).iterdir():
        if file_path.is_dir():
            continue  # 跳过子文件夹
        
        ext = file_path.suffix.lower()
        
        if ext in extensions:
            print(f"📄 正在加载：{file_path.name}")
            try:
                loader_class = extensions[ext]
                loader = loader_class(str(file_path))
                docs = loader.load()
                all_documents.extend(docs)
                print(f"   ✅ 加载成功，共 {len(docs)} 页/段")
            except Exception as e:
                print(f"   ❌ 加载失败：{e}")
        else:
            print(f"⚠️ 跳过不支持的文件：{file_path.name}")
    
    return all_documents