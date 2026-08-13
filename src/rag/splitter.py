# ============================================================
# 文本切分器
# ============================================================
# 功能：把长文档切成小块，便于向量化和检索
# ============================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def get_text_splitter():
    """
    创建文本切分器实例
    
    返回：
        RecursiveCharacterTextSplitter 实例
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,        # 每块最大字符数
        chunk_overlap=config.CHUNK_OVERLAP,  # 重叠字符数
        separators=[                         # 切分优先级（由粗到细）
            "\n\n",      # 1. 段落分隔
            "\n",        # 2. 换行
            "。", "！", "？",  # 3. 句号/感叹号/问号
            "；",        # 4. 分号
            "，",        # 5. 逗号
            " ",         # 6. 空格
            ""           # 7. 字符（兜底）
        ]
    )