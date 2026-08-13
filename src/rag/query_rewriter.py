# ============================================================
# 查询重写器 (Query Rewriter)
# ============================================================
# 功能：将用户口语化问题改写成多个检索友好的查询
# ============================================================

import re
from typing import List, Optional

from langchain_openai import ChatOpenAI

import config


def rewrite_queries(
    llm: ChatOpenAI,
    question: str,
    history_text: str = "",
    count: int = 3,
) -> List[str]:
    """
    将用户问题改写成多个更适合检索的查询
    
    参数：
        llm: 大模型实例（用于改写）
        question: 用户原始问题
        history_text: 对话历史（可选）
        count: 生成的查询数量
    
    返回：
        queries: 包含改写查询的列表（至少包含原始问题）
    
    原理：
        用户口语化问题 → LLM 生成多个检索友好变体 → 多路检索 → 提高召回率
    """
    
    # ---------- 构建提示词 ----------
    system_prompt = """你是一个查询改写助手。你的任务是将用户的问题改写成多个更适合检索的查询。

改写规则：
1. 提取核心关键词和实体（错误码、文件名、工具名、参数名）
2. 补充同义词和相关术语
3. 去除口语化表达，使用正式/技术术语
4. 每个查询从不同角度描述问题
5. 直接输出查询，每行一个，不要编号

示例：
用户问题：那个数据库报错怎么办？
输出：
数据库连接失败 错误处理 排查步骤
DBS 数据库报错 解决方法 常见错误码
数据库连接超时 异常处理 运维手册"""

    user_prompt = f"用户问题：{question}"
    if history_text:
        user_prompt = f"对话历史：{history_text}\n\n{user_prompt}"
    
    # ---------- 调用 LLM ----------
    try:
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        # ---------- 解析输出 ----------
        variants = []
        for line in response.content.strip().split('\n'):
            line = line.strip()
            # 去除可能的编号（如 "1." "2." "3."）
            line = re.sub(r'^[\d]+[\.、\)]\s*', '', line)
            if line:
                variants.append(line)
        
        # 去重（保留原始顺序）
        variants = list(dict.fromkeys(variants))
        
        # 过滤掉和原始问题完全相同的变体（避免重复检索）
        variants = [v for v in variants if v != question]
        
        # 如果没有变体，用原始问题兜底
        if not variants:
            return [question]
        
        # 最多返回 count 个
        return variants[:count]
        
    except Exception as e:
        # 任何错误都退回原始问题，不影响主流程
        print(f"⚠️ 查询改写失败：{e}，使用原始问题")
        return [question]