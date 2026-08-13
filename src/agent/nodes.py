# ============================================================
# LangGraph 节点定义
# ============================================================
# 功能：定义 LangGraph 的四个核心节点
#   1. analyze_question: LLM 自主决策是否使用工具
#   2. rewrite_query: 查询重写（新增）
#   3. execute_tools: 执行工具调用（支持多查询）
#   4. generate_answer: 生成最终答案
# ============================================================

import re
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage

import config
from .state import AgentState


def analyze_question(state, llm_with_tools):
    """
    Node 1：LLM 自主决策是否使用工具
    
    参数：
        state: 当前状态
        llm_with_tools: 绑定了工具的 LLM
    
    返回：
        更新后的状态
    """
    question = state["question"]
    history = state.get("messages", [])
    
    # 每轮开始时清空上一轮留下的检索结果，避免陈旧上下文串扰新一轮
    state["retrieved_docs"] = ""
    state["sources"] = []
    state["rewritten_queries"] = []
    state["answer"] = ""  # 清空上一轮答案，避免 UI 复读旧回答（走工具轮由 UI/API 层补全）

    # 防御：剔除历史中可能残留的 system 提示词（旧会话升级前遗留）
    history = [
        m for m in history
        if not (isinstance(m, dict) and m.get("role") == "system")
    ]

    # 系统提示词（中文）
    system_prompt = """你是"小R"，一名华为云 HCS（Huawei Cloud Stack）的智能值班助手，帮助值班人员快速定位和解决云上问题。

你的职责：
- 基于华为云 HCS 产品文档知识库（如 GaussDB 参考指南等）回答技术问题
- 对值班问题自动分类：错误码查询 / 日志路径查询 / 操作指南查询
- 遇到错误码时给出：错误描述、可能原因、解决建议

你可以使用以下工具：

1. **rag_search**：从华为云文档知识库中检索信息
   - 适用：错误码含义与处理、日志路径、操作指南、配置说明等值班相关问题
   - 示例："错误码 DBS.200026 表示什么？怎么处理？" → 调用 rag_search

2. **get_weather**：查询实时天气
   - 适用：城市天气、温度、要不要带伞
   - 示例："上海天气怎么样" → 调用 get_weather，参数 city="上海"

3. **get_current_time**：获取当前日期和时间
   - 适用："今天几号"、"现在几点"
   - 示例："今天几号" → 调用 get_current_time

工作流程：
1. 判断用户的问题需要哪个工具
2. 需要查华为云文档 → 调用 rag_search
3. 需要查天气 → 调用 get_weather
4. 需要查时间 → 调用 get_current_time
5. 不需要工具 → 直接回答

重要规则：
- 用户问华为云/GaussDB 相关技术问题、错误码、日志路径、操作步骤时，必须调用 rag_search！
- 用户问天气/时间时，必须调用对应的工具！
- 如果 rag_search 返回多个文档，请整合所有文档的信息回答
- 回答要基于检索到的文档内容，不要凭空编造
- 不要拒绝回答！"""
    
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": question}]
    
    response = llm_with_tools.invoke(messages)
    
    # 关键修复：只保存干净的对话历史（不含 system 提示词），
    # 否则每轮都会把 system 再拼接一份，多轮后提示词爆炸式累积
    state["messages"] = history + [{"role": "user", "content": question}, response]

    if hasattr(response, 'tool_calls') and response.tool_calls:
        return state
    else:
        state["answer"] = response.content
        return state


# ============================================================
# 🆕 新增：查询重写节点
# ============================================================

def rewrite_query_node(state, llm) -> AgentState:
    """
    Node：将用户问题改写成多个检索友好的查询
    
    工作流程：
        1. 从 state 中获取用户问题
        2. 如果配置关闭，直接存原始问题
        3. 调用 LLM 生成 2-3 个改写查询
        4. 存入 state["rewritten_queries"]
    
    参数：
        state: 当前状态
        llm: 大模型实例（不带工具绑定）
    
    返回：
        更新后的状态
    """
    question = state["question"]
    
    # 如果配置关闭了查询重写，直接存原始问题
    if not hasattr(config, 'QUERY_REWRITE_ENABLED') or not config.QUERY_REWRITE_ENABLED:
        state["rewritten_queries"] = [question]
        print(f"🔄 查询重写已关闭，使用原始问题")
        return state
    
    # 调用查询重写器
    from ..rag.query_rewriter import rewrite_queries
    
    try:
        # 获取对话历史（用于上下文）
        history = state.get("messages", [])
        history_text = ""
        if history:
            recent = history[-4:] if len(history) > 4 else history
            for msg in recent:
                if hasattr(msg, 'get'):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                elif hasattr(msg, 'type'):
                    role = msg.type
                    content = msg.content
                else:
                    continue
                if role in ("user", "human"):
                    history_text += f"用户：{content}\n"
                elif role in ("assistant", "ai"):
                    if content:
                        history_text += f"助手：{content}\n"
        
        # 生成改写查询
        queries = rewrite_queries(
            llm=llm,
            question=question,
            history_text=history_text,
            count=getattr(config, 'QUERY_REWRITE_COUNT', 3)
        )
        state["rewritten_queries"] = queries
        print(f"🔄 查询重写：{queries}")
        
    except Exception as e:
        # 兜底：任何错误都退回原始问题
        print(f"⚠️ 查询重写失败：{e}，使用原始问题")
        state["rewritten_queries"] = [question]
    
    return state


def execute_tools(state, rag_tool, rag_tool_multi, weather_tool, time_tool):
    """
    Node 3：执行工具调用（支持多工具 + 多查询检索）

    参数：
        state: 当前状态
        rag_tool: RAG 检索工具（单查询用，候选数默认 TOP_K_FIRST）
        rag_tool_multi: RAG 检索工具（多查询用，候选数 QUERY_REWRITE_TOP_K）
        weather_tool: 天气工具
        time_tool: 时间工具

    返回：
        更新后的状态
    """
    messages = state["messages"]
    last_message = messages[-1]
    tool_calls = last_message.tool_calls
    
    # 工具映射
    tool_map = {
        "rag_search": rag_tool,
        "get_weather": weather_tool,
        "get_current_time": time_tool,
    }
    
    all_sources = []  # 聚合本次所有工具调用的来源，避免多工具调用时只保留最后一个

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_func = tool_map.get(tool_name)
        
        if not tool_func:
            continue
        
        # ---------- 特殊处理 rag_search（支持多查询） ----------
        if tool_name == "rag_search":
            # 检查是否有改写查询
            rewritten = state.get("rewritten_queries", [])
            original_query = state.get("question", "")
            
            # 判断是否启用多查询：有改写查询且不只包含原始问题
            is_multi_query = (
                rewritten and 
                len(rewritten) > 0 and 
                (len(rewritten) > 1 or rewritten[0] != original_query)
            )
            
            if is_multi_query and getattr(config, 'QUERY_REWRITE_ENABLED', True):
                print(f"🔍 多查询检索：共 {len(rewritten)} 个查询")
                print(f"   📝 查询列表：{rewritten}")
                
                # 多查询检索：收集所有查询的结果
                all_results = []
                all_sources = []
                
                for i, q in enumerate(rewritten):
                    print(f"   🔎 执行查询 {i+1}/{len(rewritten)}：{q}")
                    # 多查询用专用工具（每个查询只取 QUERY_REWRITE_TOP_K 个候选）
                    single_result = rag_tool_multi.invoke({"query": q})
                    all_results.append(single_result)
                
                # 合并结果（去重）
                # 注意：rag_tool 返回的是格式化字符串，包含文档内容和来源
                # 我们通过提取来源来去重
                merged_result = ""
                seen_sources = set()
                
                for i, result in enumerate(all_results):
                    # 提取来源
                    sources = []
                    if "【所有来源】" in result:
                        source_match = re.search(r'【所有来源】\n(.*)', result, re.DOTALL)
                        if source_match:
                            sources_text = source_match.group(1)
                            sources = [s.strip().replace('- ', '') for s in sources_text.split('\n') if s.strip()]
                    
                    # 检查来源是否已存在（去重）
                    new_sources = [s for s in sources if s not in seen_sources]
                    for s in new_sources:
                        seen_sources.add(s)
                    
                    # 如果这个查询有新的来源，保留结果
                    if new_sources:
                        # 提取文档内容部分（去掉来源部分）
                        content_part = re.sub(r'\n\n【所有来源】.*', '', result, flags=re.DOTALL)
                        if i > 0:
                            merged_result += f"\n\n--- [查询 {i+1}：{rewritten[i]}] ---\n\n"
                        merged_result += content_part
                
                # 如果合并结果为空，使用第一个查询的结果
                if not merged_result and all_results:
                    merged_result = all_results[0]
                
                # 添加来源信息
                if seen_sources:
                    merged_result += "\n\n【所有来源】\n"
                    for s in seen_sources:
                        merged_result += f"  - {s}\n"
                
                result = merged_result
                print(f"   ✅ 多查询检索完成，合并来源数：{len(seen_sources)}")
                
            else:
                # 单查询检索（原有逻辑）
                print(f"🔍 单查询检索：{original_query}")
                result = rag_tool.invoke(tool_call["args"])
            
            # 处理结果（来源提取和状态保存）
            messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
            
            # 提取来源
            sources = []
            if "【所有来源】" in result:
                source_match = re.search(r'【所有来源】\n(.*)', result, re.DOTALL)
                if source_match:
                    sources_text = source_match.group(1)
                    sources = [s.strip().replace('- ', '') for s in sources_text.split('\n') if s.strip()]
            
            state["retrieved_docs"] = result
            all_sources.extend(sources)
        
        # ---------- 天气工具 ----------
        elif tool_name == "get_weather":
            result = tool_func.invoke(tool_call["args"])
            messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
            state["retrieved_docs"] = result
            all_sources.append("天气数据来源：wttr.in")
        
        # ---------- 时间工具 ----------
        elif tool_name == "get_current_time":
            result = tool_func.invoke(tool_call["args"])
            messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
            state["retrieved_docs"] = result
            all_sources.append("时间来源：系统时间")
    
    # 聚合去重后统一写入来源（多工具调用时不再只保留最后一个）
    state["sources"] = list(dict.fromkeys(all_sources))
    state["messages"] = messages
    return state


def generate_answer(state, llm):
    """
    Node 4：生成最终答案
    
    参数：
        state: 当前状态
        llm: 大模型对象
    
    返回：
        更新后的状态
    """
    question = state["question"]
    history = state.get("messages", [])
    
    # 构建历史文本
    history_text = ""
    if history:
        recent = history[-4:] if len(history) > 4 else history
        for msg in recent:
            if hasattr(msg, 'get'):
                role = msg.get("role", "")
                content = msg.get("content", "")
            elif hasattr(msg, 'type'):
                role = msg.type
                content = msg.content
            else:
                continue
            if role in ("user", "human"):
                history_text += f"用户：{content}\n"
            elif role in ("assistant", "ai"):
                if content:
                    history_text += f"助手：{content}\n"
    
    # 根据是否有检索结果构建提示词
    if state.get("retrieved_docs"):
        prompt = f"""对话历史：
{history_text if history_text else "（无历史对话）"}

参考资料：
{state['retrieved_docs']}

当前用户问题：{question}

【重要】
1. 如果有多个文档，请整合所有文档的信息回答
2. 按逻辑顺序组织回答（如：知识点1、知识点2、知识点3）
3. 如果某些信息在多个文档中重复，只保留最完整的版本
4. 请先**分类**用户的问题（错误码查询/日志路径查询/操作指南查询）。
5. 如果是错误码，必须返回：**错误描述、可能原因、解决建议**。
6. 如果涉及多个文档，按"相关性从高到低"排列。
7. 结尾必须附上参考来源。
8. 如果参考资料中找不到相关信息，请明确告知
9. 回答要准确、完整、有条理"""
    else:
        prompt = f"""对话历史：
{history_text if history_text else "（无历史对话）"}

当前用户问题：{question}

请直接回答用户的问题。
如果用户的问题和之前的对话有关，请结合上下文回答。
用简洁、友好的语气。"""
    
    state["prompt"] = prompt    # 把 prompt 存起来，让 UI 层去流式调用
    return state  # 不立即生成，只保存 prompt