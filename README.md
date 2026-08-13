# 智能值班助手 Agent（华为云 HCS）

面向华为云 HCS（Huawei Cloud Stack）的**智能值班助手**：基于 **RAG + LangGraph** 的问答智能体，将值班问题（错误码 / 日志 / 操作指南）自动分类，检索知识库并推荐解决方案。

## 核心能力

| 能力 | 说明 |
| --- | --- |
| **自动化问题分类** | LLM 自主判断问题类型（错误处理 / 日志分析 / 操作指南），并按错误码场景输出"错误描述 / 原因 / 建议" |
| **知识检索** | FAISS 语义检索 + BM25 关键词检索（jieba 中文分词）多路召回，CrossEncoder 精排，查询重写提升召回率 |
| **解决方案推荐** | 基于检索到的参考资料，生成带引用来源的解决方案 |

## 技术栈

`LangGraph` · `LangChain` · `FAISS` · `BM25(jieba)` · `sentence-transformers` · `DeepSeek` · `Streamlit` · `FastAPI`

## 快速开始

```bash
# 1. 安装依赖（版本已锁定）
pip install -r requirements.txt

# 2. 配置 DeepSeek API Key（代码中不保存明文 Key）
set DEEPSEEK_API_KEY=你的key          # Windows CMD
$env:DEEPSEEK_API_KEY = "你的key"     # Windows PowerShell
export DEEPSEEK_API_KEY=你的key       # Linux/macOS

# 3. 启动 Web 界面
streamlit run src/ui/streamlit_app.py
```

> 首次使用前需准备知识库（放入 `docs/`）并构建向量索引：`python scripts/build_index.py`

## 常见操作

- **加文档到知识库**：把 `.pdf/.docx/.txt` 放进 `docs/`，跑 `python scripts/build_index.py --sync-repo`（一键同步到云端索引），提交推送后云端自动更新。详见 [使用手册.md](使用手册.md) 4.3/4.4 节
- **换 API / 换 LLM**：改 `config.py` 的 `LLM_BASE_URL` / `LLM_MODEL`（或设同名环境变量），不用改代码。详见 [使用手册.md](使用手册.md) 4.5 节

## 文档

- **[使用手册.md](使用手册.md)** — 功能说明、运行方式、Streamlit Cloud 部署指南
- **[代码说明.md](代码说明.md)** — 系统架构、各模块详解、核心数据流与历史修复

## 在线演示

已部署至 Streamlit Cloud，支持远程体验问答、引用来源与多轮对话。
