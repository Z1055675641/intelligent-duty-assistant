# ============================================================
# 全局配置文件
# ============================================================
# 所有配置集中管理，修改后自动生效
# ============================================================

import os
from pathlib import Path

# ---------- 基础路径 ----------
# BASE_DIR: 项目根目录（智能值班助手Agent）
BASE_DIR = Path(__file__).parent.resolve()

# DOCS_DIR: 文档存放目录
DOCS_DIR = BASE_DIR / "docs"

# INDEX_SAVE_PATH: FAISS 向量索引保存路径
# 注意：FAISS C++ 后端不支持含中文的路径，必须使用纯 ASCII 路径
# 优先级：环境变量 FAISS_INDEX_PATH > 仓库内 data/faiss_index（仅当仓库路径为 ASCII，如云端）> 本地 F:/faiss_index_data
_index_env = os.environ.get("FAISS_INDEX_PATH", "")
if _index_env:
    INDEX_SAVE_PATH = Path(_index_env)
elif str(BASE_DIR).isascii():
    # 仓库路径为 ASCII（例如云端克隆目录），直接用随仓库携带的索引
    INDEX_SAVE_PATH = BASE_DIR / "data" / "faiss_index"
else:
    # 本机中文路径（FAISS 不支持），退回本地 ASCII 绝对路径
    INDEX_SAVE_PATH = Path("F:/faiss_index_data")

# ---------- API 配置 ----------
# DeepSeek API Key：只从环境变量读取（本机本地用 set / $env: 设置，云端用 Streamlit Secrets 注入）
# ⚠️ 严禁把真实 Key 写进本文件提交——仓库公开后会被所有人看到、被人滥用
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not DEEPSEEK_API_KEY:
    # 不用 emoji：config 在导入时执行，此时 stdout 还没套 UTF-8 wrapper，GBK 控制台遇到 ⚠️ 会 UnicodeEncodeError
    print("[警告] 未设置 DEEPSEEK_API_KEY 环境变量，问答将无法调用 DeepSeek。本地可执行：set DEEPSEEK_API_KEY=你的key（CMD）或 $env:DEEPSEEK_API_KEY=你的key（PowerShell）")

# ---------- LLM 供应商配置 ----------
# 换模型 / 换 API 供应商：改这里（或设环境变量 LLM_BASE_URL / LLM_MODEL 覆盖），不用改代码
# 兼容任何 OpenAI 格式接口：base_url 填服务商的 OpenAI 兼容端点，model 填模型名
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")

# ---------- 模型配置 ----------
# 嵌入模型（把文字转成 384 维向量）
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Rerank 模型（对检索结果精排）
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Hugging Face 模型缓存目录（云端可通过环境变量 HF_CACHE_DIR 覆盖）
CACHE_DIR = os.environ.get("HF_CACHE_DIR", str(Path.home() / ".cache" / "huggingface" / "hub"))

# ---------- 文本切分配置 ----------
# chunk_size: 每块最大字符数（越大，块越完整，但检索精度略降）
CHUNK_SIZE = 2500

# chunk_overlap: 相邻块重叠字符数（防止边界信息丢失）
CHUNK_OVERLAP = 300

# ---------- 检索配置 ----------
# FAISS 语义检索配置
TOP_K_FIRST = 50       # FAISS 返回候选数
TOP_K_FINAL = 15       # Rerank 精排最终数

# BM25 关键词检索配置
BM25_K = 50            # BM25 返回候选数

# 多路召回权重配置
ENSEMBLE_WEIGHTS = [0.5, 0.5]  # [语义检索权重, BM25权重]，和为1

# ---------- LLM 配置 ----------
# temperature: 0=最确定，1=最随机
TEMPERATURE = 0.3

# ---------- 查询重写配置（🆕 新增） ----------
# 是否启用查询重写（True=开启，False=关闭）
# 开启后，用户口语化问题会被改写成多个检索友好的查询
# 关闭后，直接使用原始问题检索
QUERY_REWRITE_ENABLED = True

# 每次生成多少个改写查询（建议 2-3 个）
# 越多召回率越高，但检索耗时也越长
# 注：调成 2 以降低回答等待时间（本地 Rerank 耗时随查询数/候选数线性增长）
QUERY_REWRITE_COUNT = 2

# 查询重写使用的温度参数（低温度保证输出稳定）
QUERY_REWRITE_TEMPERATURE = 0.1

# 每个改写查询检索返回的候选数
# 2 个查询 × 20 个候选 = 40 个候选 → 去重 → Rerank 取 15 个
# 注：之前是 3×30=90，改成 2×20=40 后检索+Rerank 本地耗时约减半
QUERY_REWRITE_TOP_K = 20

# ---------- 环境配置 ----------
# 模型下载/离线模式：
# - 本机默认离线（HF_OFFLINE=1，用本地缓存加载模型）
# - 云端/需要联网下载模型时：设环境变量 HF_OFFLINE=0（Streamlit Cloud Secrets 注入）
# 下载源默认官方源 huggingface.co（云端服务器在境外，hf-mirror 连不上）
# 国内本地如需联网下载，可显式设 HF_ENDPOINT=https://hf-mirror.com 覆盖
HF_OFFLINE = os.environ.get("HF_OFFLINE", "1")
os.environ["HF_HUB_OFFLINE"] = HF_OFFLINE
os.environ["TRANSFORMERS_OFFLINE"] = HF_OFFLINE
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")