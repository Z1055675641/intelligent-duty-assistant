# ============================================================
# 嵌入模型加载器
# ============================================================
# 功能：加载 Hugging Face 嵌入模型（本地缓存优先，云端允许联网下载）
# ============================================================

import os
import urllib.request

from langchain_huggingface import HuggingFaceEmbeddings

import config


def probe_hf_endpoint(model_id):
    """
    探测 HuggingFace 下载源连通性（云端联网排障用）。

    测试的是真实下载路径：{base}/{model_id}/resolve/main/config.json。
    返回：[(base_url, 是否可达, 说明), ...]
    """
    results = []
    for base in ("https://huggingface.co", "https://hf-mirror.com"):
        url = f"{base}/{model_id}/resolve/main/config.json"
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=8) as resp:
                results.append((base, True, f"HTTP {resp.status}"))
        except Exception as e:
            results.append((base, False, f"{type(e).__name__}: {e}"))
    return results


def probe_and_report(model_id):
    """把连通性结果打进 Logs；本地离线模式跳过"""
    if config.HF_OFFLINE == "1":
        return
    for base, ok, info in probe_hf_endpoint(model_id):
        print(f"[网络诊断] {base} → {'✅ 可达' if ok else '❌ 不可达'} ({info})")


def get_embeddings():
    """
    加载嵌入模型

    返回：
        HuggingFaceEmbeddings 实例

    说明：
        local_files_only 跟随离线开关 config.HF_OFFLINE：
        - 本机默认离线（HF_OFFLINE=1）：只从本地缓存加载
        - 云端部署（HF_OFFLINE=0）：允许联网下载模型
    """
    probe_and_report(config.EMBEDDING_MODEL)
    try:
        return HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            cache_folder=config.CACHE_DIR,
            model_kwargs={"device": "cpu", "local_files_only": config.HF_OFFLINE == "1"},
            encode_kwargs={"normalize_embeddings": True}
        )
    except Exception as e:
        import traceback
        print(f"❌ 嵌入模型加载失败：{type(e).__name__}: {e!r}")
        traceback.print_exc()
        raise