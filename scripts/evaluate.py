# ============================================================
# RAG 评测脚本（三变体消融对比）
# ============================================================
# 功能：
#   对评测集（scripts/eval_set.json）中的问题，分别用三种配置跑检索/端到端，
#   统计检索命中率与回答正确率，输出对比表，用于评估查询重写 / BM25 的贡献。
#
# 三种变体：
#   baseline   : 查询重写 开 + BM25 开（当前线上配置）
#   no_rewrite : 查询重写 关 + BM25 开
#   no_bm25    : 查询重写 开 + BM25 关（只保留 FAISS 语义检索）
#
# 用法：
#   python scripts/evaluate.py                 # 默认：检索命中率模式（快）
#   python scripts/evaluate.py --mode full     # 端到端：跑完整 Agent + 生成答案
#   python scripts/evaluate.py --limit 5       # 只跑前 5 题（冒烟测试）
#   python scripts/evaluate.py --variants baseline,no_rewrite
#   python scripts/evaluate.py --out F:/RAG学习/测试结果/eval_results.json
#
# 说明：
#   - 需要 DEEPSEEK_API_KEY 环境变量（查询重写/回答生成都要调用 LLM）
#   - 结果增量写盘，中断后可重跑续跑（已完成的题目自动跳过）
# ============================================================

import sys
import os
import io
import json
import time
import argparse

# 解决 Windows GBK 终端 emoji 打印问题（line_buffering=True：保证进度实时刷新，
# 否则 stdout 会块缓冲，长时间运行下 print 的进度要攒够才显示）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

# 项目根目录加入路径
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 离线开关：本地默认离线；云端设 HF_OFFLINE=0 联网下载。
# 必须在所有 langchain/transformers 导入前设置；config.py 也会同步这些值。
os.environ.setdefault("HF_OFFLINE", "1")
os.environ["HF_HUB_OFFLINE"] = os.environ["HF_OFFLINE"]
os.environ["TRANSFORMERS_OFFLINE"] = os.environ["HF_OFFLINE"]
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
print(f"[启动诊断] HF_OFFLINE={os.environ.get('HF_OFFLINE')} DEEPSEEK_API_KEY={'已设置' if os.environ.get('DEEPSEEK_API_KEY') else '未设置'}")

import config
from langchain_openai import ChatOpenAI
from src.rag.retriever import (
    load_retrieval_components,
    create_ensemble_retriever, retrieve_with_rerank, format_docs,
)
from src.rag.query_rewriter import rewrite_queries
from src.agent.graph import build_agent

# 默认评测集 / 输出
DEFAULT_EVAL_SET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set.json")
DEFAULT_OUT = os.path.join(ROOT, "data", "eval_results.json")

VARIANTS = ["baseline", "no_rewrite", "no_bm25"]

CATEGORY_NAMES = {
    "error_code": "错误码",
    "log_path": "日志路径",
    "operation": "操作指南",
    "routing": "路由/边界",
}


def load_eval_set(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_results(out_path):
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(results, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def apply_variant_config(variant):
    """运行前把 config 切到对应变体，并返回该变体应使用的 ensemble 检索器。"""
    rewrite_on = variant != "no_rewrite"
    config.QUERY_REWRITE_ENABLED = rewrite_on
    return rewrite_on


# ------------------------------------------------------------
# 检索命中率模式（直接检索，不跑完整 Agent，快）
# ------------------------------------------------------------
def run_retrieval(q, ensemble, reranker, rewrite_llm):
    """对单条评测返回 (retrieval_hit, seconds, detail)。"""
    question = q["question"]
    rewrite_on = config.QUERY_REWRITE_ENABLED
    t0 = time.time()
    if rewrite_on:
        queries = rewrite_queries(
            llm=rewrite_llm,
            question=question,
            history_text="",
            count=config.QUERY_REWRITE_COUNT,
        )
        merged = ""
        for i, query in enumerate(queries):
            docs = retrieve_with_rerank(
                query, ensemble, reranker, top_k=config.QUERY_REWRITE_TOP_K
            )
            merged += format_docs(docs)
            if i > 0:
                merged += "\n---\n"
    else:
        docs = retrieve_with_rerank(question, ensemble, reranker)
        merged = format_docs(docs)

    gold_terms = (q.get("gold_code"),) + tuple(q.get("gold", []))
    hit = any(g and g in merged for g in gold_terms if g)
    return hit, time.time() - t0, merged


# ------------------------------------------------------------
# 端到端模式（跑完整 Agent + 生成答案，慢但更真实）
# ------------------------------------------------------------
def run_full(q, agent, answer_llm):
    """对单条评测返回 (retrieval_hit, answer_hit, routing_ok, seconds, answer)。"""
    t0 = time.time()
    state = {
        "question": q["question"],
        "messages": [],
        "retrieved_docs": "",
        "sources": [],
        "answer": "",
        "prompt": "",
        "rewritten_queries": [],
    }
    result = agent.invoke(state)

    retrieved = result.get("retrieved_docs", "")
    sources = result.get("sources", [])

    # 检索命中：gold 关键词出现在召回内容
    gold_terms = (q.get("gold_code"),) + tuple(q.get("gold", []))
    retrieval_hit = any(g and g in retrieved for g in gold_terms if g)

    # 生成最终答案（走工具轮由 prompt 生成；无工具轮直接有 answer）
    ans = result.get("answer") or ""
    if not ans and result.get("prompt"):
        ans = answer_llm.invoke([{"role": "user", "content": result["prompt"]}]).content or ""
    answer_hit = any(g and g in ans for g in gold_terms if g)

    # 路由正确性
    expect = q.get("expect")
    routing_ok = None
    if expect == "no_tool":
        routing_ok = bool(ans) and not any("天气数据来源" in s or "时间来源" in s for s in sources)
    elif expect == "weather":
        routing_ok = any("天气数据来源" in s for s in sources)
    elif expect == "time":
        routing_ok = any("时间来源" in s for s in sources)
    elif expect == "rag":
        routing_ok = retrieval_hit

    return retrieval_hit, answer_hit, routing_ok, time.time() - t0, ans


def summarize(results, mode):
    """按变体 × 类别输出汇总表。"""
    print("\n" + "=" * 78)
    print("评测汇总" + "（检索命中率）" if mode == "retrieval" else "（端到端：命中率 / 回答正确 / 路由正确）")
    print("=" * 78)

    header = f"{'变体':<12s} | " + " | ".join(f"{CATEGORY_NAMES[c]:<10s}" for c in CATEGORY_NAMES) + " | 总计"
    print(header)
    print("-" * len(header))

    for variant in VARIANTS:
        vres = results.get("variants", {}).get(variant, {})
        if not vres:
            continue
        cells = []
        for cat in CATEGORY_NAMES:
            cat_items = {qid: r for qid, r in vres.items()
                         if r.get("category") == cat and not r.get("skipped")}
            if not cat_items:
                cells.append("   -   ")
                continue
            ok = sum(1 for r in cat_items.values() if r.get("hit"))
            if mode == "full" and cat == "routing":
                ok = sum(1 for r in cat_items.values() if r.get("routing_ok"))
            cells.append(f"{ok}/{len(cat_items)}")
        total_ok = sum(1 for r in vres.values() if not r.get("skipped") and
                       (r.get("hit") if mode == "retrieval" else r.get("routing_ok") if r.get("category") == "routing" else r.get("answer_hit")))
        total = sum(1 for r in vres.values() if not r.get("skipped"))
        pct = 100.0 * total_ok / total if total else 0.0
        print(f"{variant:<12s} | " + " | ".join(f"{c:<10s}" for c in cells) + f" | {total_ok}/{total} = {pct:.1f}%")

    # 基线对照（no_rewrite / no_bm25 相对 baseline 的差异）
    base = results.get("variants", {}).get("baseline", {})
    base_ok = sum(1 for r in base.values() if not r.get("skipped") and
                  (r.get("hit") if mode == "retrieval" else r.get("answer_hit")))
    base_total = sum(1 for r in base.values() if not r.get("skipped"))
    print("-" * len(header))
    if base_total:
        print(f"baseline 整体正确/命中: {base_ok}/{base_total} = {100.0*base_ok/base_total:.1f}%")
    for other in ("no_rewrite", "no_bm25"):
        ores = results.get("variants", {}).get(other, {})
        o_ok = sum(1 for r in ores.values() if not r.get("skipped") and
                   (r.get("hit") if mode == "retrieval" else r.get("answer_hit")))
        o_total = sum(1 for r in ores.values() if not r.get("skipped"))
        if base_total and o_total:
            delta = (o_ok / o_total - base_ok / base_total) * 100
            print(f"{other:<12s} 相对 baseline: {o_ok}/{o_total} ({delta:+.1f} pp)")


def main():
    parser = argparse.ArgumentParser(description="RAG 三变体消融评测")
    parser.add_argument("--set", default=DEFAULT_EVAL_SET, help="评测集 JSON 路径")
    parser.add_argument("--out", default=DEFAULT_OUT, help="结果输出 JSON 路径")
    parser.add_argument("--mode", choices=["retrieval", "full"], default="retrieval",
                        help="retrieval=只测检索命中率(快)；full=端到端含回答(慢)")
    parser.add_argument("--variants", default=",".join(VARIANTS), help="要跑的变体，逗号分隔")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题")
    args = parser.parse_args()

    if not config.DEEPSEEK_API_KEY:
        print("❌ 未设置 DEEPSEEK_API_KEY，评测需要调用 LLM。")
        print("   CMD:       set DEEPSEEK_API_KEY=sk-你的key")
        print("   PowerShell: $env:DEEPSEEK_API_KEY = \"sk-你的key\"")
        sys.exit(1)

    questions = load_eval_set(args.set)
    if args.limit:
        questions = questions[:args.limit]
    variants = [v for v in args.variants.split(",") if v in VARIANTS]
    print(f"评测集 {len(questions)} 题，变体: {variants}，模式: {args.mode}")

    results = load_results(args.out)
    results.setdefault("variants", {})
    results.setdefault("meta", {
        "eval_set": args.set,
        "mode": args.mode,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    # ---------- 加载模型/检索器（一次性） ----------
    print("\n加载检索组件（FAISS + Rerank + BM25）...")
    _, faiss_retriever, bm25_retriever, reranker = load_retrieval_components()

    # 两个 ensemble：双路 / 仅语义（BM25 关）
    config.ENSEMBLE_WEIGHTS = [0.5, 0.5]
    ensemble_both = create_ensemble_retriever(faiss_retriever, bm25_retriever)
    config.ENSEMBLE_WEIGHTS = [1.0, 0.0]
    ensemble_semantic = create_ensemble_retriever(faiss_retriever, bm25_retriever)
    config.ENSEMBLE_WEIGHTS = [0.5, 0.5]  # 还原默认

    rewrite_llm = ChatOpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.LLM_BASE_URL,
        model=config.LLM_MODEL,
        temperature=config.QUERY_REWRITE_TEMPERATURE,
    )
    answer_llm = ChatOpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.LLM_BASE_URL,
        model=config.LLM_MODEL,
        temperature=config.TEMPERATURE,
    )

    for variant in variants:
        rewrite_on = apply_variant_config(variant)
        agent = None
        if args.mode == "full":
            config.ENSEMBLE_WEIGHTS = [1.0, 0.0] if variant == "no_bm25" else [0.5, 0.5]
            agent = build_agent(faiss_retriever, bm25_retriever, reranker)
            config.ENSEMBLE_WEIGHTS = [0.5, 0.5]

        vres = results["variants"].setdefault(variant, {})
        print(f"\n===== 变体 {variant}（查询重写{'开' if rewrite_on else '关'}，"
              f"BM25{'开' if variant != 'no_bm25' else '关'}）=====")
        for q in questions:
            qid = q["id"]
            if qid in vres and not vres[qid].get("skipped"):
                continue  # 已跑过，续跑跳过

            try:
                if args.mode == "retrieval":
                    if q.get("category") == "routing":
                        vres[qid] = {"category": q["category"], "skipped": True}
                        print(f"   [{qid}] 跳过（路由类题目只在 full 模式评测）")
                        continue
                    ensemble = ensemble_both if variant != "no_bm25" else ensemble_semantic
                    hit, secs, _ = run_retrieval(q, ensemble, reranker, rewrite_llm)
                    vres[qid] = {"category": q["category"], "hit": hit, "seconds": round(secs, 2)}
                    print(f"   [{qid}] {'✅' if hit else '❌'} 命中  ({secs:.1f}s)  {q['question'][:40]}")
                else:
                    rh, ah, rok, secs, ans = run_full(q, agent, answer_llm)
                    is_routing = q.get("category") == "routing"
                    ok_flag = rok if is_routing else ah
                    vres[qid] = {
                        "category": q["category"],
                        "hit": rh,
                        "answer_hit": ah,
                        "routing_ok": rok,
                        "seconds": round(secs, 2),
                        "answer": ans[:300],
                    }
                    label = "✅" if ok_flag else "❌"
                    print(f"   [{qid}] {label} 检索命中={rh} 回答={ah} ({secs:.1f}s)  {q['question'][:40]}")
            except Exception as e:
                print(f"   [{qid}] ⚠️ 失败：{type(e).__name__}: {e}")
                vres[qid] = {"category": q["category"], "skipped": True, "error": str(e)}

            save_results(results, args.out)  # 增量保存

    summarize(results, args.mode)
    print(f"\n结果已保存：{args.out}")


if __name__ == "__main__":
    main()
