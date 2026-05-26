"""
实验 #18B: 语义缓存 v2 — Normalize + Embed + Rerank 三层
- 同样 50 条 query 测试
- 对照 v1 baseline (threshold=0.85, 命中率 28%)
- 期望命中率提升到 50%+
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semantic_cache_v2 import SemanticCacheV2
from semantic_cache import call_llm_with_tokens
import requests

BAILIAN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


def _api_key():
    return os.environ["DASHSCOPE_API_KEY"]


def llm_judge(question: str, answer: str) -> bool:
    prompt = f"""判断答案是否合理回答了问题。
问题: {question}
答案: {answer}

合理: 答案直接、相关、信息正确
不合理: 文不对题，或回答了另一个问题

【输出 - 严格 JSON】
{{"reasonable": true/false}}

只输出 JSON。"""
    resp = requests.post(
        BAILIAN_URL,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        },
        json={
            "model": "qwen-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
        },
        timeout=30,
    )
    try:
        import re

        text = resp.json()["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(m.group(0))["reasonable"]
    except Exception:
        return None


# 同 v1 的测试集
from experiments.semantic_cache_demo import QUERIES


def run_v2(embed_th: float = 0.70, rerank_th: float = 0.80):
    cache = SemanticCacheV2(
        embed_threshold=embed_th, rerank_threshold=rerank_th, top_k=5
    )
    trace = []
    print(f"\n>>> v2 配置: embed_th={embed_th}, rerank_th={rerank_th}")

    for i, (group, q) in enumerate(QUERIES, 1):
        r = cache.get_or_compute(q, call_llm_with_tokens)
        trace.append(
            {
                "i": i,
                "group": group,
                "query": q,
                "normalized": r["normalized_query"],
                "from_cache": r["from_cache"],
                "matched": r["matched_query"],
                "matched_norm": r["matched_norm"],
                "embed_sim": r["embed_sim"],
                "rerank_score": r["rerank_score"],
                "answer": r["answer"][:80],
                "tokens_saved": r["tokens_saved"],
                "latency_ms": r["latency_ms"],
            }
        )
        flag = "HIT" if r["from_cache"] else "MISS"
        rs = f"rerank={r['rerank_score']:.3f}" if r["rerank_score"] else ""
        print(f"  [{group}] {i:2d} {flag:4s} embed={r['embed_sim']:.3f} {rs}  {q[:30]}")
        if r["from_cache"]:
            print(
                f"        ↳ matched: {r['matched_query'][:40]}  (q→norm: {r['normalized_query'][:30]})"
            )

    by_group = {}
    for t in trace:
        g = t["group"]
        by_group.setdefault(g, {"hit": 0, "total": 0, "latency_sum": 0})
        by_group[g]["total"] += 1
        by_group[g]["latency_sum"] += t["latency_ms"]
        if t["from_cache"]:
            by_group[g]["hit"] += 1

    summary = cache.summary()
    print(
        f"\n  ── 总命中: {summary['hits']}/{summary['queries']} ({summary['hit_rate']:.0%})  节省 token: {summary['tokens_saved']}"
    )

    # 错误命中率
    print("  ── 检查 C+D 组命中合理性...")
    judged = []
    for t in trace:
        if t["group"] in ("C", "D") and t["from_cache"]:
            ok = llm_judge(t["query"], t["answer"])
            judged.append((t["i"], t["group"], t["query"], t["answer"], ok))
    wrong_hits = sum(1 for _, _, _, _, ok in judged if ok is False)
    print(
        f"     {len(judged)} 条命中, {wrong_hits} 条不合理 ({wrong_hits / max(1, len(judged)):.0%})"
    )

    return {
        "config": {"embed_threshold": embed_th, "rerank_threshold": rerank_th},
        "summary": summary,
        "by_group": by_group,
        "trace": trace,
        "judged": [
            {"i": i, "group": g, "query": q, "answer": a, "reasonable": ok}
            for i, g, q, a, ok in judged
        ],
        "wrong_hits": wrong_hits,
    }


def main():
    print(f"\n{'=' * 70}")
    print("实验 #18B: 语义缓存 v2（Normalize + Rerank）")
    print(f"{'=' * 70}")

    result = run_v2(embed_th=0.70, rerank_th=0.50)

    # 对照 v1（从 v1 results 读 0.85 那行）
    v1_path = os.path.join(os.path.dirname(__file__), "semantic_cache_results.json")
    v1_data = None
    if os.path.exists(v1_path):
        with open(v1_path) as f:
            v1_results = json.load(f)
        v1_data = next((r for r in v1_results if r["threshold"] == 0.85), None)

    # 对照表
    print(f"\n\n{'=' * 70}")
    print("v1 vs v2 对比")
    print(f"{'=' * 70}")
    print(f"{'指标':<22}{'v1 (th=0.85)':<18}{'v2 (norm+rerank)':<18}{'差值':<10}")
    if v1_data:
        v1s = v1_data["summary"]
        v1g = v1_data["by_group"]
        v2s = result["summary"]
        v2g = result["by_group"]
        rows = [
            (
                "总命中率",
                f"{v1s['hit_rate']:.0%}",
                f"{v2s['hit_rate']:.0%}",
                f"+{(v2s['hit_rate'] - v1s['hit_rate']) * 100:.0f}%",
            ),
            (
                "C 同义改写",
                f"{v1g['C']['hit']}/10",
                f"{v2g['C']['hit']}/10",
                f"+{v2g['C']['hit'] - v1g['C']['hit']}",
            ),
            (
                "D 相关但不同",
                f"{v1g['D']['hit']}/10",
                f"{v2g['D']['hit']}/10",
                f"+{v2g['D']['hit'] - v1g['D']['hit']}",
            ),
            (
                "E 完全不同",
                f"{v1g['E']['hit']}/10",
                f"{v2g['E']['hit']}/10",
                f"+{v2g['E']['hit'] - v1g['E']['hit']}",
            ),
            (
                "节省 token",
                str(v1s["tokens_saved"]),
                str(v2s["tokens_saved"]),
                f"+{v2s['tokens_saved'] - v1s['tokens_saved']}",
            ),
            (
                "错误命中",
                str(v1_data["wrong_hits"]),
                str(result["wrong_hits"]),
                f"+{result['wrong_hits'] - v1_data['wrong_hits']}",
            ),
        ]
        for r in rows:
            print(f"{r[0]:<22}{r[1]:<18}{r[2]:<18}{r[3]:<10}")

    # 看 v2 的 normalize 效果（取几个例子）
    print("\n--- Query Normalization 效果样本 ---")
    for t in result["trace"][:15]:
        if t["query"] != t["normalized"]:
            print(f"  原: {t['query']:<35} → 标准: {t['normalized']}")

    out_path = os.path.join(os.path.dirname(__file__), "semantic_cache_v2_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果: {out_path}")


if __name__ == "__main__":
    main()
