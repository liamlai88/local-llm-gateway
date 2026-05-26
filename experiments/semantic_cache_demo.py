"""
实验 #18: 语义缓存降本对照
- 50 条 query 分 5 组（原始/重复/同义/相关/无关）
- 4 个 threshold (0.85 / 0.90 / 0.95 / 0.98)
- 指标：命中率 / token 节省 / 平均延迟 / 错误命中率
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semantic_cache import SemanticCache, call_llm_with_tokens
import requests

BAILIAN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


def _api_key():
    return os.environ["DASHSCOPE_API_KEY"]


def llm_judge(question: str, answer: str) -> bool:
    """判断 answer 是否合理回答了 question。返回 True/False"""
    prompt = f"""判断下面的答案是否合理回答了问题。
问题: {question}
答案: {answer}

【判断规则】
- 合理: 答案直接、相关、信息正确（即使简略）
- 不合理: 答案文不对题，或回答的是另一个问题

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
        text = resp.json()["choices"][0]["message"]["content"]
        import re

        m = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(m.group(0))["reasonable"]
    except Exception:
        return None


# ========== 测试集 (50 条) ==========
QUERIES = [
    # A. 10 条原始 (建立缓存)
    ("A", "ecs.gn7e 实例配什么 GPU？"),
    ("A", "包月套餐 PRO 多少钱？"),
    ("A", "A100 显存多大？"),
    ("A", "怎么开通 GPU 实例？"),
    ("A", "ecs.gn7i 适合什么场景？"),
    ("A", "退款政策是什么？"),
    ("A", "包年和包月哪个划算？"),
    ("A", "A10 和 A100 性能差距多大？"),
    ("A", "技术支持响应时间是多少？"),
    ("A", "我能试用吗？"),
    # B. 10 条完全重复 (期望 100% 命中)
    ("B", "ecs.gn7e 实例配什么 GPU？"),
    ("B", "包月套餐 PRO 多少钱？"),
    ("B", "A100 显存多大？"),
    ("B", "怎么开通 GPU 实例？"),
    ("B", "ecs.gn7i 适合什么场景？"),
    ("B", "退款政策是什么？"),
    ("B", "包年和包月哪个划算？"),
    ("B", "A10 和 A100 性能差距多大？"),
    ("B", "技术支持响应时间是多少？"),
    ("B", "我能试用吗？"),
    # C. 10 条同义改写 (关键测试)
    ("C", "ecs.gn7e 上面用的是哪款 GPU?"),
    ("C", "PRO 套餐的价格是多少呢"),
    ("C", "A100 GPU 内存是多少 GB"),
    ("C", "GPU 实例怎么购买和开通"),
    ("C", "gn7i 实例的应用场景"),
    ("C", "你们退款规则"),
    ("C", "买包年合算还是包月合算"),
    ("C", "A10 跟 A100 哪个性能更强"),
    ("C", "客服响应大概多久"),
    ("C", "可以免费试一下吗"),
    # D. 10 条相关但不同 (边界 - 同主题不同关注点)
    ("D", "ecs.gn7e 几个 CPU 核心？"),  # 同实例，不同属性
    ("D", "PRO 套餐包含什么服务？"),  # 同套餐，不同维度
    ("D", "A100 算力有多强？"),  # 同 GPU，不同属性
    ("D", "GPU 实例创建后多久能用？"),  # 同主题，不同问题
    ("D", "gn7i 比 gn7e 便宜多少？"),  # 相关比较
    ("D", "我们公司想批量购买打折吗？"),  # 价格相关，新主题
    ("D", "包月可以中途升级吗？"),  # 套餐相关，新功能
    ("D", "A10 适合训练大模型吗？"),  # GPU 相关，新场景
    ("D", "工单怎么提交？"),  # 支持相关，新方式
    ("D", "试用账号有什么限制？"),  # 试用相关，新角度
    # E. 10 条完全不同主题 (期望 0% 命中)
    ("E", "今天上海天气怎么样？"),
    ("E", "推荐一本机器学习的书"),
    ("E", "怎么做番茄炒蛋"),
    ("E", "iPhone 15 多少钱"),
    ("E", "北京到上海高铁多久"),
    ("E", "Python 怎么学"),
    ("E", "美元对人民币汇率"),
    ("E", "明天周几"),
    ("E", "巴黎奥运会金牌榜"),
    ("E", "Linux 怎么查磁盘空间"),
]


def run_one_threshold(threshold: float, judge_subset_size: int = 20):
    """跑一遍 50 条 query，返回汇总"""
    cache = SemanticCache(threshold=threshold)
    trace = []

    print(f"\n>>> Threshold = {threshold}")
    for i, (group, q) in enumerate(QUERIES, 1):
        r = cache.get_or_compute(q, call_llm_with_tokens)
        trace.append(
            {
                "i": i,
                "group": group,
                "query": q,
                "from_cache": r["from_cache"],
                "similarity": r["similarity"],
                "matched": r["matched_query"],
                "answer": r["answer"][:80],
                "tokens_saved": r["tokens_saved"],
                "latency_ms": r["latency_ms"],
            }
        )
        flag = "HIT" if r["from_cache"] else "MISS"
        print(f"  [{group}] {i:2d} {flag:4s} sim={r['similarity']:.3f}  {q[:30]}")

    # 按组统计命中率
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
        f"  ── 总命中: {summary['hits']}/{summary['queries']} ({summary['hit_rate']:.0%})  节省 token: {summary['tokens_saved']}"
    )

    # 对命中样本做错误命中率检查（仅 C+D 组，A/B 不会有问题）
    print("  ── 检查 C+D 组命中答案合理性...")
    judged = []
    for t in trace:
        if t["group"] in ("C", "D") and t["from_cache"]:
            ok = llm_judge(t["query"], t["answer"])
            judged.append((t["i"], t["group"], t["query"], t["answer"], ok))
    wrong_hits = sum(1 for _, _, _, _, ok in judged if ok is False)
    print(
        f"     检查 {len(judged)} 条命中, {wrong_hits} 条不合理（错误命中率: {wrong_hits / max(1, len(judged)):.0%}）"
    )

    return {
        "threshold": threshold,
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
    print("实验 #18: 语义缓存 — 阈值扫描")
    print(f"{'=' * 70}")

    all_results = []
    for th in [0.85, 0.90, 0.95, 0.98]:
        r = run_one_threshold(th)
        all_results.append(r)

    # 汇总
    print(f"\n\n{'=' * 70}")
    print("阈值对照表")
    print(f"{'=' * 70}")
    header = f"{'threshold':<12}{'命中率':<10}{'A':<8}{'B':<8}{'C':<8}{'D':<8}{'E':<8}{'错误命中':<10}{'省 token':<10}"
    print(header)
    for r in all_results:
        s = r["summary"]
        g = r["by_group"]
        row = f"{r['threshold']:<12.2f}{s['hit_rate']:<10.0%}"
        for grp in ["A", "B", "C", "D", "E"]:
            x = g.get(grp, {"hit": 0, "total": 0})
            row += f"{x['hit']}/{x['total']:<6}"
        row += f"{r['wrong_hits']:<10}{s['tokens_saved']:<10}"
        print(row)

    out_path = os.path.join(os.path.dirname(__file__), "semantic_cache_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果: {out_path}")


if __name__ == "__main__":
    main()
