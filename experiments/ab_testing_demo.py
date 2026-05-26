"""
实验 #19: A/B Testing + 在线评测 + 自动回滚
- 两个 prompt 变体（A=简短 / B=CoT 结构化）
- 30 条客服 query 模拟 30 个用户请求
- 实时打 RAGAS faithfulness/relevance
- Welch's t-test 显著性检验
- 自动决策：promote / rollback / hold
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ab_testing import ABTester

# ===== 两个变体 =====
PROMPT_A = """你是阿里云销售客服。简洁直接回答用户问题，控制在 2 句话内。"""

PROMPT_B = """你是阿里云销售客服。回答时遵循以下步骤：
1. 先理解用户需求（用 1 句话复述）
2. 给出核心答案（数字 / 配置 / 价格）
3. 补充 1 个相关建议或注意事项

要专业，结构清晰，每个回答 3-4 句。"""

# ===== 模拟 KB（给 RAGAS 评 faithfulness 用）=====
KB_CONTEXT = """ecs.gn7e: NVIDIA A100 GPU, 80GB 显存, 312 TFLOPS, 按量 ¥68/小时, 包月 ¥35000
ecs.gn7i: NVIDIA A10 GPU, 24GB 显存, 125 TFLOPS, 按量 ¥18/小时, 包月 ¥9800
包月套餐 PRO: 含 ecs.gn7e + 1TB 存储 + 100GB 流量, ¥40000/月
包月套餐 LITE: 含 ecs.gn7i + 500GB 存储 + 50GB 流量, ¥12000/月
SLA: 99.95% 可用性, 故障 30 分钟内响应
退款: 包月可申请按未使用天数比例退款，需在购买后 7 天内
技术支持: 工单 24 小时响应, 紧急工单 30 分钟内"""

# ===== 30 条客服 query (模拟用户)=====
QUERIES = [
    ("u001", "ecs.gn7e 配什么 GPU？"),
    ("u002", "包月套餐 PRO 多少钱？"),
    ("u003", "A100 显存多大"),
    ("u004", "怎么开通 GPU 实例"),
    ("u005", "ecs.gn7i 适合什么场景"),
    ("u006", "退款政策"),
    ("u007", "包年和包月哪个划算"),
    ("u008", "A10 和 A100 性能差距"),
    ("u009", "技术支持响应时间"),
    ("u010", "我能试用吗"),
    ("u011", "包月套餐 LITE 比 PRO 便宜多少"),
    ("u012", "gn7e 按量付费每小时多少"),
    ("u013", "SLA 是多少"),
    ("u014", "包月套餐含哪些资源"),
    ("u015", "A100 算力多强"),
    ("u016", "退款多久到账"),
    ("u017", "买包月比按量付费划算吗"),
    ("u018", "怎么提交工单"),
    ("u019", "ecs.gn7e 包月一年总共多少钱"),
    ("u020", "新加坡可以使用 ecs.gn7e 吗"),
    ("u021", "gn7i 适合大模型训练吗"),
    ("u022", "我们公司想批量购买"),
    ("u023", "试用账号有什么限制"),
    ("u024", "工单响应多快"),
    ("u025", "包月套餐 PRO 流量超了怎么算"),
    ("u026", "GPU 实例多久能交付"),
    ("u027", "A10 性价比怎么样"),
    ("u028", "包月可以提前退订吗"),
    ("u029", "客服电话是多少"),
    ("u030", "ecs.gn7e 包月有折扣吗"),
]


def main():
    print(f"\n{'=' * 70}")
    print("实验 #19: A/B Testing + 在线评测 + 自动回滚")
    print(f"{'=' * 70}\n")

    ab = ABTester(
        variants={"A_simple": PROMPT_A, "B_cot": PROMPT_B},
        traffic={"A_simple": 0.5, "B_cot": 0.5},
    )

    print(f"测试 {len(QUERIES)} 条 query，按 user_id 哈希分流 50/50\n")

    for user_id, query in QUERIES:
        r = ab.call(user_id, query, context=KB_CONTEXT)
        v = r["variant"][:8]
        f = r["faithfulness"]
        rel = r["relevance"]
        f_str = f"{f:.2f}" if f is not None else "N/A"
        rel_str = f"{rel:.2f}" if rel is not None else "N/A"
        print(
            f"  [{v}] {user_id} F={f_str} R={rel_str} len={r['length']:3d}  {query[:30]}"
        )

    # 汇总
    print(f"\n\n{'=' * 70}")
    print("Variant 汇总")
    print(f"{'=' * 70}")
    s = ab.stats()
    for v in ["A_simple", "B_cot"]:
        if v not in s:
            continue
        d = s[v]
        print(f"\n{v} (n={d['n']}):")
        print(f"  Faithfulness:    {d['faithfulness']:.3f}")
        print(f"  Relevance:       {d['relevance']:.3f}")
        print(f"  平均长度:        {d['length']:.0f} 字符")
        print(f"  平均延迟:        {d['latency_ms']:.0f} ms")
        print(
            f"  平均 input/out:  {d['avg_input_tokens']:.0f} / {d['avg_output_tokens']:.0f} token"
        )

    # 显著性
    print(f"\n{'=' * 70}")
    print("显著性检验 (Welch's t-test, B vs A)")
    print(f"{'=' * 70}")
    print(
        f"{'指标':<18}{'mean_A':<10}{'mean_B':<10}{'delta':<10}{'t':<10}{'p':<10}{'显著':<8}"
    )
    for metric, t in s["_tests"].items():
        if t.get("t") is None:
            continue
        sig = "✓" if t["significant"] else "—"
        print(
            f"{metric:<18}{t['mean_a']:<10.3f}{t['mean_b']:<10.3f}{t['delta']:<+10.3f}{t['t']:<10.3f}{t['p']:<10.4f}{sig:<8}"
        )

    # 自动决策
    print(f"\n{'=' * 70}")
    print("自动决策引擎")
    print(f"{'=' * 70}")
    d = ab.decision(rollback_threshold=0.15)
    print(f"  决策: {d['decision']}")
    print(f"  理由: {d['reason']}")
    if "rollback_to" in d:
        print(f"  → 自动回滚到: {d['rollback_to']}")
    if "promote" in d:
        print(f"  → 自动升级: {d['promote']}")

    # 持久化
    out_path = os.path.join(os.path.dirname(__file__), "ab_testing_results.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "stats": s,
                "decision": d,
                "records": ab.records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n详细结果: {out_path}")


if __name__ == "__main__":
    main()
