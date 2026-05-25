"""
实验 #14: Context Engineering 四策略对照
- Baseline (全量) / Write (scratchpad) / Select (top-3) / Compress (每3轮)
- 10 轮客服对话, 第 10 轮考召回（姓名/公司/产品/价格等）
- 指标: 累计 input token / 召回准确率 / 平均延迟
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import context_engineering as ce


SYSTEM_PROMPT = "你是阿里云的资深销售客服，专业、友好。回答用户问题时基于上下文中提到的信息，不要编造。"

# ===== 10 轮对话脚本 =====
# 前 9 轮: 用户陆续提供信息 + 问问题
# 第 10 轮: 召回测试，必须用到分布在各轮的事实
DIALOGUE = [
    "你好，我叫王明，是 ACME 数据公司的 CTO，我们想做大模型推理服务。",
    "我们主要业务是金融风控，预计每天处理 100 万次推理请求。",
    "请简单介绍下你们的 GPU 实例选项。",
    "ecs.gn7e 配 A100 是吧？显存多大？",
    "对比 ecs.gn7i 的 A10，性能差距大吗？",
    "我倾向选 ecs.gn7e，包月价格是多少？",
    "顺便聊下，你们公司最近的 AI 大会在哪个城市？",
    "我们如果选了 ecs.gn7e，要怎么对接销售？",
    "好的，那帮我看看你们最近有什么活动。",
    # 第 10 轮: 召回测试
    "我们今天聊了不少，帮我总结一下：(1) 我的名字和公司是什么？(2) 我们的业务是什么？(3) 我倾向选哪个产品？(4) 这个产品配的 GPU 型号和显存？(5) 这个产品包月价格？",
]

# 第 10 轮的"应答应包含"关键词（用于打分）
RECALL_CHECKS = [
    ("王明", "姓名"),
    ("ACME", "公司"),
    ("金融风控", "业务"),
    ("ecs.gn7e", "产品"),
    ("A100", "GPU 型号"),
    ("80GB", "显存"),
    ("35000", "价格"),
]


def run_one_strategy(strategy: ce.ContextStrategy, dialogue: list) -> dict:
    """跑完整 10 轮对话，返回所有指标"""
    metrics = {
        "name": strategy.name,
        "turns": [],
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "overhead_tokens": 0,  # Write/Compress 额外的 LLM 调用
        "total_latency_ms": 0,
        "final_answer": "",
    }

    for i, user_msg in enumerate(dialogue, 1):
        t0 = time.time()
        messages = strategy.build_messages(user_msg)
        r = ce.call_llm(messages, max_tokens=500)
        latency = (time.time() - t0) * 1000

        strategy.add_turn(user_msg, r["answer"])

        # 策略特有的后处理
        overhead = 0
        if isinstance(strategy, ce.WriteStrategy):
            overhead = strategy.update_scratchpad(user_msg, r["answer"])
        elif isinstance(strategy, ce.CompressStrategy):
            overhead = strategy.maybe_compress()

        metrics["turns"].append(
            {
                "turn": i,
                "user": user_msg[:50],
                "assistant": r["answer"][:80],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "overhead_tokens": overhead,
                "latency_ms": round(latency, 1),
            }
        )
        metrics["total_input_tokens"] += r["input_tokens"]
        metrics["total_output_tokens"] += r["output_tokens"]
        metrics["overhead_tokens"] += overhead
        metrics["total_latency_ms"] += latency

        if i == len(dialogue):
            metrics["final_answer"] = r["answer"]

        print(
            f"  [{strategy.name}] Turn {i:2d}: in={r['input_tokens']:4d} out={r['output_tokens']:3d} overhead={overhead:3d} {latency:.0f}ms"
        )

    # 召回评分
    final = metrics["final_answer"]
    metrics["recall_hits"] = [(kw, label, kw in final) for kw, label in RECALL_CHECKS]
    metrics["recall_score"] = sum(1 for _, _, hit in metrics["recall_hits"] if hit)

    return metrics


def main():
    print(f"\n{'=' * 70}")
    print("实验 #14: Context Engineering 四策略")
    print(f"对话轮数: {len(DIALOGUE)}, 召回考点: {len(RECALL_CHECKS)}")
    print(f"{'=' * 70}\n")

    strategies = [
        ce.BaselineStrategy(SYSTEM_PROMPT),
        ce.WriteStrategy(SYSTEM_PROMPT),
        ce.SelectStrategy(SYSTEM_PROMPT, top_k=3),
        ce.CompressStrategy(SYSTEM_PROMPT, compress_every=3, keep_recent=2),
    ]

    all_results = []
    for s in strategies:
        print(f"\n>>> 策略: {s.name}")
        m = run_one_strategy(s, DIALOGUE)
        all_results.append(m)

    # 汇总
    print(f"\n\n{'=' * 70}")
    print("汇总")
    print(f"{'=' * 70}")
    print(
        f"{'策略':<12}{'累计 input':<14}{'overhead':<12}{'output':<10}{'召回':<10}{'平均延迟(ms)':<14}"
    )
    for m in all_results:
        avg_lat = m["total_latency_ms"] / len(DIALOGUE)
        print(
            f"{m['name']:<12}{m['total_input_tokens']:<14}{m['overhead_tokens']:<12}{m['total_output_tokens']:<10}{m['recall_score']}/{len(RECALL_CHECKS):<8}{avg_lat:<14.0f}"
        )

    print("\n召回项细节:")
    print(f"{'考点':<12}" + "".join(f"{m['name']:<12}" for m in all_results))
    for i, (kw, label) in enumerate(RECALL_CHECKS):
        line = f"{label}({kw})".ljust(12)
        for m in all_results:
            mark = "✓" if m["recall_hits"][i][2] else "✗"
            line += f"{mark:<12}"
        print(line)

    # 保存
    out_path = os.path.join(
        os.path.dirname(__file__), "context_engineering_results.json"
    )
    with open(out_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果: {out_path}")


if __name__ == "__main__":
    main()
