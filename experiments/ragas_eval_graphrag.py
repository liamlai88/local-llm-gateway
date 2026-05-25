"""
实验 #15: 用 RAGAS-Lite 评测 GraphRAG vs Naive RAG (#13 的延伸)

数据源: experiments/graphrag_results.json
对每个样本算 4 个指标:
  - Faithfulness (反幻觉)
  - Answer Relevance (切题)
  - Context Precision (噪声率)
  - Context Recall (该召回的都召回了吗)
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ragas_lite as rl


# 用 #13 的设计文档构造 ground truth（每题的"理想答案"）
GROUND_TRUTH = {
    "A100 比 A10 多多少显存？": "NVIDIA A100 显存 80GB，NVIDIA A10 显存 24GB，A100 比 A10 多 56GB。",
    "ecs.gn7e 用的 GPU 比 ecs.gn7i 用的 GPU 多多少显存？": "ecs.gn7e 用 A100 (80GB)，ecs.gn7i 用 A10 (24GB)，相差 56GB。",
    "包月套餐 PRO 里的 GPU 实例按量付费每小时多少钱？": "包月套餐 PRO 包含 ecs.gn7e，ecs.gn7e 按量付费 ¥68/小时。",
    "李四的上级负责什么 GPU 实例？": "李四的上级是张三（技术 VP），但张三不直接负责具体 GPU 实例，无法确定。",
    "张三的下属的下属负责的产品包月多少钱？": "张三的下属是李四，李四的下属是王五，王五负责 ecs.gn7e，包月 ¥35000。",
}


def chunk_naive_context(question: str, full_context_str: str) -> list:
    """Naive RAG 没记录 chunk 列表，但我们知道 top_k=3，按 \\n 切"""
    return [c.strip() for c in full_context_str.split("\n") if c.strip()]


def chunk_graph_context(context_str: str) -> list:
    """GraphRAG 的 context 是每行一个三元组"""
    return [c.strip() for c in context_str.split("\n") if c.strip()]


def main():
    # 读取 #13 结果
    src = os.path.join(os.path.dirname(__file__), "graphrag_results.json")
    with open(src) as f:
        data = json.load(f)

    naive_results = data["results"]["naive_rag"]
    graph_results = data["results"]["graphrag"]

    print(f"\n{'=' * 70}")
    print("实验 #15: RAGAS-Lite 评测 GraphRAG vs Naive RAG")
    print(f"样本数: {len(naive_results)} 题 × 2 模式")
    print(f"{'=' * 70}\n")

    # 重建 naive RAG 的 context（从 #13 那份用 KB_DOCS top-3，简化为全 KB）
    import graphrag as gr

    kb_all = "\n".join(gr.KB_DOCS)

    eval_results = {"naive_rag": [], "graphrag": []}

    for i, (nr, gr_r) in enumerate(zip(naive_results, graph_results)):
        q = nr["q"]
        gt = GROUND_TRUTH.get(q, "")
        print(f"\n--- Q{i + 1}: {q[:50]}")

        # Naive RAG
        if nr.get("ok") is not None and "ERROR" not in nr["answer"]:
            print("  [naive_rag] 评测中...")
            # naive 没存 context 细节，用 top-3 KB 近似（实际是 cosine top-3，这里简化用全 KB）
            naive_eval = {
                "q": q,
                "answer": nr["answer"],
                "metrics": {
                    "faithfulness": rl.faithfulness(nr["answer"], kb_all),
                    "answer_relevance": rl.answer_relevance(q, nr["answer"]),
                    "context_precision": rl.context_precision(q, gr.KB_DOCS),
                    "context_recall": rl.context_recall(gt, kb_all) if gt else None,
                },
            }
            eval_results["naive_rag"].append(naive_eval)
            print(
                f"    F={naive_eval['metrics']['faithfulness']['score']:.2f}  "
                f"R={naive_eval['metrics']['answer_relevance']['score']:.2f}  "
                f"CP={naive_eval['metrics']['context_precision']['score']:.2f}  "
                f"CR={naive_eval['metrics']['context_recall']['score']:.2f}"
            )

        # GraphRAG
        if "ERROR" not in gr_r["answer"]:
            print("  [graphrag]  评测中...")
            ctx = gr_r.get("context", "")
            chunks = chunk_graph_context(ctx)
            graph_eval = {
                "q": q,
                "answer": gr_r["answer"],
                "seeds": gr_r.get("seeds", []),
                "metrics": {
                    "faithfulness": rl.faithfulness(gr_r["answer"], ctx),
                    "answer_relevance": rl.answer_relevance(q, gr_r["answer"]),
                    "context_precision": rl.context_precision(q, chunks),
                    "context_recall": rl.context_recall(gt, ctx) if gt else None,
                },
            }
            eval_results["graphrag"].append(graph_eval)
            print(
                f"    F={graph_eval['metrics']['faithfulness']['score']:.2f}  "
                f"R={graph_eval['metrics']['answer_relevance']['score']:.2f}  "
                f"CP={graph_eval['metrics']['context_precision']['score']:.2f}  "
                f"CR={graph_eval['metrics']['context_recall']['score']:.2f}"
            )

    # 汇总
    print(f"\n\n{'=' * 70}")
    print("汇总（4 指标平均分）")
    print(f"{'=' * 70}")
    print(
        f"{'模式':<14}{'Faithful':<12}{'Relevance':<12}{'CtxPrec':<12}{'CtxRecall':<12}"
    )
    for mode in ["naive_rag", "graphrag"]:
        rs = eval_results[mode]
        if not rs:
            continue

        def avg(metric):
            vals = [
                r["metrics"][metric]["score"] for r in rs if r["metrics"].get(metric)
            ]
            return sum(vals) / len(vals) if vals else 0.0

        print(
            f"{mode:<14}{avg('faithfulness'):<12.3f}{avg('answer_relevance'):<12.3f}"
            f"{avg('context_precision'):<12.3f}{avg('context_recall'):<12.3f}"
        )

    out = os.path.join(os.path.dirname(__file__), "ragas_eval_results.json")
    with open(out, "w") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果: {out}")


if __name__ == "__main__":
    main()
