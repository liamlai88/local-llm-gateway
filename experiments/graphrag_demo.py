"""
实验 #13: GraphRAG 多跳检索 vs 朴素 RAG

5 道多跳测试题，对照:
- naive_rag: 向量相似 top-3 + 生成
- graphrag: 实体抽取 + 2-hop 子图 + 生成
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import graphrag


TESTS = [
    # (question, expected_keywords [必须出现一个])
    ("A100 比 A10 多多少显存？", ["56"]),
    ("ecs.gn7e 用的 GPU 比 ecs.gn7i 用的 GPU 多多少显存？", ["56"]),
    ("包月套餐 PRO 里的 GPU 实例按量付费每小时多少钱？", ["68"]),
    (
        "李四的上级负责什么 GPU 实例？",
        ["ecs.gn7", "ecs.gn7e"],
    ),  # 张三是李四上级，但张三不直接负责实例；这题考验图遍历理解
    ("张三的下属的下属负责的产品包月多少钱？", ["35000"]),
]


def grade(answer: str, expected: list) -> bool:
    return any(k.lower() in answer.lower() for k in expected)


def main():
    print(f"\n{'=' * 70}")
    print("实验 #13: GraphRAG 多跳检索 vs 朴素 RAG")
    print(f"{'=' * 70}\n")

    # Phase 1: 建图（一次性，~10 LLM 调用）
    print(">>> Phase 1: 抽三元组建图 (qwen-turbo)...\n")
    t0 = time.time()
    G = graphrag.build_graph(graphrag.KB_DOCS, verbose=True)
    build_time = time.time() - t0
    print(
        f"\n图统计: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边, 耗时 {build_time:.1f}s"
    )
    print(f"节点列表: {list(G.nodes())}\n")

    # Phase 2: 5 题对照
    results = {"naive_rag": [], "graphrag": []}

    for i, (q, expected) in enumerate(TESTS, 1):
        print(f"\n--- Q{i}: {q}")
        print(f"    预期含: {expected}")

        # Naive RAG
        try:
            r_naive = graphrag.naive_rag_query(q, graphrag.KB_DOCS, top_k=3)
            ok_naive = grade(r_naive["answer"], expected)
        except Exception as e:
            r_naive = {"answer": f"ERROR: {e}", "latency_ms": 0}
            ok_naive = False
        print(
            f"    [naive_rag] {'✓' if ok_naive else '✗'}  {r_naive['latency_ms']:.0f}ms  -> {r_naive['answer'][:80]}"
        )

        # GraphRAG
        try:
            r_graph = graphrag.graphrag_query(G, q, hops=2)
            ok_graph = grade(r_graph["answer"], expected)
        except Exception as e:
            r_graph = {"answer": f"ERROR: {e}", "latency_ms": 0, "seeds": []}
            ok_graph = False
        print(
            f"    [graphrag]  {'✓' if ok_graph else '✗'}  {r_graph['latency_ms']:.0f}ms  seeds={r_graph.get('seeds', [])}"
        )
        print(f"                  -> {r_graph['answer'][:80]}")

        results["naive_rag"].append(
            {
                "q": q,
                "answer": r_naive["answer"],
                "ok": ok_naive,
                "latency_ms": r_naive["latency_ms"],
            }
        )
        results["graphrag"].append(
            {
                "q": q,
                "answer": r_graph["answer"],
                "ok": ok_graph,
                "latency_ms": r_graph["latency_ms"],
                "seeds": r_graph.get("seeds", []),
                "context": r_graph.get("context", ""),
            }
        )

    # 汇总
    print(f"\n\n{'=' * 70}")
    print("汇总")
    print(f"{'=' * 70}")
    for mode in ["naive_rag", "graphrag"]:
        rs = results[mode]
        acc = sum(1 for r in rs if r["ok"]) / len(rs) * 100
        avg_lat = sum(r["latency_ms"] for r in rs) / len(rs)
        print(f"{mode:<14} 准确率 {acc:.1f}%  平均延迟 {avg_lat:.0f}ms")

    # 保存
    out_path = os.path.join(os.path.dirname(__file__), "graphrag_results.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "graph_stats": {
                    "nodes": G.number_of_nodes(),
                    "edges": G.number_of_edges(),
                    "build_time_s": build_time,
                },
                "graph_edges": [
                    (u, v, d["relation"]) for u, v, d in G.edges(data=True)
                ],
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n详细结果: {out_path}")


if __name__ == "__main__":
    main()
