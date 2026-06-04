"""
实验 #22 本体化 RAG：自由抽取 vs 本体约束抽取（能否抬高 #13 的 75% 天花板）

#13 结论：GraphRAG 75% < Naive 100%，病根是"自由抽三元组"质量差——
实体不归一（A100/NVIDIA A100/A100 GPU 各成一个节点）、关系类型五花八门、噪声边多。

本体化的赌注：先用"业务访谈"提炼一套领域本体(schema)——实体类型 + 受控关系词表 + 实体归一规则，
让 LLM 只能按 schema 抽取，从而抬高抽取质量这个天花板。

复用 graphrag.py（#13）：build/query/naive 基础设施不动，只替换"抽取器"。
两臂对照：
  free      = graphrag.build_graph         （#13 原样，自由抽取）
  ontology  = build_graph_ontology         （本体约束 + 归一）
度量：图质量（节点/边/关系类型数/非法关系边/实体碎片化）+ 5 道多跳题准确率。
运行：export DASHSCOPE_API_KEY=... ; python experiments/ontology_rag.py
"""

import os
import re
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import graphrag  # noqa: E402
import networkx as nx  # noqa: E402

RESULTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ontology_rag_results.json"
)

# ========== 领域本体（业务访谈产物：人工定义的 schema）==========
# 实体类型 + 受控关系词表（每种关系约束连接的实体类型）+ 实体归一规则
ENTITY_TYPES = ["套餐", "实例", "GPU", "人员", "规格值"]
RELATION_VOCAB = {
    "包含": ("套餐", "实例"),
    "总价": ("套餐", "规格值"),
    "使用": ("实例", "GPU"),
    "按量价格": ("实例", "规格值"),
    "包月价格": ("实例", "规格值"),
    "显存": ("GPU", "规格值"),
    "算力": ("GPU", "规格值"),
    "下属": ("人员", "人员"),
    "负责": ("人员", "实例"),
}
# 实体归一：把各种表面写法映射到规范名（canonical），消除碎片化
CANONICAL = {
    "ecs.gn7e": "ecs.gn7e 实例",
    "ecs.gn7e-c12g1.3xlarge": "ecs.gn7e 实例",
    "ecs.gn7i": "ecs.gn7i 实例",
    "ecs.gn7i-c8g1.2xlarge": "ecs.gn7i 实例",
    "a100": "NVIDIA A100",
    "nvidia a100": "NVIDIA A100",
    "a100 gpu": "NVIDIA A100",
    "a10": "NVIDIA A10",
    "nvidia a10": "NVIDIA A10",
    "a10 gpu": "NVIDIA A10",
    "pro": "包月套餐 PRO",
    "包月套餐pro": "包月套餐 PRO",
    "lite": "包月套餐 LITE",
    "包月套餐lite": "包月套餐 LITE",
}
# 核心概念锚点：用于度量"实体碎片化"（同一概念被拆成几个节点）
CORE_CONCEPTS = {
    "A100": ["a100"],
    "A10": ["a10"],
    "gn7e": ["gn7e"],
    "gn7i": ["gn7i"],
    "PRO": ["pro", "套餐 pro"],
    "LITE": ["lite", "套餐 lite"],
    "张三": ["张三"],
    "李四": ["李四"],
    "王五": ["王五"],
}


def normalize_entity(e: str) -> str:
    key = e.strip().lower()
    if key in CANONICAL:
        return CANONICAL[key]
    for k, v in CANONICAL.items():
        if k in key:  # 子串命中（如 "ecs.gn7e-c12g1.3xlarge 实例"）
            return v
    return e.strip()


# ========== 本体约束抽取 prompt ==========
def _ontology_prompt(doc: str) -> str:
    rels = "\n".join(f"  - {r}: ({h}) → ({t})" for r, (h, t) in RELATION_VOCAB.items())
    canon = "\n".join(
        f"  - {k} → {v}"
        for k, v in [
            ("A100/NVIDIA A100/A100 GPU", "NVIDIA A100"),
            ("ecs.gn7e 及其全称", "ecs.gn7e 实例"),
            ("PRO/包月套餐PRO", "包月套餐 PRO"),
        ]
    )
    return f"""从文档抽取三元组，**只能使用下面本体定义的关系类型**，并把实体名归一到规范名。

【实体类型】{", ".join(ENTITY_TYPES)}

【受控关系词表】（只能用这些关系；括号是允许连接的实体类型）
{rels}

【实体归一规则】（同一实体的不同写法必须统一）
{canon}
  - 数字+单位作为规格值整体（如 "80GB" "¥68/小时" "312 TFLOPS"）

【规则】
- 关系必须严格取自上面词表，禁止自创（如禁止用"容量/内存/价钱"，显存统一用"显存"）
- 主语/宾语必须归一到规范名
- 抽不出符合 schema 的就少抽，不要硬凑

【输出格式 - 严格 JSON】
{{"triples": [["主语", "关系", "宾语"], ...]}}

文档: {doc}"""


def extract_triples_ontology(doc: str):
    out = graphrag.call_llm(_ontology_prompt(doc))
    try:
        data = json.loads(re.search(r"\{.*\}", out, re.DOTALL).group(0))
        triples = []
        for t in data["triples"]:
            if len(t) != 3:
                continue
            s, r, o = normalize_entity(t[0]), t[1].strip(), normalize_entity(t[2])
            triples.append((s, r, o))
        return triples
    except Exception as e:
        print(f"[WARN] 本体抽取失败: {e}")
        return []


def build_graph_ontology(docs, verbose=False) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for i, doc in enumerate(docs):
        for s, r, o in extract_triples_ontology(doc):
            G.add_edge(s, o, relation=r, source_doc=i)
            if verbose:
                print(f"    ({s}) --[{r}]--> ({o})")
    return G


# ========== 图质量度量 ==========
def measure_graph(G: nx.MultiDiGraph) -> dict:
    rels = [d["relation"] for _, _, d in G.edges(data=True)]
    distinct_rels = sorted(set(rels))
    allowed = set(RELATION_VOCAB.keys())
    invalid_edges = sum(1 for r in rels if r not in allowed)
    # 实体碎片化：每个核心概念被拆成几个节点（理想=1），超出部分即碎片
    nodes_lower = [(n, n.lower()) for n in G.nodes()]
    frag = 0
    frag_detail = {}
    for concept, anchors in CORE_CONCEPTS.items():
        hit = [n for n, nl in nodes_lower if any(a in nl for a in anchors)]
        frag_detail[concept] = len(hit)
        if len(hit) > 1:
            frag += len(hit) - 1  # 多出来的就是碎片
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "distinct_relation_types": len(distinct_rels),
        "relation_types": distinct_rels,
        "invalid_relation_edges": invalid_edges,  # 不在本体词表内的边数
        "entity_fragments": frag,  # 实体碎片化总数（0=完美归一）
        "fragment_detail": frag_detail,
    }


# ========== 多跳测试集（与 #13 完全一致，可比）==========
TESTS = [
    ("A100 比 A10 多多少显存？", ["56"]),
    ("ecs.gn7e 用的 GPU 比 ecs.gn7i 用的 GPU 多多少显存？", ["56"]),
    ("包月套餐 PRO 里的 GPU 实例按量付费每小时多少钱？", ["68"]),
    ("李四的上级负责什么 GPU 实例？", ["ecs.gn7", "ecs.gn7e"]),
    ("张三的下属的下属负责的产品包月多少钱？", ["35000"]),
]


def grade(answer, expected):
    return any(k.lower() in answer.lower() for k in expected)


def run_qa(G, label):
    print(f"\n--- {label} 多跳问答 ---")
    rows, ok = [], 0
    for q, exp in TESTS:
        try:
            r = graphrag.graphrag_query(G, q, hops=2)
            correct = grade(r["answer"], exp)
        except Exception as e:
            r = {"answer": f"ERR:{e}", "latency_ms": 0, "seeds": []}
            correct = False
        ok += correct
        rows.append(
            {
                "q": q,
                "ok": correct,
                "answer": r["answer"][:80],
                "seeds": r.get("seeds", []),
            }
        )
        print(f"  [{'✓' if correct else '✗'}] {q[:24]:<24} {r['answer'][:46]}")
    acc = round(ok / len(TESTS), 3)
    print(f"  >>> 准确率 {acc}")
    return {"accuracy": acc, "rows": rows}


def main():
    if not os.getenv("DASHSCOPE_API_KEY"):
        sys.exit("DASHSCOPE_API_KEY 未设置，先 source ~/.zshrc")
    docs = graphrag.KB_DOCS

    print("=== 臂1 自由抽取（#13 原样）建图 ===")
    Gf = graphrag.build_graph(docs, verbose=False)
    mf = measure_graph(Gf)
    print(
        f"  节点 {mf['nodes']} 边 {mf['edges']} 关系类型 {mf['distinct_relation_types']} "
        f"碎片 {mf['entity_fragments']} 非法边 {mf['invalid_relation_edges']}"
    )
    print(f"  关系词表: {mf['relation_types']}")

    print("\n=== 臂2 本体约束抽取建图 ===")
    Go = build_graph_ontology(docs, verbose=False)
    mo = measure_graph(Go)
    print(
        f"  节点 {mo['nodes']} 边 {mo['edges']} 关系类型 {mo['distinct_relation_types']} "
        f"碎片 {mo['entity_fragments']} 非法边 {mo['invalid_relation_edges']}"
    )
    print(f"  关系词表: {mo['relation_types']}")

    qa_free = run_qa(Gf, "臂1 自由抽取")
    qa_onto = run_qa(Go, "臂2 本体约束")

    print("\n========== 对照 ==========")
    print(f"{'指标':<16}{'自由抽取':<12}{'本体约束'}")
    print(
        f"{'关系类型数':<14}{mf['distinct_relation_types']:<12}{mo['distinct_relation_types']}"
    )
    print(f"{'实体碎片数':<14}{mf['entity_fragments']:<12}{mo['entity_fragments']}")
    print(
        f"{'非法关系边':<14}{mf['invalid_relation_edges']:<12}{mo['invalid_relation_edges']}"
    )
    print(f"{'多跳准确率':<14}{qa_free['accuracy']:<12}{qa_onto['accuracy']}")

    out = {
        "free": {
            "graph": mf,
            "qa": qa_free,
            "edges": [(u, v, d["relation"]) for u, v, d in Gf.edges(data=True)],
        },
        "ontology": {
            "graph": mo,
            "qa": qa_onto,
            "edges": [(u, v, d["relation"]) for u, v, d in Go.edges(data=True)],
        },
    }
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果已落盘 {RESULTS}")


if __name__ == "__main__":
    main()
