"""
GraphRAG 模块 - 实验 #13
- 内嵌一份产品+组织小 KB（不依赖 chroma）
- 用百炼 qwen-turbo 抽三元组
- 用 networkx 建有向图
- 检索: 从 query 抽实体 → 2-hop 子图 → 拼 context → 生成
"""

import os
import re
import json
import time
import requests
from typing import List, Dict, Tuple

import networkx as nx

BAILIAN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


# ========== 1. Mini KB（精心设计，确保有多跳路径）==========
KB_DOCS = [
    # 产品文档
    "包月套餐 PRO 包含 ecs.gn7e 实例 + 1TB 存储 + 100GB 流量。套餐总价 ¥40000/月。",
    "包月套餐 LITE 包含 ecs.gn7i 实例 + 500GB 存储 + 50GB 流量。套餐总价 ¥12000/月。",
    "ecs.gn7e-c12g1.3xlarge 实例使用 NVIDIA A100 GPU。按量付费 ¥68/小时，包月 ¥35000。",
    "ecs.gn7i-c8g1.2xlarge 实例使用 NVIDIA A10 GPU。按量付费 ¥18/小时，包月 ¥9800。",
    "NVIDIA A100 显存 80GB，FP16 算力 312 TFLOPS。",
    "NVIDIA A10 显存 24GB，FP16 算力 125 TFLOPS。",
    # 组织文档
    "张三是技术 VP，李四是张三的下属。",
    "李四是 GenAI 平台组组长，王五是李四的下属。",
    "王五负责 ecs.gn7e 实例的产品线。",
    "李四亲自负责 ecs.gn7i 实例的产品线。",
]


# ========== 2. LLM 调用工具 ==========
def call_llm(prompt: str, model: str = "qwen-turbo") -> str:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY 未设置")
    resp = requests.post(
        BAILIAN_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 800,
        },
        timeout=60,
    )
    data = resp.json()
    if "choices" not in data:
        raise RuntimeError(f"LLM error: {data}")
    return data["choices"][0]["message"]["content"]


# ========== 3. 三元组抽取 ==========
EXTRACT_PROMPT = """从下面文档中抽取所有 (主语, 关系, 宾语) 三元组。

【规则】
- 实体用文档中的原词，不要泛化（如 "NVIDIA A100" 不要写成 "A100 GPU"）
- 关系用动词或介词，简短（包含/使用/下属/负责/显存/价格/算力 等）
- 数字+单位作为整体宾语（如 "80GB"、"¥35000"、"¥68/小时"）
- 一条文档可能产生多个三元组

【输出格式 - 严格 JSON】
{"triples": [["主语", "关系", "宾语"], ...]}

【示例】
文档: "包月套餐 PRO 包含 ecs.gn7e 实例。套餐总价 ¥40000/月。"
输出: {"triples": [["包月套餐 PRO", "包含", "ecs.gn7e 实例"], ["包月套餐 PRO", "总价", "¥40000/月"]]}

文档: """


def extract_triples(doc: str) -> List[Tuple[str, str, str]]:
    output = call_llm(EXTRACT_PROMPT + doc)
    try:
        m = re.search(r"\{.*\}", output, re.DOTALL)
        data = json.loads(m.group(0))
        return [tuple(t) for t in data["triples"] if len(t) == 3]
    except Exception as e:
        print(f"[WARN] 抽取失败: {e}\n输出: {output[:200]}")
        return []


# ========== 4. 建图 ==========
def build_graph(docs: List[str], verbose: bool = True) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for i, doc in enumerate(docs):
        if verbose:
            print(f"[{i + 1}/{len(docs)}] 抽取: {doc[:50]}...")
        triples = extract_triples(doc)
        for s, r, o in triples:
            G.add_edge(s, o, relation=r, source_doc=i)
            if verbose:
                print(f"    ({s}) --[{r}]--> ({o})")
    return G


# ========== 5. Query 实体抽取 ==========
QUERY_ENTITY_PROMPT = """从下面问题中抽取出现的实体。实体可能是人名、产品名、规格名等。

【可用实体表】（只能从这里选，命中哪个就返回哪个，可多个）
{entities}

【输出格式 - 严格 JSON】
{{"entities": ["实体1", "实体2"]}}

问题: {question}"""


def extract_query_entities(question: str, all_entities: List[str]) -> List[str]:
    prompt = QUERY_ENTITY_PROMPT.format(
        entities="\n".join(f"- {e}" for e in all_entities),
        question=question,
    )
    output = call_llm(prompt)
    try:
        m = re.search(r"\{.*\}", output, re.DOTALL)
        data = json.loads(m.group(0))
        return [e for e in data["entities"] if e in all_entities]
    except Exception:
        # fallback: 字符串匹配
        return [e for e in all_entities if e in question]


# ========== 6. 子图检索 ==========
def get_subgraph_context(
    G: nx.MultiDiGraph, seed_entities: List[str], hops: int = 2
) -> str:
    """从 seed 出发做 N-hop 扩展（无向遍历），返回所有相关三元组的文本"""
    if not seed_entities:
        return ""
    visited = set(seed_entities)
    frontier = set(seed_entities)
    for _ in range(hops):
        new_frontier = set()
        for node in frontier:
            if node not in G:
                continue
            # 出边
            for _, neighbor, _ in G.out_edges(node, data=True):
                if neighbor not in visited:
                    new_frontier.add(neighbor)
            # 入边
            for pred, _, _ in G.in_edges(node, data=True):
                if pred not in visited:
                    new_frontier.add(pred)
        visited |= new_frontier
        frontier = new_frontier

    # 收集所有内部边
    lines = []
    for u, v, data in G.edges(data=True):
        if u in visited and v in visited:
            lines.append(f"({u}) --[{data['relation']}]--> ({v})")
    return "\n".join(lines)


# ========== 7. 生成答案 ==========
ANSWER_PROMPT = """根据下面的知识图谱事实回答问题。

【知识图谱事实】
{context}

【问题】
{question}

【要求】
- 只能基于上述事实推理，不要凭记忆
- 涉及数字计算请明确写出算式
- 答案简洁，包含关键数字"""


def graphrag_query(G: nx.MultiDiGraph, question: str, hops: int = 2) -> Dict:
    start = time.time()
    all_entities = list(G.nodes())

    # Step 1: query → entities
    seeds = extract_query_entities(question, all_entities)

    # Step 2: subgraph
    context = get_subgraph_context(G, seeds, hops=hops)

    # Step 3: generate
    answer = call_llm(ANSWER_PROMPT.format(context=context, question=question))

    return {
        "answer": answer.strip(),
        "seeds": seeds,
        "context": context,
        "latency_ms": round((time.time() - start) * 1000, 1),
    }


# ========== 8. Baseline: 朴素 RAG（向量相似 top-k）==========
def naive_rag_query(question: str, docs: List[str], top_k: int = 3) -> Dict:
    """复用百炼 embedding，跳过 chroma 简化对照"""
    start = time.time()
    api_key = os.environ.get("DASHSCOPE_API_KEY")

    # 批量 embed
    def embed(texts: List[str]) -> List[List[float]]:
        resp = requests.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"model": "text-embedding-v2", "input": texts},
            timeout=30,
        )
        return [d["embedding"] for d in resp.json()["data"]]

    doc_embeds = embed(docs)
    q_embed = embed([question])[0]

    # cosine
    def cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb + 1e-9)

    scored = sorted(
        [(cos(q_embed, e), d) for e, d in zip(doc_embeds, docs)], reverse=True
    )
    top_docs = [d for _, d in scored[:top_k]]
    context = "\n".join(top_docs)

    answer = call_llm(ANSWER_PROMPT.format(context=context, question=question))
    return {
        "answer": answer.strip(),
        "context": context,
        "latency_ms": round((time.time() - start) * 1000, 1),
    }
