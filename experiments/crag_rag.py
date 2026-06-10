"""
实验 #23 自纠错 RAG（CRAG, Corrective RAG）

朴素 RAG 的硬伤：召回不好也照样硬塞进去生成 → 幻觉（尤其语料里根本没答案时，会编一个）。
CRAG 加一个「检索质量评估器」把召回分三档，分别纠错：
  CORRECT   → 直接生成
  AMBIGUOUS → 知识精炼（句子级过滤，去噪）→ 生成
  INCORRECT → query 改写 + 重检索；仍差则弃答"信息不足"（拒绝幻觉）

对照 naive RAG，测两类问题：
  ① 可答（语料里有）：CRAG 应≈naive，不能因纠错把对的搞坏
  ② 不可答（语料里没有，H100/MI300）：CRAG 应检测 INCORRECT → 弃答；naive 大概率幻觉

复用 graphrag.call_llm；自带轻量内存向量检索（百炼 embedding + cosine）。
运行：export DASHSCOPE_API_KEY=... ; python experiments/crag_rag.py
"""

import os
import re
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import graphrag  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crag_results.json")

# ========== 语料（#13 的 KB + 几条干扰，但故意不含 H100 / MI300）==========
DOCS = list(graphrag.KB_DOCS) + [
    "数据中心位于新加坡和法兰克福，提供 99.95% SLA。",
    "所有 GPU 实例支持按量付费和包月两种计费方式。",
]


# ========== 轻量向量检索 ==========
def embed(texts):
    api_key = os.environ["DASHSCOPE_API_KEY"]
    import requests

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


_DOC_EMB = None


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb + 1e-9)


def retrieve(query, top_k=3):
    global _DOC_EMB
    if _DOC_EMB is None:
        _DOC_EMB = embed(DOCS)
    qe = embed([query])[0]
    scored = sorted(((_cos(qe, e), d) for e, d in zip(_DOC_EMB, DOCS)), reverse=True)
    return [(s, d) for s, d in scored[:top_k]]


# ========== 检索质量评估器（CRAG 核心）==========
EVAL_PROMPT = """判断下面检索到的资料能否回答问题。只输出一个词：
- CORRECT  : 资料直接包含答案
- AMBIGUOUS: 资料相关但需筛选/不完整
- INCORRECT: 资料与问题无关，不包含答案

【问题】{q}
【检索资料】
{ctx}

只输出 CORRECT / AMBIGUOUS / INCORRECT 之一："""


def evaluate_retrieval(query, docs):
    ctx = "\n".join(f"- {d}" for d in docs)
    out = graphrag.call_llm(EVAL_PROMPT.format(q=query, ctx=ctx)).strip().upper()
    for label in ("INCORRECT", "AMBIGUOUS", "CORRECT"):
        if label in out:
            return label
    return "AMBIGUOUS"


# ========== 纠错动作 ==========
REWRITE_PROMPT = """下面这个问题检索不到好资料，请把它改写成更利于检索的查询（提取核心实体和属性，去掉口语）。
只输出改写后的查询，不要解释。
原问题：{q}"""

REFINE_PROMPT = """从下面资料里，只挑出与问题直接相关的句子，逐条列出；无关的丢弃。
【问题】{q}
【资料】
{ctx}
相关句子："""

ANSWER_PROMPT = """根据资料回答问题。
【资料】
{ctx}
【问题】{q}
【要求】只能基于资料回答；若资料中没有答案，必须明确回答"信息不足，无法回答"，禁止编造。
答案："""

# 无弃答指令版（第三臂对照：证明"廉价 prompt 指令"才是修幻觉的真正杠杆）
ANSWER_PROMPT_NOGUARD = """根据资料回答问题。
【资料】
{ctx}
【问题】{q}
答案："""


def rewrite_query(query):
    return graphrag.call_llm(REWRITE_PROMPT.format(q=query)).strip().strip('"')


def refine(query, docs):
    ctx = "\n".join(f"- {d}" for d in docs)
    return graphrag.call_llm(REFINE_PROMPT.format(q=query, ctx=ctx)).strip()


def generate(query, ctx):
    return graphrag.call_llm(ANSWER_PROMPT.format(ctx=ctx, q=query)).strip()


# ========== 被测系统 ==========
def naive_rag(query, top_k=3):
    start = time.time()
    docs = [d for _, d in retrieve(query, top_k)]
    ans = generate(query, "\n".join(docs))
    return {
        "answer": ans,
        "llm_calls": 1,
        "latency_ms": round((time.time() - start) * 1000, 1),
        "path": "naive",
    }


def naive_rag_noguard(query, top_k=3):
    """无弃答指令：暴露 naive RAG 在不可答问题上的真实幻觉行为。"""
    start = time.time()
    docs = [d for _, d in retrieve(query, top_k)]
    ans = graphrag.call_llm(
        ANSWER_PROMPT_NOGUARD.format(ctx="\n".join(docs), q=query)
    ).strip()
    return {
        "answer": ans,
        "llm_calls": 1,
        "latency_ms": round((time.time() - start) * 1000, 1),
        "path": "naive-noguard",
    }


def crag(query, top_k=3):
    start = time.time()
    calls = 0
    docs = [d for _, d in retrieve(query, top_k)]
    verdict = evaluate_retrieval(query, docs)
    calls += 1

    if verdict == "CORRECT":
        ctx = "\n".join(docs)
        path = "correct→direct"
    elif verdict == "AMBIGUOUS":
        ctx = refine(query, docs)
        calls += 1
        path = "ambiguous→refine"
    else:  # INCORRECT → 改写重检
        rq = rewrite_query(query)
        calls += 1
        docs2 = [d for _, d in retrieve(rq, top_k + 2)]
        v2 = evaluate_retrieval(query, docs2)
        calls += 1
        if v2 == "INCORRECT":
            return {
                "answer": "信息不足，无法回答。",
                "llm_calls": calls,
                "latency_ms": round((time.time() - start) * 1000, 1),
                "path": "incorrect→abstain",
                "verdict": verdict,
            }
        ctx = "\n".join(docs2)
        path = f"incorrect→rewrite({v2})"

    ans = generate(query, ctx)
    calls += 1
    return {
        "answer": ans,
        "llm_calls": calls,
        "latency_ms": round((time.time() - start) * 1000, 1),
        "path": path,
        "verdict": verdict,
    }


# ========== 测试集：可答 + 不可答（语料没有）==========
TESTS = [
    # (问题, 类型, 评分依据)
    {"q": "NVIDIA A100 显存多大？", "type": "answerable", "must_any": ["80"]},
    {"q": "ecs.gn7e 按量付费每小时多少钱？", "type": "answerable", "must_any": ["68"]},
    {
        "q": "包月套餐 PRO 总价多少？",
        "type": "answerable",
        "must_any": ["40000", "4万"],
    },
    {
        "q": "最便宜的 GPU 实例按量付费每小时多少钱？",
        "type": "answerable",
        "must_any": ["18"],
    },
    {"q": "H100 的显存是多少 GB？", "type": "unanswerable", "must_any": []},
    {"q": "你们支持 AMD MI300 加速卡吗？", "type": "unanswerable", "must_any": []},
]

ABSTAIN_MARK = [
    "信息不足",
    "无法回答",
    "没有",
    "未提及",
    "不包含",
    "无相关",
    "不支持",
    "未提供",
]
HALLUC_SPEC = re.compile(r"\d+\s*(GB|G\b|TFLOPS|GiB)", re.IGNORECASE)


def grade(case, answer):
    a = answer or ""
    if case["type"] == "answerable":
        return int(any(k.lower() in a.lower() for k in case["must_any"]))
    # unanswerable：弃答=对(1)；给出具体规格数字=幻觉(0)
    abstained = any(m in a for m in ABSTAIN_MARK)
    hallucinated = bool(HALLUC_SPEC.search(a)) and not abstained
    return int(abstained and not hallucinated)


def run(name, fn):
    print(f"\n=== {name} ===")
    rows = {"answerable": [0, 0], "unanswerable": [0, 0]}  # [correct, total]
    out = []
    for c in TESTS:
        r = fn(c["q"])
        ok = grade(c, r["answer"])
        rows[c["type"]][0] += ok
        rows[c["type"]][1] += 1
        out.append({**c, **r, "correct": ok})
        print(
            f"  [{ok}] ({c['type'][:4]}) {c['q'][:22]:<22} {r.get('path', ''):<20} "
            f"calls={r['llm_calls']} -> {r['answer'][:40]}"
        )
    ans_acc = rows["answerable"][0] / rows["answerable"][1]
    safe = rows["unanswerable"][0] / rows["unanswerable"][1]
    calls = sum(r["llm_calls"] for r in out)
    print(f"  >>> 可答准确率 {ans_acc:.2f} | 不可答安全率 {safe:.2f} | 总调用 {calls}")
    return {
        "answerable_acc": round(ans_acc, 3),
        "unanswerable_safe": round(safe, 3),
        "total_calls": calls,
        "rows": out,
    }


def main():
    if not os.getenv("DASHSCOPE_API_KEY"):
        sys.exit("DASHSCOPE_API_KEY 未设置")
    res = {
        "naive_noguard": run("Naive RAG（无弃答指令）", naive_rag_noguard),
        "naive": run("Naive RAG（+弃答指令）", naive_rag),
        "crag": run("CRAG 自纠错", crag),
    }

    print("\n========== 对照 ==========")
    print(f"{'系统':<20}{'可答准确率':<12}{'不可答安全率':<14}{'总调用'}")
    for k, label in [
        ("naive_noguard", "Naive(无指令)"),
        ("naive", "Naive(+弃答指令)"),
        ("crag", "CRAG"),
    ]:
        s = res[k]
        print(
            f"{label:<12}{s['answerable_acc']:<12}{s['unanswerable_safe']:<14}{s['total_calls']}"
        )

    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n结果已落盘 {RESULTS}")


if __name__ == "__main__":
    main()
