"""
实验 #24 CRAG 的 break-even 点：弱模型上自纠错 RAG 会不会翻盘

#23 结论：强模型(qwen-turbo)下 CRAG 冗余——base model 自己就不幻觉。
#24 换弱生成器(本地 qwen2.5-1.5b)，测 CRAG 的价值是否随 base model 变弱而出现。

四臂（生成器都用弱模型 1.5B）：
  A Naive-弱(无弃答指令)      → 暴露弱模型真实幻觉
  B Naive-弱(+弃答指令)       → 廉价 prompt 杠杆能否救弱模型
  C CRAG(弱评估器, 全1.5B)    → 评估器也弱时还有用吗
  D CRAG(强评估器, 1.5B生成+turbo评估/改写) → 强评估器纠偏弱生成器

复用 #23 的语料/检索/测试集/评分；LLM 调用走 agent.call_llm（支持 local/bailian）。
运行：ollama serve & ; export DASHSCOPE_API_KEY=... ; python experiments/crag_weak.py
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agent  # noqa: E402
import crag_rag as cr  # noqa: E402  复用 DOCS/retrieve/prompts/TESTS/grade

RESULTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "crag_weak_results.json"
)

WEAK = ("qwen2.5-1.5b", "local")
STRONG = ("qwen-turbo", "bailian")


def llm(prompt, prov):
    model, provider = prov
    return agent.call_llm(
        [{"role": "user", "content": prompt}], model=model, provider=provider
    )


# ========== Naive（弱生成器）==========
def naive(query, gen, guard=True, top_k=3):
    start = time.time()
    docs = [d for _, d in cr.retrieve(query, top_k)]
    tmpl = cr.ANSWER_PROMPT if guard else cr.ANSWER_PROMPT_NOGUARD
    ans = llm(tmpl.format(ctx="\n".join(docs), q=query), gen).strip()
    return {
        "answer": ans,
        "llm_calls": 1,
        "latency_ms": round((time.time() - start) * 1000, 1),
        "path": "naive" + ("" if guard else "-noguard"),
    }


# ========== CRAG（生成器/评估器可分别配置）==========
def crag(query, gen, ev, top_k=3):
    start = time.time()
    calls = 0
    docs = [d for _, d in cr.retrieve(query, top_k)]
    verdict = _eval(query, docs, ev)
    calls += 1

    if verdict == "CORRECT":
        ctx = "\n".join(docs)
        path = "correct→direct"
    elif verdict == "AMBIGUOUS":
        ctx = llm(
            cr.REFINE_PROMPT.format(q=query, ctx="\n".join(f"- {d}" for d in docs)), ev
        ).strip()
        calls += 1
        path = "ambiguous→refine"
    else:
        rq = llm(cr.REWRITE_PROMPT.format(q=query), ev).strip().strip('"')
        calls += 1
        docs2 = [d for _, d in cr.retrieve(rq, top_k + 2)]
        v2 = _eval(query, docs2, ev)
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

    ans = llm(
        cr.ANSWER_PROMPT.format(ctx=ctx, q=query), gen
    ).strip()  # 生成始终用弱模型
    calls += 1
    return {
        "answer": ans,
        "llm_calls": calls,
        "latency_ms": round((time.time() - start) * 1000, 1),
        "path": path,
        "verdict": verdict,
    }


def _eval(query, docs, ev):
    out = (
        llm(cr.EVAL_PROMPT.format(q=query, ctx="\n".join(f"- {d}" for d in docs)), ev)
        .strip()
        .upper()
    )
    for label in ("INCORRECT", "AMBIGUOUS", "CORRECT"):
        if label in out:
            return label
    return "AMBIGUOUS"


def run(name, fn):
    print(f"\n=== {name} ===")
    rows = {"answerable": [0, 0], "unanswerable": [0, 0]}
    out = []
    for c in cr.TESTS:
        r = fn(c["q"])
        ok = cr.grade(c, r["answer"])
        rows[c["type"]][0] += ok
        rows[c["type"]][1] += 1
        out.append({**c, **r, "correct": ok})
        print(
            f"  [{ok}] ({c['type'][:4]}) {c['q'][:20]:<20} {r.get('path', ''):<22} "
            f"-> {r['answer'][:38]}"
        )
    ans_acc = rows["answerable"][0] / rows["answerable"][1]
    safe = rows["unanswerable"][0] / rows["unanswerable"][1]
    calls = sum(r["llm_calls"] for r in out)
    print(f"  >>> 可答 {ans_acc:.2f} | 不可答安全 {safe:.2f} | 调用 {calls}")
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
        "A_naive_weak_noguard": run(
            "A Naive-弱(无弃答指令)", lambda q: naive(q, WEAK, guard=False)
        ),
        "B_naive_weak_guard": run(
            "B Naive-弱(+弃答指令)", lambda q: naive(q, WEAK, guard=True)
        ),
        "C_crag_weak_eval": run(
            "C CRAG(弱评估器 全1.5B)", lambda q: crag(q, WEAK, WEAK)
        ),
        "D_crag_strong_eval": run(
            "D CRAG(强评估器 1.5B生成+turbo评估)", lambda q: crag(q, WEAK, STRONG)
        ),
    }

    print("\n========== 四臂对照（生成器均为弱模型 1.5B）==========")
    print(f"{'臂':<26}{'可答准确率':<12}{'不可答安全率':<14}{'调用'}")
    for k, label in [
        ("A_naive_weak_noguard", "A Naive无指令"),
        ("B_naive_weak_guard", "B Naive+弃答指令"),
        ("C_crag_weak_eval", "C CRAG弱评估器"),
        ("D_crag_strong_eval", "D CRAG强评估器"),
    ]:
        s = res[k]
        print(
            f"{label:<24}{s['answerable_acc']:<12}{s['unanswerable_safe']:<14}{s['total_calls']}"
        )

    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n结果已落盘 {RESULTS}")


if __name__ == "__main__":
    main()
