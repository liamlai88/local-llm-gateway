"""
RAGAS-Lite: 4 个核心 RAG 评测指标的自实现
- Faithfulness: 答案是否基于 context（反幻觉）
- Answer Relevance: 答案是否切题
- Context Precision: 召回的 context 噪声率
- Context Recall: 该召回的内容是否都召回了（需要 ground truth）

依赖百炼 qwen-turbo 做 LLM judge + text-embedding-v2 算相似度。
"""

import os
import re
import json
import requests
from typing import List, Dict, Optional

BAILIAN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
EMBED_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"


def _api_key() -> str:
    k = os.environ.get("DASHSCOPE_API_KEY")
    if not k:
        raise RuntimeError("DASHSCOPE_API_KEY 未设置")
    return k


def call_llm(prompt: str, model: str = "qwen-turbo", max_tokens: int = 500) -> str:
    resp = requests.post(
        BAILIAN_URL,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        },
        timeout=60,
    )
    data = resp.json()
    if "choices" not in data:
        raise RuntimeError(f"LLM error: {data}")
    return data["choices"][0]["message"]["content"]


def embed(texts: List[str]) -> List[List[float]]:
    resp = requests.post(
        EMBED_URL,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        },
        json={"model": "text-embedding-v2", "input": texts},
        timeout=30,
    )
    return [d["embedding"] for d in resp.json()["data"]]


def cos_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb + 1e-9)


def _extract_json(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in: {text[:200]}")
    return json.loads(m.group(0))


# ========== 1. Faithfulness（忠诚度）==========
# 步骤: 答案 → 分原子声明 → 每个声明判断是否被 context 支持
def faithfulness(answer: str, context: str) -> Dict:
    # Step 1: 拆分原子声明
    split_prompt = f"""把下面答案拆分成原子声明列表（每条只表达一个事实）。

答案: {answer}

【输出格式 - 严格 JSON】
{{"claims": ["声明1", "声明2", ...]}}

只输出 JSON。"""
    try:
        claims = _extract_json(call_llm(split_prompt, max_tokens=400))["claims"]
    except Exception as e:
        return {"score": 0.0, "claims": [], "supported": [], "error": str(e)}

    if not claims:
        return {"score": 1.0, "claims": [], "supported": []}

    # Step 2: 每个声明判断是否被 context 支持
    judge_prompt = f"""判断下面每条声明是否被给定的 context 直接支持。

Context:
{context}

声明列表:
{json.dumps(claims, ensure_ascii=False, indent=2)}

【判断规则】
- "supported": context 中有明确事实支持该声明
- "not_supported": context 中没有，或与 context 矛盾

【输出格式 - 严格 JSON】
{{"verdicts": ["supported" 或 "not_supported", ...]}}
（数组长度必须等于声明数）

只输出 JSON。"""
    try:
        verdicts = _extract_json(call_llm(judge_prompt, max_tokens=400))["verdicts"]
    except Exception as e:
        return {"score": 0.0, "claims": claims, "supported": [], "error": str(e)}

    supported = [v == "supported" for v in verdicts[: len(claims)]]
    score = sum(supported) / len(claims) if claims else 1.0
    return {"score": round(score, 3), "claims": claims, "supported": supported}


# ========== 2. Answer Relevance（答案相关性）==========
# 步骤: 答案 → 反推问题 × 3 → 与原问题 embedding 相似度均值
def answer_relevance(question: str, answer: str, n_generated: int = 3) -> Dict:
    gen_prompt = f"""根据下面答案，反推 {n_generated} 个最可能问到的问题（用户可能问什么导致这个答案）。

答案: {answer}

【输出格式 - 严格 JSON】
{{"questions": ["问题1", "问题2", "问题3"]}}

只输出 JSON。"""
    try:
        gen_qs = _extract_json(call_llm(gen_prompt, max_tokens=300))["questions"]
    except Exception as e:
        return {"score": 0.0, "generated": [], "error": str(e)}

    if not gen_qs:
        return {"score": 0.0, "generated": []}

    embeds = embed([question] + gen_qs)
    sims = [cos_sim(embeds[0], e) for e in embeds[1:]]
    return {
        "score": round(sum(sims) / len(sims), 3),
        "generated": gen_qs,
        "sims": [round(s, 3) for s in sims],
    }


# ========== 3. Context Precision（上下文精度）==========
# 步骤: 把 context 按"片段"切（多文档时是天然单元；单文档按段落切）
#       对每个片段判断是否对回答有帮助
def context_precision(question: str, context_chunks: List[str]) -> Dict:
    if not context_chunks:
        return {"score": 0.0, "useful": []}

    judge_prompt = f"""判断下面每个 context 片段是否对回答问题"{question}"有用。

片段列表:
{json.dumps(context_chunks, ensure_ascii=False, indent=2)}

【判断规则】
- "useful": 片段包含回答问题所需的信息
- "useless": 片段与问题无关或不提供有用信息

【输出格式 - 严格 JSON】
{{"verdicts": ["useful" 或 "useless", ...]}}
（长度必须等于片段数）

只输出 JSON。"""
    try:
        verdicts = _extract_json(call_llm(judge_prompt, max_tokens=300))["verdicts"]
    except Exception as e:
        return {"score": 0.0, "useful": [], "error": str(e)}

    useful = [v == "useful" for v in verdicts[: len(context_chunks)]]
    # 简单平均（标准 RAGAS 用 rank-weighted，这里简化）
    score = sum(useful) / len(useful) if useful else 0.0
    return {"score": round(score, 3), "useful": useful, "n_chunks": len(context_chunks)}


# ========== 4. Context Recall（上下文召回率）==========
# 步骤: ground_truth → 分原子声明 → 每个声明判断是否被 context 支持
def context_recall(ground_truth: str, context: str) -> Dict:
    split_prompt = f"""把下面参考答案拆分成原子声明。

参考答案: {ground_truth}

【输出格式 - 严格 JSON】
{{"claims": ["声明1", ...]}}

只输出 JSON。"""
    try:
        claims = _extract_json(call_llm(split_prompt, max_tokens=300))["claims"]
    except Exception as e:
        return {"score": 0.0, "error": str(e)}

    if not claims:
        return {"score": 1.0, "claims": []}

    judge_prompt = f"""判断下面每条参考答案声明是否能在 context 中找到支持。

Context:
{context}

声明:
{json.dumps(claims, ensure_ascii=False, indent=2)}

【输出格式 - 严格 JSON】
{{"verdicts": ["found" 或 "not_found", ...]}}

只输出 JSON。"""
    try:
        verdicts = _extract_json(call_llm(judge_prompt, max_tokens=300))["verdicts"]
    except Exception as e:
        return {"score": 0.0, "claims": claims, "error": str(e)}

    found = [v == "found" for v in verdicts[: len(claims)]]
    score = sum(found) / len(claims)
    return {"score": round(score, 3), "claims": claims, "found": found}


# ========== 综合评测入口 ==========
def evaluate(
    question: str,
    answer: str,
    context: str,
    context_chunks: Optional[List[str]] = None,
    ground_truth: Optional[str] = None,
) -> Dict:
    """对一个样本跑全部 4 个指标"""
    result = {
        "question": question,
        "answer": answer[:200],
        "faithfulness": faithfulness(answer, context),
        "answer_relevance": answer_relevance(question, answer),
    }
    if context_chunks:
        result["context_precision"] = context_precision(question, context_chunks)
    if ground_truth:
        result["context_recall"] = context_recall(ground_truth, context)
    return result
