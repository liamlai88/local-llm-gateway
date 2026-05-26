"""
Semantic Cache v2 — 三层缓存架构（实验 #18B）
1. LLM Normalize 改写 query 成标准形式
2. Embedding 召回 top-K（低阈值，扩大候选）
3. Cross-Encoder Rerank 精排（高阈值，确保准确）
"""

import os
import time
import requests
from typing import Callable, Dict, List, Optional

BAILIAN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
EMBED_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
RERANK_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
)


def _api_key():
    return os.environ["DASHSCOPE_API_KEY"]


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


def rerank(query: str, documents: List[str]) -> List[float]:
    """调百炼 gte-rerank，返回每个 doc 的 relevance_score（0-1）"""
    resp = requests.post(
        RERANK_URL,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gte-rerank",
            "input": {"query": query, "documents": documents},
            "parameters": {"return_documents": False, "top_n": len(documents)},
        },
        timeout=30,
    )
    data = resp.json()
    if "output" not in data:
        raise RuntimeError(f"rerank error: {data}")
    # 按原顺序返回 score
    results = data["output"]["results"]
    scores = [0.0] * len(documents)
    for r in results:
        scores[r["index"]] = r["relevance_score"]
    return scores


# ========== LLM Query Normalization ==========
NORMALIZE_PROMPT = """请把下面用户问题改写成简洁的标准查询形式，去除多余的虚词和口语化表达。

【规则】
- 保留所有关键事实词（产品名、数字、专有名词）
- 去掉"呢"、"啊"、"麻烦"、"请问"、"我想知道"等填充词
- 转成中性陈述形式，不要疑问句的"是多少？"

【示例】
原问题: "PRO 套餐的价格是多少呢"
标准化: "包月套餐 PRO 价格"

原问题: "ecs.gn7e 上面用的是哪款 GPU?"
标准化: "ecs.gn7e GPU 型号"

【现在改写】
原问题: {q}
标准化:"""


def normalize_query(q: str) -> str:
    """调 LLM 改写为标准形式，单行返回"""
    resp = requests.post(
        BAILIAN_URL,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        },
        json={
            "model": "qwen-turbo",
            "messages": [{"role": "user", "content": NORMALIZE_PROMPT.format(q=q)}],
            "max_tokens": 50,
        },
        timeout=30,
    )
    data = resp.json()
    text = data["choices"][0]["message"]["content"].strip()
    # 取第一行非空（防 LLM 输出多余内容）
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith(("【", "标准化")):
            return line
    return q  # fallback：用原始 query


# ========== Cache v2 ==========
class SemanticCacheV2:
    """三层缓存：normalize → embedding 召回 → rerank 精排"""

    def __init__(
        self,
        embed_threshold: float = 0.70,
        rerank_threshold: float = 0.80,
        top_k: int = 5,
    ):
        self.embed_threshold = embed_threshold
        self.rerank_threshold = rerank_threshold
        self.top_k = top_k
        self.entries: List[Dict] = []
        self.stats = {
            "queries": 0,
            "hits": 0,
            "embed_recall_hits": 0,  # embedding 找到 top-k 候选
            "rerank_confirms": 0,  # rerank 确认通过
            "tokens_saved": 0,
            "normalize_tokens": 0,  # 改写消耗的 token
            "llm_calls": 0,
        }

    def lookup(self, q_norm: str, q_emb: List[float]) -> Optional[Dict]:
        """两阶段：embedding 召回 top-K + rerank 精排"""
        if not self.entries:
            return None

        # 阶段 1: embedding 召回 top-K（按 cos sim 排序）
        sims = [(cos_sim(q_emb, e["embedding"]), e) for e in self.entries]
        sims.sort(reverse=True, key=lambda x: x[0])
        candidates = [
            (s, e) for s, e in sims[: self.top_k] if s >= self.embed_threshold
        ]

        if not candidates:
            return None
        self.stats["embed_recall_hits"] += 1

        # 阶段 2: rerank 精排
        # 用 normalized query 和候选的 normalized query 一起做 rerank
        docs = [c[1]["query_norm"] for c in candidates]
        try:
            scores = rerank(q_norm, docs)
        except Exception:
            # rerank 失败，退化为仅 embedding 判断
            best_sim, best_e = candidates[0]
            if best_sim >= 0.85:
                return {"entry": best_e, "rerank_score": None, "embed_sim": best_sim}
            return None

        # 找最高 rerank score
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        if scores[best_idx] >= self.rerank_threshold:
            self.stats["rerank_confirms"] += 1
            return {
                "entry": candidates[best_idx][1],
                "rerank_score": scores[best_idx],
                "embed_sim": candidates[best_idx][0],
            }
        return None

    def get_or_compute(self, query: str, llm_fn: Callable[[str], Dict]) -> Dict:
        t0 = time.time()
        self.stats["queries"] += 1

        # Step 1: Normalize
        q_norm = normalize_query(query)
        self.stats["normalize_tokens"] += 50  # 粗估

        # Step 2: Embed normalized query
        q_emb = embed([q_norm])[0]

        # Step 3: Lookup with rerank
        match = self.lookup(q_norm, q_emb)
        if match:
            self.stats["hits"] += 1
            tokens_saved = match["entry"].get("tokens_used", 0)
            self.stats["tokens_saved"] += tokens_saved
            return {
                "answer": match["entry"]["answer"],
                "from_cache": True,
                "matched_query": match["entry"]["query"],
                "matched_norm": match["entry"]["query_norm"],
                "normalized_query": q_norm,
                "embed_sim": round(match["embed_sim"], 4),
                "rerank_score": round(match["rerank_score"], 4)
                if match["rerank_score"]
                else None,
                "latency_ms": round((time.time() - t0) * 1000, 1),
                "tokens_saved": tokens_saved,
            }

        # Miss
        self.stats["llm_calls"] += 1
        result = llm_fn(query)
        tokens_used = result.get("input_tokens", 0) + result.get("output_tokens", 0)
        self.entries.append(
            {
                "query": query,
                "query_norm": q_norm,
                "embedding": q_emb,
                "answer": result["answer"],
                "tokens_used": tokens_used,
            }
        )
        return {
            "answer": result["answer"],
            "from_cache": False,
            "matched_query": None,
            "matched_norm": None,
            "normalized_query": q_norm,
            "embed_sim": 0,
            "rerank_score": None,
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "tokens_saved": 0,
        }

    def summary(self):
        s = self.stats
        hit_rate = s["hits"] / s["queries"] if s["queries"] else 0
        return {**s, "hit_rate": round(hit_rate, 4), "total_entries": len(self.entries)}
