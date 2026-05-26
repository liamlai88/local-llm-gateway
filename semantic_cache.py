"""
Semantic Cache - 实验 #18
用 embedding cosine 相似度做查询缓存层

接口:
  cache = SemanticCache(threshold=0.95)
  result = cache.get_or_compute(query, llm_fn)
  # result = {"answer", "from_cache", "matched_query", "similarity", "latency_ms", "tokens_saved"}
"""

import os
import time
import requests
from typing import Callable, Dict, List, Optional

BAILIAN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
EMBED_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"


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


class SemanticCache:
    """简单的内存语义缓存。生产环境换 Redis + 向量索引（如 redis-stack）"""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.entries: List[Dict] = []  # [{query, embedding, answer, tokens_used}]
        self.stats = {
            "queries": 0,
            "hits": 0,
            "tokens_saved": 0,
            "embed_calls": 0,
            "llm_calls": 0,
        }

    def lookup(self, query: str, query_emb: Optional[List[float]] = None):
        """返回 (matched_entry, similarity) 或 (None, best_sim)"""
        if not self.entries:
            return None, 0.0
        if query_emb is None:
            query_emb = embed([query])[0]
            self.stats["embed_calls"] += 1

        best = None
        best_sim = 0.0
        for e in self.entries:
            s = cos_sim(query_emb, e["embedding"])
            if s > best_sim:
                best_sim = s
                best = e
        if best_sim >= self.threshold:
            return best, best_sim
        return None, best_sim

    def get_or_compute(self, query: str, llm_fn: Callable[[str], Dict]) -> Dict:
        """
        llm_fn(query) -> {"answer": str, "input_tokens": int, "output_tokens": int}
        返回 {answer, from_cache, matched_query, similarity, latency_ms, tokens_saved}
        """
        t0 = time.time()
        self.stats["queries"] += 1

        # 算 query embedding（每次都要算 embedding，无论命不命中）
        query_emb = embed([query])[0]
        self.stats["embed_calls"] += 1

        matched, sim = self.lookup(query, query_emb)
        if matched is not None:
            # 命中
            self.stats["hits"] += 1
            tokens_saved = matched.get("tokens_used", 0)
            self.stats["tokens_saved"] += tokens_saved
            return {
                "answer": matched["answer"],
                "from_cache": True,
                "matched_query": matched["query"],
                "similarity": round(sim, 4),
                "latency_ms": round((time.time() - t0) * 1000, 1),
                "tokens_saved": tokens_saved,
            }

        # Miss：调 LLM
        self.stats["llm_calls"] += 1
        result = llm_fn(query)
        tokens_used = result.get("input_tokens", 0) + result.get("output_tokens", 0)

        # 入缓存
        self.entries.append(
            {
                "query": query,
                "embedding": query_emb,
                "answer": result["answer"],
                "tokens_used": tokens_used,
            }
        )

        return {
            "answer": result["answer"],
            "from_cache": False,
            "matched_query": None,
            "similarity": round(sim, 4),  # 最高相似度但低于阈值
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "tokens_saved": 0,
        }

    def summary(self):
        s = self.stats
        hit_rate = s["hits"] / s["queries"] if s["queries"] else 0
        return {
            **s,
            "hit_rate": round(hit_rate, 4),
            "total_entries": len(self.entries),
        }


# ========== LLM 调用辅助 ==========
def call_llm_with_tokens(
    query: str, system_prompt: str = "你是销售客服，简洁回答用户的问题"
) -> Dict:
    resp = requests.post(
        BAILIAN_URL,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        },
        json={
            "model": "qwen-turbo",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            "max_tokens": 300,
        },
        timeout=60,
    )
    data = resp.json()
    usage = data.get("usage", {})
    return {
        "answer": data["choices"][0]["message"]["content"],
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }
