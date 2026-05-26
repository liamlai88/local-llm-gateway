"""
A/B Testing + 在线评测 + 自动回滚 — 实验 #19
- 多 prompt 变体管理（按 user_id hash 分流）
- LLM-as-judge 在线打分（Faithfulness / Relevance）
- 统计显著性测试（Welch's t-test）
- 自动决策（promote / rollback / 继续观察）
"""

import os
import time
import json
import math
import hashlib
import requests
from typing import Dict, List, Optional

BAILIAN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


def _api_key():
    return os.environ["DASHSCOPE_API_KEY"]


def call_llm(messages, max_tokens=400, model="qwen-turbo"):
    resp = requests.post(
        BAILIAN_URL,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        },
        json={"model": model, "messages": messages, "max_tokens": max_tokens},
        timeout=60,
    )
    data = resp.json()
    usage = data.get("usage", {})
    return {
        "answer": data["choices"][0]["message"]["content"],
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }


# ========== Traffic Splitter ==========
class TrafficSplitter:
    """按 user_id 稳定哈希分流，避免同用户多次请求被分到不同 variant"""

    def __init__(self, variants: Dict[str, float]):
        """variants: {variant_name: traffic_ratio}，ratio 总和 = 1.0"""
        total = sum(variants.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"variant ratios 总和 = {total}, 应该 = 1.0")
        # 累积区间: {variant: (lo, hi)}
        self.intervals = {}
        cum = 0.0
        for name, ratio in variants.items():
            self.intervals[name] = (cum, cum + ratio)
            cum += ratio

    def assign(self, user_id: str) -> str:
        # MD5 hash → [0, 1) 区间
        h = int(hashlib.md5(user_id.encode()).hexdigest()[:8], 16)
        bucket = (h % 10000) / 10000.0
        for name, (lo, hi) in self.intervals.items():
            if lo <= bucket < hi:
                return name
        # fallback
        return list(self.intervals.keys())[-1]


# ========== 在线评分 (LLM-as-judge) ==========
def online_judge(query: str, answer: str, context: Optional[str] = None) -> Dict:
    """
    返回 {faithfulness: 0-1, relevance: 0-1}
    一次 LLM 调用同时打两个分（省钱）
    """
    ctx_block = f"\n【可用上下文】\n{context}\n" if context else ""
    prompt = f"""你是 AI 回答质量审查员。给下面的 AI 回答打两个分（0-1，1=最好）。

【用户问题】
{query}
{ctx_block}
【AI 回答】
{answer}

【两个指标】
1. faithfulness（忠诚度）: 答案是否基于上下文，没有编造。无上下文则看答案是否合理且不矛盾
2. relevance（相关性）: 答案是否切题，回答了用户的问题

【输出格式 - 严格 JSON】
{{"faithfulness": 0.x, "relevance": 0.x}}

只输出 JSON，分数保留 2 位小数。"""

    try:
        r = call_llm([{"role": "user", "content": prompt}], max_tokens=80)
        import re

        m = re.search(r"\{.*\}", r["answer"], re.DOTALL)
        d = json.loads(m.group(0))
        return {
            "faithfulness": float(d.get("faithfulness", 0)),
            "relevance": float(d.get("relevance", 0)),
        }
    except Exception:
        return {"faithfulness": None, "relevance": None}


# ========== 统计：Welch's t-test ==========
def welch_t_test(a: List[float], b: List[float]) -> Dict:
    """
    返回 {t_stat, df, p_value (单侧 approximate), significant}
    单侧检验: H1: mean(b) > mean(a)
    """
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if len(a) < 2 or len(b) < 2:
        return {"t": None, "p": None, "significant": False, "note": "样本太少"}

    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return {"t": 0, "p": 0.5, "significant": False, "mean_a": ma, "mean_b": mb}
    t = (mb - ma) / se
    # 自由度 (Welch-Satterthwaite)
    df_num = (va / len(a) + vb / len(b)) ** 2
    df_den = (va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1)
    df = df_num / df_den if df_den > 0 else 1

    # 简化 p-value 估算 (用正态近似，当 df > 30 较准)
    # p = 1 - CDF(t) 单侧
    p = 0.5 * (1 - math.erf(abs(t) / math.sqrt(2)))
    if t < 0:
        p = 1 - p

    return {
        "t": round(t, 3),
        "df": round(df, 1),
        "p": round(p, 4),
        "significant": p < 0.05,
        "mean_a": round(ma, 4),
        "mean_b": round(mb, 4),
        "delta": round(mb - ma, 4),
    }


# ========== A/B Tester ==========
class ABTester:
    def __init__(self, variants: Dict[str, str], traffic: Dict[str, float]):
        """
        variants: {name: system_prompt}
        traffic: {name: ratio}
        """
        self.variants = variants
        self.splitter = TrafficSplitter(traffic)
        self.records = {name: [] for name in variants}

    def call(self, user_id: str, query: str, context: Optional[str] = None) -> Dict:
        variant = self.splitter.assign(user_id)
        sys = self.variants[variant]
        msgs = [{"role": "system", "content": sys}]
        if context:
            msgs.append({"role": "system", "content": f"知识库：{context}"})
        msgs.append({"role": "user", "content": query})

        t0 = time.time()
        r = call_llm(msgs)
        latency = (time.time() - t0) * 1000

        # 在线评分（同步，生产可异步）
        scores = online_judge(query, r["answer"], context=context)

        record = {
            "user_id": user_id,
            "variant": variant,
            "query": query,
            "answer": r["answer"],
            "latency_ms": round(latency, 1),
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "faithfulness": scores["faithfulness"],
            "relevance": scores["relevance"],
            "length": len(r["answer"]),
        }
        self.records[variant].append(record)
        return record

    def stats(self) -> Dict:
        """返回每个 variant 的均值汇总 + A vs B 的显著性检验"""
        summary = {}
        for v, rs in self.records.items():
            if not rs:
                continue

            def m(k):
                vals = [r[k] for r in rs if r[k] is not None]
                return sum(vals) / len(vals) if vals else None

            summary[v] = {
                "n": len(rs),
                "faithfulness": m("faithfulness"),
                "relevance": m("relevance"),
                "latency_ms": m("latency_ms"),
                "length": m("length"),
                "avg_input_tokens": m("input_tokens"),
                "avg_output_tokens": m("output_tokens"),
            }

        # A vs B 显著性
        if len(self.records) == 2:
            names = list(self.records.keys())
            a_recs = self.records[names[0]]
            b_recs = self.records[names[1]]
            tests = {}
            for metric in ["faithfulness", "relevance", "length", "latency_ms"]:
                a_vals = [r[metric] for r in a_recs]
                b_vals = [r[metric] for r in b_recs]
                tests[metric] = welch_t_test(a_vals, b_vals)
            summary["_tests"] = tests
            summary["_compared"] = names

        return summary

    def decision(self, rollback_threshold: float = 0.15) -> Dict:
        """
        自动决策:
        - rollback: B 在任一关键指标比 A 低 >threshold 且显著
        - promote: B 在 faithfulness 或 relevance 比 A 高且显著
        - hold: 继续观察
        """
        s = self.stats()
        if "_tests" not in s:
            return {"decision": "hold", "reason": "未做 A/B 对比"}

        names = s["_compared"]
        a_name, b_name = names[0], names[1]
        tests = s["_tests"]

        # 检查回滚条件
        for key in ["faithfulness", "relevance"]:
            t = tests[key]
            if (
                t.get("delta") is not None
                and t["delta"] < -rollback_threshold
                and t["significant"]
            ):
                return {
                    "decision": "rollback",
                    "reason": f"{b_name} 的 {key} 比 {a_name} 低 {abs(t['delta']):.2f} 且显著 (p={t['p']})",
                    "rollback_to": a_name,
                }

        # 检查升级条件
        improvements = []
        for key in ["faithfulness", "relevance"]:
            t = tests[key]
            if t.get("delta") is not None and t["delta"] > 0 and t["significant"]:
                improvements.append(f"{key} +{t['delta']:.2f}")

        if improvements:
            return {
                "decision": "promote",
                "reason": f"{b_name} 显著优于 {a_name}: {', '.join(improvements)}",
                "promote": b_name,
            }

        return {"decision": "hold", "reason": "无统计显著差异，继续观察"}
