"""
多 Provider 模型价格表 + 成本计算
价格数据采集时间: 2026-05
单位:
  - 百炼: 人民币 ¥ / 1K tokens
  - OpenAI / Anthropic: 美元 $ / 1M tokens（内部转换成 ¥/1K，按 ¥7.2/$）
所有对外接口统一返回 ¥/1K tokens
"""

USD_TO_CNY = 7.2


# ========== 价格表（input / output per 1K tokens, 单位 ¥）==========
PRICING = {
    # --- 阿里云百炼 ---
    "qwen-turbo": {
        "input": 0.0003,
        "output": 0.0006,
        "currency": "CNY",
        "provider": "bailian",
    },
    "qwen-plus": {
        "input": 0.0008,
        "output": 0.002,
        "currency": "CNY",
        "provider": "bailian",
    },
    "qwen-max": {
        "input": 0.02,
        "output": 0.06,
        "currency": "CNY",
        "provider": "bailian",
    },
    "qwen-long": {
        "input": 0.0005,
        "output": 0.002,
        "currency": "CNY",
        "provider": "bailian",
    },
    "text-embedding-v2": {
        "input": 0.0007,
        "output": 0.0,
        "currency": "CNY",
        "provider": "bailian",
    },
    "gte-rerank": {
        "input": 0.0008,
        "output": 0.0,
        "currency": "CNY",
        "provider": "bailian",
        "unit": "per_query",
    },
    # --- OpenAI ---
    "gpt-4o": {
        "input": 2.50 * USD_TO_CNY / 1000,
        "output": 10.0 * USD_TO_CNY / 1000,
        "currency": "CNY",
        "provider": "openai",
    },
    "gpt-4o-mini": {
        "input": 0.15 * USD_TO_CNY / 1000,
        "output": 0.60 * USD_TO_CNY / 1000,
        "currency": "CNY",
        "provider": "openai",
    },
    # --- Anthropic ---
    "claude-sonnet-4": {
        "input": 3.0 * USD_TO_CNY / 1000,
        "output": 15.0 * USD_TO_CNY / 1000,
        "currency": "CNY",
        "provider": "anthropic",
    },
    "claude-haiku-4": {
        "input": 0.80 * USD_TO_CNY / 1000,
        "output": 4.0 * USD_TO_CNY / 1000,
        "currency": "CNY",
        "provider": "anthropic",
    },
    # --- 本地（电费摊销估算，M5 24GB 跑 Qwen2.5-7B Q4，~20W 推理功率）---
    # 假设 1000 token/s 输出速度, ¥0.6/度电, 24h 推理 ≈ 0.48 度 ≈ ¥0.29
    # 平均到 token: ¥0.29 / (1000 * 86400) = ~¥3.4e-9 / token ≈ 0.0000034 / 1K
    # 实际硬件折旧 + 电费综合算 ¥0.00001/1K（仍然便宜 30 倍）
    "qwen2.5-1.5b-local": {
        "input": 0.00001,
        "output": 0.00001,
        "currency": "CNY",
        "provider": "local",
    },
    "qwen2.5-7b-local": {
        "input": 0.00005,
        "output": 0.00005,
        "currency": "CNY",
        "provider": "local",
    },
}


def calc_cost(
    model: str, input_tokens: int, output_tokens: int = 0, n_queries: int = 1
) -> float:
    """返回总成本 (¥)"""
    if model not in PRICING:
        raise ValueError(f"未知模型: {model}. 可用: {list(PRICING.keys())}")
    p = PRICING[model]
    if p.get("unit") == "per_query":
        return n_queries * p["input"]
    cost = (input_tokens / 1000) * p["input"] + (output_tokens / 1000) * p["output"]
    return cost


def estimate_scenario(
    model: str,
    avg_input_tokens: int,
    avg_output_tokens: int,
    queries_per_day: int,
    days: int = 30,
) -> dict:
    """估算场景总成本"""
    per_call = calc_cost(model, avg_input_tokens, avg_output_tokens)
    total_calls = queries_per_day * days
    total = per_call * total_calls
    return {
        "model": model,
        "per_call_cny": round(per_call, 6),
        "queries_per_day": queries_per_day,
        "days": days,
        "total_calls": total_calls,
        "total_cny": round(total, 2),
        "monthly_cny": round(total / days * 30, 2),
        "yearly_cny": round(total / days * 365, 2),
    }


def compare_models(
    models: list,
    avg_input_tokens: int,
    avg_output_tokens: int,
    queries_per_day: int,
    days: int = 30,
) -> list:
    """对比多个模型在同一场景下的成本"""
    rows = []
    for m in models:
        r = estimate_scenario(
            m, avg_input_tokens, avg_output_tokens, queries_per_day, days
        )
        rows.append(r)
    rows.sort(key=lambda x: x["total_cny"])
    return rows


def format_table(rows: list, title: str = "") -> str:
    """打印对比表"""
    out = []
    if title:
        out.append(f"\n=== {title} ===")
    headers = ["model", "per_call", "monthly", "yearly"]
    out.append(f"{'model':<22}{'per_call ¥':<14}{'monthly ¥':<14}{'yearly ¥':<14}")
    out.append("-" * 64)
    for r in rows:
        out.append(
            f"{r['model']:<22}{r['per_call_cny']:<14.6f}{r['monthly_cny']:<14.2f}{r['yearly_cny']:<14.2f}"
        )
    return "\n".join(out)


if __name__ == "__main__":
    # 演示几个常见场景
    print(
        format_table(
            compare_models(
                [
                    "qwen-turbo",
                    "qwen-plus",
                    "qwen-max",
                    "gpt-4o",
                    "gpt-4o-mini",
                    "claude-sonnet-4",
                    "qwen2.5-7b-local",
                ],
                avg_input_tokens=500,
                avg_output_tokens=200,
                queries_per_day=10000,
            ),
            title="场景 1: 客服 QA, 500/200 token, 1万次/天",
        )
    )

    print(
        format_table(
            compare_models(
                ["qwen-turbo", "qwen-plus", "qwen-max", "gpt-4o"],
                avg_input_tokens=3000,
                avg_output_tokens=500,
                queries_per_day=100000,
            ),
            title="场景 2: 长文档 RAG, 3K/500 token, 10万次/天",
        )
    )

    print(
        format_table(
            compare_models(
                ["qwen-turbo", "qwen-plus", "qwen2.5-1.5b-local", "qwen2.5-7b-local"],
                avg_input_tokens=200,
                avg_output_tokens=100,
                queries_per_day=1000000,
            ),
            title="场景 3: 风控/分类, 200/100 token, 100万次/天",
        )
    )
