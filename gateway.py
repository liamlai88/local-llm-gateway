"""
企业级 AI Gateway v2 - 模拟阿里云百炼 MaaS 架构
新增: 流式输出 / 限流 / 监控面板 / 历史记录
"""
import os
import time
import json
import logging
from collections import defaultdict, deque
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import StreamingResponse, HTMLResponse, Response
import httpx
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import rag  # RAG 模块

# ========== 配置 ==========
OLLAMA_BASE = "http://localhost:11434"

MODEL_ROUTES = {
    "fast":     "qwen2.5-1.5b",
    "quality":  "qwen2.5-1.5b-q8",
    "long":     "qwen2.5-1.5b-32k",
}

PRICING = {
    "qwen2.5-1.5b":      {"input": 0.0003, "output": 0.0006},
    "qwen2.5-1.5b-q8":   {"input": 0.0008, "output": 0.0016},
    "qwen2.5-1.5b-32k":  {"input": 0.0010, "output": 0.0020},
}

# 限流配置（每分钟最大请求数）
RATE_LIMITS = {
    "free":       10,
    "enterprise": 1000,
}

API_KEYS = {
    "sk-demo-001": {"name": "测试客户A", "tier": "free"},
    "sk-demo-002": {"name": "企业客户B", "tier": "enterprise"},
}

# 内容审核敏感词库（生产环境会用专业服务，如阿里云内容安全）
BANNED_WORDS = ["暴力", "色情", "赌博", "诈骗", "毒品"]

# ========== 状态存储 ==========
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
log = logging.getLogger("gateway")

app = FastAPI(title="AI Gateway v2")

stats = {
    "total_requests": 0, "total_cost": 0.0, "total_tokens": 0,
    "blocked": 0, "cache_hits": 0,
}
recent_logs = deque(maxlen=50)              # 最近50条请求
rate_window = defaultdict(lambda: deque())  # 每个key的请求时间戳
response_cache = {}                          # 简单内存缓存 {hash: response}

# ========== Prometheus 指标 ==========
# 请求计数器（按用户、模型、状态拆维度）
m_requests = Counter(
    "ai_gateway_requests_total",
    "Total chat completion requests",
    ["user", "model", "status"],
)
# 延迟直方图（自动算 P50/P95/P99）
m_latency = Histogram(
    "ai_gateway_latency_seconds",
    "Request latency by model",
    ["model"],
    buckets=(0.01, 0.05, 0.1, 0.3, 0.5, 1, 2, 5, 10),
)
# Token 消耗
m_tokens = Counter(
    "ai_gateway_tokens_total",
    "Total tokens by direction and model",
    ["direction", "model"],  # direction: input/output
)
# 成本累计
m_cost = Counter(
    "ai_gateway_cost_cny_total",
    "Total cost in CNY",
    ["model"],
)
# 缓存命中率
m_cache = Counter(
    "ai_gateway_cache_total",
    "Cache hits and misses",
    ["result"],  # hit/miss
)
# 内容审核拦截
m_blocked = Counter(
    "ai_gateway_blocked_total",
    "Blocked requests by moderation",
    ["word"],
)
# 在飞请求数（瞬时值）
m_in_flight = Gauge(
    "ai_gateway_in_flight",
    "Currently processing requests",
)


# ========== 工具函数 ==========
def calc_cost(model: str, p_tokens: int, c_tokens: int) -> float:
    if model not in PRICING:
        return 0.0
    p = PRICING[model]
    return (p_tokens * p["input"] + c_tokens * p["output"]) / 1000


def resolve_model(name: str) -> str:
    return MODEL_ROUTES.get(name, name)


def auth(api_key: Optional[str]) -> tuple[str, dict]:
    if not api_key or not api_key.startswith("Bearer "):
        raise HTTPException(401, "Missing Authorization header")
    key = api_key.replace("Bearer ", "")
    if key not in API_KEYS:
        raise HTTPException(401, "Invalid API key")
    return key, API_KEYS[key]


def moderate(messages: list) -> Optional[str]:
    """内容审核：返回命中的敏感词，None表示通过"""
    text = " ".join(m.get("content", "") for m in messages)
    for word in BANNED_WORDS:
        if word in text:
            return word
    return None


def cache_key(body: dict) -> str:
    """生成缓存key（基于model+messages的hash）"""
    import hashlib
    payload = json.dumps({
        "model": body.get("model"),
        "messages": body.get("messages"),
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()


def check_rate_limit(key: str, tier: str):
    """滑动窗口限流：1分钟内的请求数"""
    now = time.time()
    window = rate_window[key]
    # 移除1分钟前的记录
    while window and window[0] < now - 60:
        window.popleft()
    limit = RATE_LIMITS.get(tier, 10)
    if len(window) >= limit:
        raise HTTPException(429, f"Rate limit exceeded: {limit}/min for {tier} tier")
    window.append(now)


# ========== 业务路由 ==========
@app.get("/v1/models")
def list_models():
    return {"data": [
        {"id": k, "actual": v, "pricing": PRICING.get(v, {})}
        for k, v in MODEL_ROUTES.items()
    ]}


@app.post("/v1/chat/completions")
async def chat(request: Request, authorization: Optional[str] = Header(None)):
    key, user = auth(authorization)
    check_rate_limit(key, user["tier"])

    body = await request.json()
    requested = body.get("model", "fast")
    actual = resolve_model(requested)
    body["model"] = actual
    is_stream = body.get("stream", False)

    # 中间件1: 内容审核
    hit = moderate(body.get("messages", []))
    if hit:
        stats["blocked"] += 1
        m_blocked.labels(word=hit).inc()
        m_requests.labels(user=user["name"], model=actual, status="blocked").inc()
        record_request(user, requested, actual, 0, 0, 0, 0, status=f"blocked:{hit}")
        raise HTTPException(400, f"Content moderation failed: '{hit}' is not allowed")

    # 中间件2: 缓存命中（仅非流式）
    if not is_stream:
        ck = cache_key(body)
        if ck in response_cache:
            stats["cache_hits"] += 1
            m_cache.labels(result="hit").inc()
            m_requests.labels(user=user["name"], model=actual, status="cache_hit").inc()
            cached = response_cache[ck].copy()
            cached["x_gateway"] = {**cached.get("x_gateway", {}), "cache": "HIT", "cost_cny": 0}
            record_request(user, requested, actual, 0, 0, 0, 1, status="cache_hit")
            return cached
        m_cache.labels(result="miss").inc()

    if is_stream:
        return StreamingResponse(
            stream_chat(body, user, requested, actual),
            media_type="text/event-stream",
        )

    # 非流式
    m_in_flight.inc()
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{OLLAMA_BASE}/v1/chat/completions", json=body)
            data = resp.json()
    finally:
        m_in_flight.dec()

    latency = time.time() - start
    usage = data.get("usage", {})
    p_tok = usage.get("prompt_tokens", 0)
    c_tok = usage.get("completion_tokens", 0)
    cost = calc_cost(actual, p_tok, c_tok)

    # Prometheus 指标更新
    m_requests.labels(user=user["name"], model=actual, status="ok").inc()
    m_latency.labels(model=actual).observe(latency)
    m_tokens.labels(direction="input", model=actual).inc(p_tok)
    m_tokens.labels(direction="output", model=actual).inc(c_tok)
    m_cost.labels(model=actual).inc(cost)

    record_request(user, requested, actual, p_tok, c_tok, cost, latency)

    data["x_gateway"] = {
        "actual_model": actual,
        "cost_cny": round(cost, 6),
        "latency_ms": round(latency * 1000, 1),
        "cache": "MISS",
    }
    # 写入缓存
    response_cache[cache_key(body)] = data
    return data


async def stream_chat(body, user, requested, actual):
    """流式输出 - SSE 协议"""
    start = time.time()
    p_tok = c_tok = 0
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", f"{OLLAMA_BASE}/v1/chat/completions", json=body) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    chunk = line[6:]
                    if chunk.strip() == "[DONE]":
                        yield f"data: [DONE]\n\n"
                        break
                    try:
                        obj = json.loads(chunk)
                        if "usage" in obj and obj["usage"]:
                            p_tok = obj["usage"].get("prompt_tokens", 0)
                            c_tok = obj["usage"].get("completion_tokens", 0)
                    except:
                        pass
                    yield f"data: {chunk}\n\n"

    latency = time.time() - start
    cost = calc_cost(actual, p_tok, c_tok)
    record_request(user, requested, actual, p_tok, c_tok, cost, latency)


def record_request(user, requested, actual, p_tok, c_tok, cost, latency, status="ok"):
    """统一记录请求"""
    stats["total_requests"] += 1
    stats["total_tokens"] += p_tok + c_tok
    stats["total_cost"] += cost

    record = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "user": user["name"],
        "model": actual,
        "requested": requested,
        "tokens": p_tok + c_tok,
        "cost": round(cost, 6),
        "latency_ms": round(latency * 1000, 1),
        "status": status,
    }
    recent_logs.appendleft(record)
    log.info(json.dumps(record, ensure_ascii=False))


@app.get("/v1/stats")
def get_stats():
    return {**stats, "recent": list(recent_logs)}


@app.get("/metrics")
def metrics():
    """Prometheus 抓取端点"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ========== 健康检查 ==========
@app.get("/v1/health")
def health():
    """完整健康检查：依赖、API Key、模型路由"""
    deps = rag.check_deps()
    rag_ready = all([deps["chromadb"], deps["dashscope"]])
    bm25_ready = all([deps["rank_bm25"], deps["jieba"]])
    has_dashscope_key = bool(os.getenv("DASHSCOPE_API_KEY"))

    issues = []
    if not rag_ready:
        missing = [k for k, v in deps.items() if not v and k in ("chromadb", "dashscope")]
        issues.append(f"RAG 不可用，缺依赖: {missing} → pip install {' '.join(missing)}")
    if not bm25_ready:
        missing = [k for k, v in deps.items() if not v and k in ("rank_bm25", "jieba")]
        issues.append(f"BM25 不可用，缺依赖: {missing} → pip install rank-bm25 jieba")
    if rag_ready and not has_dashscope_key:
        issues.append("DASHSCOPE_API_KEY 未设置 → 重启时加 prefix")

    return {
        "status": "ok" if not issues else "degraded",
        "features": {
            "chat": True,
            "rag_vector": rag_ready and has_dashscope_key,
            "rag_bm25": bm25_ready,
            "rag_hybrid": rag_ready and bm25_ready and has_dashscope_key,
        },
        "dependencies": deps,
        "env": {"DASHSCOPE_API_KEY": "set" if has_dashscope_key else "missing"},
        "issues": issues,
        "models": list(MODEL_ROUTES.keys()),
    }


# ========== RAG 接口 ==========
def _rag_safe(func):
    """装饰器：把 RAG 的 RuntimeError 转成 503 + 友好提示"""
    from functools import wraps
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except RuntimeError as e:
            raise HTTPException(503, str(e))
    return wrapper


@app.post("/v1/rag/documents")
@_rag_safe
async def rag_upload(request: Request, authorization: Optional[str] = Header(None)):
    """上传文档到知识库（自动切块+向量化）"""
    auth(authorization)
    body = await request.json()
    doc_id = body.get("doc_id") or hashlib.md5(body.get("content", "").encode()).hexdigest()[:8]
    return rag.add_document(doc_id, body["content"], body.get("metadata"))


@app.post("/v1/rag/query")
@_rag_safe
async def rag_query(request: Request, authorization: Optional[str] = Header(None)):
    """RAG 查询：检索 + 增强生成"""
    key, user = auth(authorization)
    body = await request.json()
    question = body["question"]
    top_k = body.get("top_k", 3)
    mode = body.get("mode", "hybrid")  # vector / bm25 / hybrid
    use_rerank = body.get("rerank", False)
    requested_model = body.get("model", "fast")
    actual_model = resolve_model(requested_model)

    # Step 1: 检索（可选 Rerank 精排）
    start = time.time()
    chunks = rag.search(question, top_k=top_k, mode=mode, use_rerank=use_rerank)
    retrieval_ms = (time.time() - start) * 1000

    # Step 2: 拼接增强 Prompt
    context = "\n\n".join([f"[文档片段{i+1}] {c['content']}" for i, c in enumerate(chunks)])
    augmented_prompt = f"""请基于以下文档内容回答用户问题。如果文档中没有相关信息，请明确说明"文档中没有相关信息"，不要编造。

{context}

用户问题：{question}

请回答："""

    # Step 3: 调用 LLM
    gen_start = time.time()
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{OLLAMA_BASE}/v1/chat/completions",
            json={
                "model": actual_model,
                "messages": [{"role": "user", "content": augmented_prompt}],
            },
        )
        data = resp.json()
    gen_ms = (time.time() - gen_start) * 1000

    # 提取统计
    usage = data.get("usage", {})
    p_tok = usage.get("prompt_tokens", 0)
    c_tok = usage.get("completion_tokens", 0)
    cost = calc_cost(actual_model, p_tok, c_tok)

    # Prometheus 指标
    m_requests.labels(user=user["name"], model=actual_model, status="rag").inc()
    m_tokens.labels(direction="input", model=actual_model).inc(p_tok)
    m_tokens.labels(direction="output", model=actual_model).inc(c_tok)
    m_cost.labels(model=actual_model).inc(cost)
    record_request(user, "rag", actual_model, p_tok, c_tok, cost, (retrieval_ms + gen_ms) / 1000, status="rag")

    return {
        "answer": data["choices"][0]["message"]["content"],
        "sources": chunks,
        "stats": {
            "retrieval_mode": mode,
            "rerank": use_rerank,
            "retrieval_ms": round(retrieval_ms, 1),
            "generation_ms": round(gen_ms, 1),
            "total_ms": round(retrieval_ms + gen_ms, 1),
            "tokens": p_tok + c_tok,
            "cost_cny": round(cost, 6),
            "model": actual_model,
        },
    }


@app.get("/v1/rag/stats")
def rag_stats():
    return rag.stats()


@app.delete("/v1/rag/documents")
@_rag_safe
async def rag_clear(authorization: Optional[str] = Header(None)):
    auth(authorization)
    rag.clear_all()
    return {"status": "cleared"}


# ========== 监控面板 ==========
@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>AI Gateway 监控面板</title>
<style>
  body { font-family: -apple-system, sans-serif; margin: 0; padding: 20px; background: #f5f5f7; }
  h1 { margin: 0 0 20px; }
  .cards { display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; margin-bottom: 24px; }
  .card { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  .card h3 { margin: 0 0 8px; font-size: 13px; color: #888; font-weight: 500; }
  .card .v { font-size: 32px; font-weight: 600; color: #1d1d1f; }
  table { width: 100%; background: white; border-radius: 12px; border-collapse: collapse; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  th, td { padding: 12px 16px; text-align: left; font-size: 13px; }
  th { background: #fafafa; color: #666; font-weight: 500; border-bottom: 1px solid #eee; }
  tr:not(:last-child) td { border-bottom: 1px solid #f0f0f0; }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; background: #e5f3ff; color: #0066cc; }
</style>
</head>
<body>
<h1>🚀 AI Gateway 监控面板</h1>
<div class="cards">
  <div class="card"><h3>总请求数</h3><div class="v" id="req">0</div></div>
  <div class="card"><h3>总Token消耗</h3><div class="v" id="tok">0</div></div>
  <div class="card"><h3>总成本(¥)</h3><div class="v" id="cost">0.000</div></div>
  <div class="card"><h3>审核拦截</h3><div class="v" id="blocked" style="color:#d83737">0</div></div>
  <div class="card"><h3>缓存命中</h3><div class="v" id="cache" style="color:#0a8a3e">0</div></div>
</div>
<table>
  <thead><tr><th>时间</th><th>客户</th><th>路由</th><th>实际模型</th><th>Token</th><th>成本(¥)</th><th>延迟(ms)</th><th>状态</th></tr></thead>
  <tbody id="logs"></tbody>
</table>
<script>
function statusTag(s) {
  if (s === 'cache_hit') return '<span class="tag" style="background:#dff5e5;color:#0a8a3e">CACHE</span>';
  if (s && s.startsWith('blocked')) return '<span class="tag" style="background:#ffe5e5;color:#d83737">BLOCKED</span>';
  return '<span class="tag" style="background:#eee;color:#666">OK</span>';
}
async function refresh() {
  const r = await fetch('/v1/stats');
  const d = await r.json();
  document.getElementById('req').textContent = d.total_requests;
  document.getElementById('tok').textContent = d.total_tokens.toLocaleString();
  document.getElementById('cost').textContent = d.total_cost.toFixed(6);
  document.getElementById('blocked').textContent = d.blocked || 0;
  document.getElementById('cache').textContent = d.cache_hits || 0;
  document.getElementById('logs').innerHTML = (d.recent || []).map(x =>
    `<tr><td>${x.time}</td><td>${x.user}</td><td><span class="tag">${x.requested}</span></td>
     <td>${x.model}</td><td>${x.tokens}</td><td>${x.cost}</td><td>${x.latency_ms}</td>
     <td>${statusTag(x.status)}</td></tr>`
  ).join('');
}
refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""
