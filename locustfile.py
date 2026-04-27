"""
AI Gateway 压测 v2
分离测试：
- /unique 路径：每次请求都是新内容（绕过缓存，测真实模型性能）
- /hot    路径：固定问题（测缓存命中性能）
"""
import random
import uuid
from locust import HttpUser, task, between


class GatewayUser(HttpUser):
    wait_time = between(0.5, 2)

    def on_start(self):
        self.client.headers = {
            "Authorization": "Bearer sk-demo-002",  # enterprise tier 不被限流
            "Content-Type": "application/json",
        }

    @task(70)  # 70% - 真实模型推理（每次prompt不同，强制cache miss）
    def unique_query(self):
        # 加随机ID确保每次prompt都不同，绕过缓存
        unique_id = uuid.uuid4().hex[:8]
        self.client.post(
            "/v1/chat/completions",
            json={
                "model": "fast",
                "messages": [{
                    "role": "user",
                    "content": f"[{unique_id}] 用一句话告诉我现在的时间是什么概念"
                }],
                "max_tokens": 30,
            },
            name="/unique (cache MISS)",
        )

    @task(25)  # 25% - 缓存命中场景
    def hot_query(self):
        self.client.post(
            "/v1/chat/completions",
            json={
                "model": "fast",
                "messages": [{"role": "user", "content": "什么是大模型"}],
                "max_tokens": 50,
            },
            name="/hot (cache HIT)",
        )

    @task(5)  # 5% - 高质量请求（Q8 模型）
    def quality_query(self):
        unique_id = uuid.uuid4().hex[:8]
        self.client.post(
            "/v1/chat/completions",
            json={
                "model": "quality",
                "messages": [{
                    "role": "user",
                    "content": f"[{unique_id}] 简要说明Transformer注意力机制"
                }],
                "max_tokens": 80,
            },
            name="/quality (Q8)",
        )
