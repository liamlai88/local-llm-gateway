"""
Context Engineering 模块 - 实验 #14
四策略对照: Baseline / Write / Select / Compress

每个策略实现 build_messages(history, current_user_msg, system_prompt) -> List[Dict]
"""

import os
import requests
from typing import List, Dict

BAILIAN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
EMBED_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"


def _api_key() -> str:
    k = os.environ.get("DASHSCOPE_API_KEY")
    if not k:
        raise RuntimeError("DASHSCOPE_API_KEY 未设置")
    return k


def call_llm(
    messages: List[Dict], model: str = "qwen-turbo", max_tokens: int = 500
) -> Dict:
    """返回 {answer, input_tokens, output_tokens, total_tokens}"""
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
    if "choices" not in data:
        raise RuntimeError(f"LLM error: {data}")
    usage = data.get("usage", {})
    return {
        "answer": data["choices"][0]["message"]["content"],
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


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


# ========== 策略基类 ==========
class ContextStrategy:
    """每个策略管理一段对话历史，提供 build_messages 接口"""

    name = "base"

    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.history: List[Dict] = []  # [{role, content}, ...]

    def build_messages(self, user_msg: str) -> List[Dict]:
        raise NotImplementedError

    def add_turn(self, user_msg: str, assistant_msg: str):
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": assistant_msg})


# ========== 1. Baseline: 全量历史 ==========
class BaselineStrategy(ContextStrategy):
    name = "baseline"

    def build_messages(self, user_msg: str) -> List[Dict]:
        return (
            [{"role": "system", "content": self.system_prompt}]
            + self.history
            + [{"role": "user", "content": user_msg}]
        )


# ========== 2. Write: scratchpad 摘要 ==========
class WriteStrategy(ContextStrategy):
    """每轮结束后让 LLM 把关键信息写进 scratchpad；下一轮 prompt 只带 scratchpad，不带原始历史"""

    name = "write"

    def __init__(self, system_prompt: str):
        super().__init__(system_prompt)
        self.scratchpad = ""  # 累积关键信息

    def build_messages(self, user_msg: str) -> List[Dict]:
        sp_text = (
            f"\n\n【已记录的关键信息（scratchpad）】\n{self.scratchpad}"
            if self.scratchpad
            else ""
        )
        return [
            {"role": "system", "content": self.system_prompt + sp_text},
            {"role": "user", "content": user_msg},
        ]

    def update_scratchpad(self, user_msg: str, assistant_msg: str):
        """每轮结束让 LLM 决定是否更新 scratchpad"""
        prompt = f"""你是对话记录员。当前 scratchpad:
{self.scratchpad or "(空)"}

新一轮对话:
User: {user_msg}
Assistant: {assistant_msg}

【任务】更新 scratchpad，提取出本轮中**必须长期记住的事实**（姓名、公司、偏好、决定等），合并到原有 scratchpad。
- 直接输出新的完整 scratchpad（不要超过 300 字）
- 闲聊/重复内容不要记
- 用简洁条目格式

输出新 scratchpad（仅文本，不要解释）:"""
        r = call_llm([{"role": "user", "content": prompt}], max_tokens=400)
        self.scratchpad = r["answer"].strip()
        return r["input_tokens"] + r["output_tokens"]


# ========== 3. Select: embedding top-k 召回 ==========
class SelectStrategy(ContextStrategy):
    name = "select"

    def __init__(self, system_prompt: str, top_k: int = 3):
        super().__init__(system_prompt)
        self.top_k = top_k
        self.embeds: List[
            List[float]
        ] = []  # 与 self.history 平行，但只对 user 轮算 embedding

    def add_turn(self, user_msg: str, assistant_msg: str):
        super().add_turn(user_msg, assistant_msg)
        # 给本轮 user+assistant 组合做 embedding
        text = f"User: {user_msg}\nAssistant: {assistant_msg}"
        self.embeds.append(embed([text])[0])

    def build_messages(self, user_msg: str) -> List[Dict]:
        if not self.embeds:
            return [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_msg},
            ]

        # 算 query 和每轮的相似度
        q_emb = embed([user_msg])[0]
        scored = [(cos_sim(q_emb, e), i) for i, e in enumerate(self.embeds)]
        scored.sort(reverse=True)
        top_indices = sorted([i for _, i in scored[: self.top_k]])  # 保持时间顺序

        selected = []
        for i in top_indices:
            selected.append(self.history[i * 2])  # user
            selected.append(self.history[i * 2 + 1])  # assistant

        return (
            [{"role": "system", "content": self.system_prompt}]
            + selected
            + [{"role": "user", "content": user_msg}]
        )


# ========== 4. Compress: 每 N 轮压缩 ==========
class CompressStrategy(ContextStrategy):
    name = "compress"

    def __init__(
        self, system_prompt: str, compress_every: int = 3, keep_recent: int = 2
    ):
        super().__init__(system_prompt)
        self.compress_every = compress_every
        self.keep_recent = keep_recent
        self.summary = ""
        self.compressed_up_to = 0  # history 中已压缩到的轮数（user+assistant 对的索引）

    def maybe_compress(self) -> int:
        """轮数到了就压缩，返回新增 token 数"""
        n_pairs = len(self.history) // 2
        if n_pairs - self.compressed_up_to < self.compress_every:
            return 0

        # 取尚未压缩的部分
        to_compress = self.history[self.compressed_up_to * 2 :]
        text = "\n".join(f"{m['role']}: {m['content']}" for m in to_compress)

        prompt = f"""请把下面对话压缩成不超过 200 字的简洁摘要，**必须保留所有事实信息**（姓名、公司、产品名、价格、决定等），可以省略寒暄。

已有摘要: {self.summary or "(无)"}

新增对话:
{text}

输出新的完整摘要（仅文本）:"""
        r = call_llm([{"role": "user", "content": prompt}], max_tokens=300)
        self.summary = r["answer"].strip()
        self.compressed_up_to = n_pairs
        return r["input_tokens"] + r["output_tokens"]

    def build_messages(self, user_msg: str) -> List[Dict]:
        sys = self.system_prompt
        if self.summary:
            sys += f"\n\n【历史对话摘要】\n{self.summary}"

        # 保留最近 keep_recent 轮原始对话
        recent = self.history[-(self.keep_recent * 2) :] if self.keep_recent else []
        return (
            [{"role": "system", "content": sys}]
            + recent
            + [{"role": "user", "content": user_msg}]
        )
