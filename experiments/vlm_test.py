"""
最小 VLM 实验：调用 Qwen-VL-Plus 分析一张本地图片
"""
import base64
import io
import os
import time
import requests
from PIL import Image

DASHSCOPE_API_KEY = os.environ["DASHSCOPE_API_KEY"]
URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

IMAGE_PATH = "/Users/liam/Calude-Learning/ai-gateway/food.jpeg"

PROMPT = """请分析这张图片，回答两个问题：
1. 是否含有政治敏感或反动内容？（是/否，给出依据）
2. 这张图看起来是真实拍摄还是 AI 生成？（给出判断理由：纹理/光影/手指/文字等）
请用 JSON 输出: {"political_risk": "yes/no", "political_reason": "...", "ai_generated": "yes/no/unsure", "ai_reason": "..."}
"""

orig_size = os.path.getsize(IMAGE_PATH)
img = Image.open(IMAGE_PATH).convert("RGB")
# 长边 resize 到 1024（VLM 标准）
img.thumbnail((1024, 1024))
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=85)
img_bytes = buf.getvalue()
img_b64 = base64.b64encode(img_bytes).decode()

print(f"📷 图片: {IMAGE_PATH}")
print(f"📦 原始: {orig_size/1024:.0f} KB → 压缩后: {len(img_bytes)/1024:.0f} KB ({img.size[0]}×{img.size[1]})")

t0 = time.time()
resp = requests.post(
    URL,
    headers={"Authorization": f"Bearer {DASHSCOPE_API_KEY}"},
    json={
        "model": "qwen-vl-plus",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        "temperature": 0.1,
    },
    timeout=60,
)
latency = time.time() - t0

print(f"\n⏱  延迟: {latency:.2f}s")
data = resp.json()
if "choices" in data:
    print(f"💬 VLM 输出:\n{data['choices'][0]['message']['content']}")
    if "usage" in data:
        u = data["usage"]
        print(f"\n📊 Token 用量: 输入 {u.get('prompt_tokens', '?')} | 输出 {u.get('completion_tokens', '?')}")
        cost = u.get("prompt_tokens", 0) * 0.008 / 1000 + u.get("completion_tokens", 0) * 0.02 / 1000
        print(f"💰 本次成本: ¥{cost:.4f}")
else:
    print(f"❌ 错误: {data}")
