"""
对比实验：本地 1.5B vs 百炼 Qwen-Max
同一道题，看小模型 + 好 Prompt 能不能逼近大模型
"""
import os
import requests
from openai import OpenAI

QUESTION = "小明有20个苹果，给了小红1/4，又卖掉剩下的60%，现在还剩多少？"
COT_PROMPT = f"{QUESTION}\n\n让我们一步一步分析，先列已知条件，再分步计算。"

# ========== 本地 1.5B ==========
def ask_local(prompt):
    resp = requests.post(
        "http://localhost:8000/v1/chat/completions",
        headers={"Authorization": "Bearer sk-demo-002", "Content-Type": "application/json"},
        json={"model": "fast", "messages": [{"role": "user", "content": prompt}], "max_tokens": 300},
    )
    return resp.json()["choices"][0]["message"]["content"]


# ========== 百炼 Qwen-Max ==========
def ask_bailian(prompt, model="qwen-max"):
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


# ========== 实验 ==========
print("="*60)
print("实验1: 本地 1.5B  +  Zero-shot（直接问）")
print("="*60)
print(ask_local(QUESTION))

print("\n" + "="*60)
print("实验2: 本地 1.5B  +  CoT（一步步思考）")
print("="*60)
print(ask_local(COT_PROMPT))

print("\n" + "="*60)
print("实验3: 百炼 Qwen-Turbo  +  Zero-shot")
print("="*60)
print(ask_bailian(QUESTION, "qwen-turbo"))

print("\n" + "="*60)
print("实验4: 百炼 Qwen-Max  +  Zero-shot")
print("="*60)
print(ask_bailian(QUESTION, "qwen-max"))

print("\n" + "="*60)
print("正确答案: 20 × 1/4 = 5 → 剩15 → 15 × 60% = 9 卖掉 → 剩 6 个")
print("="*60)
