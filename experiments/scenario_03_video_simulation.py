"""
Scenario 03 本地模拟：中东短视频平台 Multi-Agent 视频审核

模拟策略：
- Frame/OCR/ASR/Context 4 路 Agent 用 mock（带真实延迟）
- Analyst Agent 用真实 Qwen-Turbo（通过百炼）
- 对比：并行执行 vs 串行执行的延迟差异
- 三个测试视频：明确通过 / 明确违规 / 模糊判断
"""
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List

import requests

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
BAILIAN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


# ── 三个模拟视频 ────────────────────────────────────────
VIDEOS = {
    "video_safe": {
        "title": "美食教程：传统沙特卡布萨饭做法",
        "duration_sec": 60,
        "frames": ["厨房场景", "羊肉特写", "锅中翻炒", "成品摆盘"],
        "ocr_text": "Kabsa Recipe | Traditional Saudi Dish",
        "asr_text": "今天教大家做卡布萨饭，先准备 500 克羊肉，加入香料炒香。",
        "user_history": "creator_id=u_8821, 历史 230 个视频, 0 违规记录, 美食类创作者",
    },
    "video_violation": {
        "title": "未知标题",
        "duration_sec": 45,
        "frames": ["蒙面人物", "武器特写", "极端旗帜", "城市远景"],
        "ocr_text": "[阿拉伯文] 加入我们 | 联系方式: ***",
        "asr_text": "呼吁加入组织，对抗政府...（极端宣传内容）",
        "user_history": "creator_id=u_9912, 新注册7天, 已有 2 次警告记录",
    },
    "video_ambiguous": {
        "title": "纪录片片段：中东历史回顾",
        "duration_sec": 90,
        "frames": ["历史画面", "新闻片段", "采访场景", "地图展示"],
        "ocr_text": "Documentary | Middle East 1970s",
        "asr_text": "本片记录了上世纪的政治变迁，包括对一些争议事件的回顾...",
        "user_history": "creator_id=u_5501, 历史 45 个视频, 1 次申诉成功记录, 历史/教育类账号",
    },
}


@dataclass
class AgentResult:
    agent: str
    latency_ms: float
    evidence: str
    risk_signal: str  # "safe" | "suspect" | "violation"


# ── 4 路 Mock Agent（带真实延迟）─────────────────────────
async def frame_agent(video: Dict) -> AgentResult:
    """Qwen-VL 视觉分析（mock）"""
    await asyncio.sleep(2.5)  # 真实 Qwen-VL-Max 约 2-3s
    frames = video["frames"]
    danger_kw = {"武器", "极端旗帜", "蒙面", "暴力"}
    hits = [f for f in frames if any(k in f for k in danger_kw)]
    if hits:
        return AgentResult("Frame", 2500, f"检测到高危画面: {hits}", "violation")
    return AgentResult("Frame", 2500, f"画面正常: {frames[:2]}...", "safe")


async def ocr_agent(video: Dict) -> AgentResult:
    """OCR 字幕/画面文字（mock）"""
    await asyncio.sleep(0.8)  # OCR API 约 500-1000ms
    text = video["ocr_text"]
    danger_kw = ["加入我们", "联系方式", "组织"]
    if any(k in text for k in danger_kw):
        return AgentResult("OCR", 800, f"字幕含可疑招募文字: '{text}'", "suspect")
    return AgentResult("OCR", 800, f"字幕正常: '{text}'", "safe")


async def asr_agent(video: Dict) -> AgentResult:
    """Paraformer 语音识别（mock）"""
    await asyncio.sleep(3.0)  # 1分钟音频 ASR 约 2-4s
    text = video["asr_text"]
    danger_kw = ["对抗政府", "极端", "呼吁加入"]
    if any(k in text for k in danger_kw):
        return AgentResult("ASR", 3000, f"语音含极端内容: '{text[:30]}...'", "violation")
    politic_kw = ["政治", "争议事件"]
    if any(k in text for k in politic_kw):
        return AgentResult("ASR", 3000, f"语音含敏感话题: '{text[:30]}...'", "suspect")
    return AgentResult("ASR", 3000, f"语音正常: '{text[:30]}...'", "safe")


async def context_agent(video: Dict) -> AgentResult:
    """RAG 用户画像 + 规则库（mock）"""
    await asyncio.sleep(0.5)  # ChromaDB 查询 200-500ms
    history = video["user_history"]
    if "警告" in history or "新注册" in history:
        return AgentResult("Context", 500, f"用户高风险: {history}", "suspect")
    if "申诉成功" in history or "0 违规" in history:
        return AgentResult("Context", 500, f"用户低风险: {history}", "safe")
    return AgentResult("Context", 500, f"用户中性: {history}", "safe")


# ── Analyst Agent（真实 Qwen-Turbo）─────────────────────
def analyst_agent_llm(video_title: str, results: List[AgentResult]) -> Dict:
    """用 Qwen-Turbo 综合四路证据"""
    evidence = "\n".join(
        f"- [{r.agent}] 信号={r.risk_signal} | {r.evidence}" for r in results
    )
    prompt = f"""你是中东视频平台的内容审核 Analyst。基于以下 4 路 Agent 的证据，做最终判断。

视频标题: {video_title}
四路证据:
{evidence}

请输出严格 JSON 格式（不要任何额外文字）:
{{"decision": "通过" 或 "拒绝" 或 "人工复审",
  "confidence": 0.0-1.0,
  "reason": "简短理由（30字内）"}}
"""
    t0 = time.time()
    try:
        resp = requests.post(
            BAILIAN_URL,
            headers={"Authorization": f"Bearer {DASHSCOPE_API_KEY}"},
            json={
                "model": "qwen-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 200,
            },
            timeout=15,
        )
        latency_ms = (time.time() - t0) * 1000
        content = resp.json()["choices"][0]["message"]["content"].strip()
        # 抽 JSON
        import re
        m = re.search(r"\{.*\}", content, re.DOTALL)
        return {"latency_ms": latency_ms, "parsed": json.loads(m.group(0)) if m else {"raw": content}}
    except Exception as e:
        return {"latency_ms": (time.time() - t0) * 1000, "error": str(e)}


# ── 编排：并行 vs 串行 ─────────────────────────────────
async def run_parallel(video: Dict) -> List[AgentResult]:
    return await asyncio.gather(
        frame_agent(video), ocr_agent(video),
        asr_agent(video), context_agent(video),
    )


async def run_sequential(video: Dict) -> List[AgentResult]:
    r1 = await frame_agent(video)
    r2 = await ocr_agent(video)
    r3 = await asr_agent(video)
    r4 = await context_agent(video)
    return [r1, r2, r3, r4]


# ── 主流程 ─────────────────────────────────────────────
async def process_video(name: str, video: Dict, mode: str = "parallel"):
    print(f"\n{'='*70}")
    print(f"📹 视频: {name} — {video['title']}")
    print(f"   时长: {video['duration_sec']}s | 模式: {mode}")
    print(f"{'='*70}")

    t0 = time.time()
    if mode == "parallel":
        results = await run_parallel(video)
    else:
        results = await run_sequential(video)
    agent_latency_ms = (time.time() - t0) * 1000

    print(f"\n[Agent 阶段] 耗时: {agent_latency_ms:.0f}ms ({mode})")
    for r in results:
        emoji = {"safe": "✅", "suspect": "⚠️", "violation": "🚫"}[r.risk_signal]
        print(f"  {emoji} {r.agent:8s} ({r.latency_ms:.0f}ms) {r.evidence}")

    # Analyst Agent (Qwen-Turbo)
    print(f"\n[Analyst LLM] 调用 Qwen-Turbo 综合判断...")
    analyst = analyst_agent_llm(video["title"], results)
    print(f"  ⏱  延迟: {analyst['latency_ms']:.0f}ms")
    if "parsed" in analyst:
        p = analyst["parsed"]
        print(f"  📋 判定: {p.get('decision')} | 置信度: {p.get('confidence')}")
        print(f"  💬 理由: {p.get('reason')}")
    else:
        print(f"  ⚠ 解析失败: {analyst}")

    total_ms = agent_latency_ms + analyst["latency_ms"]
    print(f"\n[总耗时] {total_ms:.0f}ms")
    return {"mode": mode, "agent_ms": agent_latency_ms, "analyst_ms": analyst["latency_ms"], "total_ms": total_ms}


async def main():
    if not DASHSCOPE_API_KEY:
        print("❌ 请先 export DASHSCOPE_API_KEY")
        return

    print("\n" + "█" * 70)
    print("█  Scenario 03 本地模拟：Multi-Agent 视频审核流水线")
    print("█" * 70)

    # 第一轮：并行模式跑三个视频
    summary = []
    for name, video in VIDEOS.items():
        r = await process_video(name, video, mode="parallel")
        r["video"] = name
        summary.append(r)

    # 第二轮：用 video_safe 跑串行，对比延迟
    print("\n\n" + "▓" * 70)
    print("▓  对照实验：并行 vs 串行（video_safe）")
    print("▓" * 70)
    seq = await process_video("video_safe", VIDEOS["video_safe"], mode="sequential")

    # 汇总
    print("\n\n" + "═" * 70)
    print("📊 实验汇总")
    print("═" * 70)
    print(f"\n{'视频':20s} {'模式':12s} {'Agent(ms)':>12s} {'Analyst(ms)':>14s} {'总耗时(ms)':>14s}")
    print("-" * 70)
    for r in summary:
        print(f"{r['video']:20s} {r['mode']:12s} {r['agent_ms']:>12.0f} {r['analyst_ms']:>14.0f} {r['total_ms']:>14.0f}")
    print(f"{'video_safe':20s} {'sequential':12s} {seq['agent_ms']:>12.0f} {seq['analyst_ms']:>14.0f} {seq['total_ms']:>14.0f}")

    parallel_safe = summary[0]["agent_ms"]
    print(f"\n🎯 关键结论:")
    print(f"   并行 Agent 阶段: {parallel_safe:.0f}ms (= max 各路延迟)")
    print(f"   串行 Agent 阶段: {seq['agent_ms']:.0f}ms (= sum 各路延迟)")
    print(f"   并行加速比: {seq['agent_ms']/parallel_safe:.2f}x")


if __name__ == "__main__":
    asyncio.run(main())
