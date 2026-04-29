"""
生成 LoRA 训练数据：让 Qwen2.5-1.5B 学会 JSON 工具调用规划

数据格式：MLX-LM 要求的 messages 格式
  {"messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<JSON 工具调用计划>"}
  ]}

训练目标：模型看到问题，能输出结构化的工具调用 JSON Plan
"""
import json
import random
from pathlib import Path

random.seed(42)

# ========== System Prompt ==========
SYSTEM_PROMPT = """你是任务规划器。给定用户问题，输出 JSON 格式的工具执行计划。

可用工具：
- calculator(expression): 数学计算
- get_weather(city): 城市天气查询
- kb_search(query): 知识库检索
- extract_number(text, hint): 从文本提取数字

输出格式（严格 JSON，不要任何额外文字）：
{"plan": [{"step": 序号, "tool": "工具名", "args": {...}, "purpose": "本步目的"}]}"""


# ========== 模板生成器 ==========

# 1. 简单计算（25 条）
def gen_simple_calc():
    samples = []
    operations = [
        ("加", "+"), ("减", "-"), ("乘以", "*"), ("除以", "/"),
        ("乘", "*"), ("加上", "+"), ("减去", "-"),
    ]
    for _ in range(25):
        a = random.randint(1, 200)
        b = random.randint(1, 50)
        op_word, op = random.choice(operations)
        question = random.choice([
            f"{a} {op_word} {b} 等于多少？",
            f"帮我算 {a} {op_word} {b}",
            f"请计算 {a} {op_word} {b} 的结果",
            f"{a}{op_word}{b}是多少",
        ])
        plan = {
            "plan": [
                {"step": 1, "tool": "calculator", "args": {"expression": f"{a} {op} {b}"}, "purpose": "执行数学计算"}
            ]
        }
        samples.append((question, plan))
    return samples


# 2. 单工具天气查询（15 条）
def gen_weather():
    cities = ["杭州", "北京", "上海", "深圳", "广州", "成都", "武汉", "西安",
              "新加坡", "迪拜", "伦敦", "纽约", "东京", "首尔", "曼谷"]
    samples = []
    for city in cities:
        question = random.choice([
            f"{city}现在天气怎么样？",
            f"查一下{city}的气温",
            f"{city}今天的天气",
            f"告诉我{city}天气情况",
        ])
        plan = {
            "plan": [
                {"step": 1, "tool": "get_weather", "args": {"city": city}, "purpose": f"查询{city}天气"}
            ]
        }
        samples.append((question, plan))
    return samples


# 3. 多工具：气温差（15 条）
def gen_weather_diff():
    cities = ["杭州", "北京", "上海", "深圳", "新加坡", "迪拜", "伦敦", "东京"]
    samples = []
    for _ in range(15):
        c1, c2 = random.sample(cities, 2)
        question = random.choice([
            f"{c1}和{c2}的气温差是多少？",
            f"{c1}比{c2}冷还是热，差几度？",
            f"{c1}减去{c2}的气温",
            f"{c1}现在的气温减去{c2}的气温",
        ])
        plan = {
            "plan": [
                {"step": 1, "tool": "get_weather", "args": {"city": c1}, "purpose": f"查{c1}气温"},
                {"step": 2, "tool": "get_weather", "args": {"city": c2}, "purpose": f"查{c2}气温"},
                {"step": 3, "tool": "extract_number", "args": {"text": "<step1_result>", "hint": "气温"}, "purpose": "提取第一个城市气温"},
                {"step": 4, "tool": "extract_number", "args": {"text": "<step2_result>", "hint": "气温"}, "purpose": "提取第二个城市气温"},
                {"step": 5, "tool": "calculator", "args": {"expression": "<step3_result> - <step4_result>"}, "purpose": "计算气温差"},
            ]
        }
        samples.append((question, plan))
    return samples


# 4. 单工具知识库查询（15 条）
def gen_kb_simple():
    queries = [
        "A100 实例有什么规格？", "A10 实例的显存是多少？",
        "灵骏集群是什么？", "什么 GPU 适合大模型训练？",
        "A100 的包月价格是多少？", "A10 实例适合什么场景？",
        "阿里云有哪些 GPU 产品？", "ecs.gn7e 是什么？",
        "PAI-Lingjun 集群规模？", "H100 集群怎么定价？",
        "A100 显存多大？", "A10 GPU 几个核？",
        "万卡训练需要什么集群？", "GPU 实例的 SLA 怎么保障？",
        "AI 推理用什么 GPU？",
    ]
    samples = []
    for q in queries:
        plan = {
            "plan": [
                {"step": 1, "tool": "kb_search", "args": {"query": q}, "purpose": "检索知识库"}
            ]
        }
        samples.append((q, plan))
    return samples


# 5. 复杂：RAG + 计算（20 条）⭐ 重点
def gen_rag_calc():
    samples = []
    products = ["A100", "A10", "H100"]
    multipliers = [
        ("一天", 24, "24 小时"), ("一周", 24*7, "一周 168 小时"),
        ("一个月", 24*30, "一个月 720 小时"), ("3 天", 72, "3 天 72 小时"),
        ("半天", 12, "12 小时"), ("一年", 24*365, "一年 8760 小时"),
        ("8 小时工作日", 8, "8 小时"),
    ]
    purposes = ["按量付费", "总成本", "持续运行", "完整租用"]
    for _ in range(20):
        product = random.choice(products)
        period, hours, hint = random.choice(multipliers)
        purpose = random.choice(purposes)
        question = random.choice([
            f"阿里云 {product} 实例{purpose}{period}大概多少钱？",
            f"{product} 跑{period}（{hint}）成本是多少？",
            f"我用 {product} 实例{period}，{purpose}多少钱？基于知识库价格算",
            f"按知识库价格，{product} {purpose}{period}（{hint}）总价",
        ])
        plan = {
            "plan": [
                {"step": 1, "tool": "kb_search", "args": {"query": f"{product} 按量付费每小时价格"}, "purpose": f"检索 {product} 价格"},
                {"step": 2, "tool": "extract_number", "args": {"text": "<step1_result>", "hint": "按量付费每小时价格"}, "purpose": "提取每小时单价"},
                {"step": 3, "tool": "calculator", "args": {"expression": f"<step2_result> * {hours}"}, "purpose": f"计算 {hint} 总价"},
            ]
        }
        samples.append((question, plan))
    return samples


# 6. 边界：纯对话（10 条）— 让模型学会"无需工具时不调用"
def gen_no_tool():
    samples = []
    no_tool_qa = [
        ("你好", "你好！我是任务规划器。请告诉我具体需要查询、计算或检索的问题，我会为你规划工具调用步骤。"),
        ("你是谁？", "我是任务规划器，专门把用户问题分解成工具调用计划。"),
        ("能帮我做什么？", "我可以规划工具调用：数学计算、天气查询、知识库检索、数字提取等。"),
        ("谢谢", "不客气，有需要查询或计算的问题随时告诉我。"),
        ("再见", "再见！"),
    ]
    for q, a in no_tool_qa:
        # 这种情况输出空计划 + 自然语言
        plan = {"plan": [], "direct_answer": a}
        samples.append((q, plan))
    # 重复几次扩展数据
    return samples * 2  # 共 10 条


# ========== 汇总并写文件 ==========
def to_messages(question, plan_obj):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": json.dumps(plan_obj, ensure_ascii=False, indent=2)},
        ]
    }


def main():
    samples = []
    samples += gen_simple_calc()
    samples += gen_weather()
    samples += gen_weather_diff()
    samples += gen_kb_simple()
    samples += gen_rag_calc()
    samples += gen_no_tool()

    random.shuffle(samples)
    print(f"✓ 总共生成 {len(samples)} 条样本")

    # 8:1:1 划分
    n = len(samples)
    train_n = int(n * 0.8)
    valid_n = int(n * 0.1)

    train = samples[:train_n]
    valid = samples[train_n:train_n + valid_n]
    test = samples[train_n + valid_n:]

    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)

    for name, data in [("train", train), ("valid", valid), ("test", test)]:
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for q, plan in data:
                f.write(json.dumps(to_messages(q, plan), ensure_ascii=False) + "\n")
        print(f"  → {path.relative_to(Path.cwd())}: {len(data)} 条")


if __name__ == "__main__":
    main()
