"""
实验 1: LangChain 重写 RAG (vs 你的 rag.py)

对比目标:
  - 你的 rag.py: 自实现 chunk + embed + chroma + retrieve + LLM, ~250 行
  - LangChain 版本: ~80 行 (含注释)

跑这个脚本前确保:
  - Ollama 在跑 (qwen2.5-1.5b)
  - DASHSCOPE_API_KEY 已设置 (用百炼 Embedding)
"""
import os
import time
from typing import List

# LangChain 核心组件
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# LangChain 集成
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings


# ========== Step 1: 准备文档 ==========
docs_text = {
    "product_a100": """产品名称: 阿里云 GPU 计算实例 ecs.gn7e-c12g1.3xlarge
GPU 型号: NVIDIA A100, 显存: 80GB HBM2e
定价: 按量付费 ¥68/小时, 包月 ¥35000
适用场景: 大模型训练、微调、高吞吐推理""",

    "product_a10": """产品名称: 阿里云 GPU 计算实例 ecs.gn7i-c8g1.2xlarge
GPU 型号: NVIDIA A10, 显存: 24GB GDDR6
定价: 按量付费 ¥18/小时, 包月 ¥9800
适用场景: AI 推理、轻量微调、图形渲染""",
}

# 转成 LangChain Document 对象
documents = [
    Document(page_content=text, metadata={"source": doc_id})
    for doc_id, text in docs_text.items()
]


# ========== Step 2: Embedding + 向量库 ==========
print("→ 初始化 Embedding (百炼) + Chroma 向量库...")
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
)

# Chroma 自动 embed + 入库 (一行搞定，对比你的 rag.py 几十行)
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="langchain_rag_demo",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


# ========== Step 3: 定义 LLM ==========
llm = ChatOllama(
    model="qwen2.5-1.5b",
    base_url="http://localhost:11434",
    temperature=0.0,
)


# ========== Step 4: Prompt 模板 ==========
prompt = ChatPromptTemplate.from_template("""请基于以下文档回答用户问题，不要编造文档外的信息。

【文档】
{context}

【问题】
{question}

【回答】""")


# ========== Step 5: 拼接成 Chain (LCEL 核心) ==========
def format_docs(docs: List[Document]) -> str:
    return "\n\n---\n\n".join([d.page_content for d in docs])


# 这就是 LangChain 的灵魂：管道操作符 |
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# ========== Step 6: 测试 ==========
TESTS = [
    "A100 实例的按量付费每小时多少钱？",
    "A10 适合做什么？",
    "如果我要训练大模型，推荐用哪个？",
]

print("\n" + "=" * 70)
print("LangChain RAG 测试")
print("=" * 70)

for i, q in enumerate(TESTS, 1):
    start = time.time()
    answer = rag_chain.invoke(q)
    elapsed = (time.time() - start) * 1000
    print(f"\n[Q{i}] {q}")
    print(f"    答: {answer.strip()[:200]}")
    print(f"    延迟: {elapsed:.0f}ms")

print("\n" + "=" * 70)
print("📊 代码量对比")
print("=" * 70)
print("  你的 rag.py:        ~250 行 (含切块/索引/检索/混合 Search)")
print("  LangChain 版:        ~80 行 (本脚本)")
print("  减少:                68%")
print("  代价: 无法实现 Hybrid Search (LangChain 不直接支持 BM25 融合)")
