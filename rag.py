"""
RAG 模块 - 支撑 Gateway 的检索增强能力
- Chunking: 按段落 + 重叠切块
- Embedding: 百炼 text-embedding-v2 (1536 维)
- Vector Store: ChromaDB (嵌入式)
- Retrieval: 余弦相似度 Top-K
"""
import os
import re
import hashlib
from typing import List, Dict
import chromadb
import dashscope
from dashscope import TextEmbedding

# ========== 配置 ==========
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# 持久化向量库 (重启后数据还在)
chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"description": "RAG 知识库"},
)


# ========== Chunking ==========
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    按段落 + 重叠切块
    - 优先按段落（双换行）切
    - 段落太长则按句子切
    - 相邻块保留 overlap 字重叠
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) <= chunk_size:
            buf += "\n" + p if buf else p
        else:
            if buf:
                chunks.append(buf)
            # 段落本身超长，按句号继续切
            if len(p) > chunk_size:
                sentences = re.split(r"(?<=[。！？.!?])", p)
                buf = ""
                for s in sentences:
                    if len(buf) + len(s) <= chunk_size:
                        buf += s
                    else:
                        if buf:
                            chunks.append(buf)
                        buf = s
            else:
                buf = p
    if buf:
        chunks.append(buf)

    # 加 overlap：每块开头加上一块的最后 overlap 字
    if overlap > 0 and len(chunks) > 1:
        out = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            out.append(prev_tail + " " + chunks[i])
        return out
    return chunks


# ========== Embedding (百炼) ==========
def embed(texts: List[str]) -> List[List[float]]:
    """调用百炼 text-embedding-v2，返回 1536 维向量列表"""
    if not dashscope.api_key:
        raise RuntimeError("DASHSCOPE_API_KEY 未设置")

    # 百炼 v2 单次最多 25 条
    all_embeds = []
    for i in range(0, len(texts), 25):
        batch = texts[i:i + 25]
        resp = TextEmbedding.call(
            model="text-embedding-v2",
            input=batch,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Embedding failed: {resp.message}")
        for item in resp.output["embeddings"]:
            all_embeds.append(item["embedding"])
    return all_embeds


# ========== 文档管理 ==========
def add_document(doc_id: str, content: str, metadata: Dict = None) -> Dict:
    """切块 + 向量化 + 入库"""
    chunks = chunk_text(content)
    if not chunks:
        return {"chunks": 0}

    embeds = embed(chunks)
    ids = [f"{doc_id}__{i}" for i in range(len(chunks))]
    metadatas = [
        {"doc_id": doc_id, "chunk_idx": i, **(metadata or {})}
        for i in range(len(chunks))
    ]
    collection.add(
        ids=ids,
        embeddings=embeds,
        documents=chunks,
        metadatas=metadatas,
    )
    return {"doc_id": doc_id, "chunks": len(chunks)}


def search(query: str, top_k: int = 3) -> List[Dict]:
    """检索 Top-K 相关块"""
    query_embed = embed([query])[0]
    results = collection.query(
        query_embeddings=[query_embed],
        n_results=top_k,
    )
    return [
        {
            "content": doc,
            "metadata": meta,
            "distance": dist,  # 越小越相关
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def clear_all():
    """清空知识库"""
    global collection
    chroma_client.delete_collection("knowledge_base")
    collection = chroma_client.get_or_create_collection(name="knowledge_base")


def stats() -> Dict:
    return {
        "total_chunks": collection.count(),
    }
