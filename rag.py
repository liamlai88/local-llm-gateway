"""
RAG 模块 v3 - 懒加载依赖，启动不依赖外部包
- Vector: 语义检索（ChromaDB + 百炼 Embedding）
- BM25:   关键词检索（rank-bm25 + jieba 中文分词）
- Hybrid: RRF 倒数排名融合
"""
import os
import re
import importlib
from typing import List, Dict

# ========== 依赖懒加载 ==========
_DEPS = {
    "chromadb": None,
    "dashscope": None,
    "rank_bm25": None,
    "jieba": None,
}


def _lazy_import(name: str):
    """按需导入，缺失时返回 None"""
    if _DEPS[name] is not None:
        return _DEPS[name]
    try:
        _DEPS[name] = importlib.import_module(name)
        return _DEPS[name]
    except ImportError:
        return None


def check_deps() -> Dict[str, bool]:
    """检查所有依赖状态"""
    return {name: _lazy_import(name) is not None for name in _DEPS}


def require(*names):
    """断言依赖已安装，否则抛友好错误"""
    missing = [n for n in names if _lazy_import(n) is None]
    if missing:
        cmd = "pip install " + " ".join(missing).replace("rank_bm25", "rank-bm25")
        raise RuntimeError(
            f"缺少依赖: {missing}\n"
            f"请运行: {cmd}\n"
            f"装完后重启 Gateway (Ctrl+C 后重新 uvicorn)"
        )


# ========== 全局状态（延迟初始化）==========
_chroma_client = None
_collection = None
_bm25_index = None
_bm25_chunks = []
_bm25_ids = []
_bm25_metadatas = []


def _get_collection():
    """延迟初始化 ChromaDB"""
    global _chroma_client, _collection
    if _collection is None:
        require("chromadb")
        chromadb = _lazy_import("chromadb")
        _chroma_client = chromadb.PersistentClient(path="./chroma_data")
        _collection = _chroma_client.get_or_create_collection(name="knowledge_base")
    return _collection


# ========== 中文分词 ==========
def tokenize(text: str) -> List[str]:
    require("jieba")
    jieba = _lazy_import("jieba")
    tokens = jieba.lcut(text.lower())
    return [t for t in tokens if len(t.strip()) >= 1 and not re.match(r"^[\W_]+$", t)]


# ========== Chunking ==========
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) <= chunk_size:
            buf += "\n" + p if buf else p
        else:
            if buf:
                chunks.append(buf)
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

    if overlap > 0 and len(chunks) > 1:
        out = [chunks[0]]
        for i in range(1, len(chunks)):
            out.append(chunks[i - 1][-overlap:] + " " + chunks[i])
        return out
    return chunks


# ========== Embedding ==========
def embed(texts: List[str]) -> List[List[float]]:
    require("dashscope")
    dashscope = _lazy_import("dashscope")
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("DASHSCOPE_API_KEY 未设置（启动时需要 export 或 prefix 到 uvicorn）")
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

    all_embeds = []
    for i in range(0, len(texts), 25):
        resp = dashscope.TextEmbedding.call(model="text-embedding-v2", input=texts[i:i + 25])
        if resp.status_code != 200:
            raise RuntimeError(f"Embedding failed: {resp.message}")
        for item in resp.output["embeddings"]:
            all_embeds.append(item["embedding"])
    return all_embeds


# ========== BM25 ==========
def _rebuild_bm25():
    global _bm25_index, _bm25_chunks, _bm25_ids, _bm25_metadatas
    require("rank_bm25")
    BM25Okapi = _lazy_import("rank_bm25").BM25Okapi
    coll = _get_collection()
    data = coll.get()
    _bm25_chunks = data["documents"]
    _bm25_ids = data["ids"]
    _bm25_metadatas = data["metadatas"]
    if _bm25_chunks:
        _bm25_index = BM25Okapi([tokenize(c) for c in _bm25_chunks])
    else:
        _bm25_index = None


# ========== 文档管理 ==========
def add_document(doc_id: str, content: str, metadata: Dict = None) -> Dict:
    chunks = chunk_text(content)
    if not chunks:
        return {"chunks": 0}
    embeds = embed(chunks)
    ids = [f"{doc_id}__{i}" for i in range(len(chunks))]
    metadatas = [{"doc_id": doc_id, "chunk_idx": i, **(metadata or {})} for i in range(len(chunks))]
    _get_collection().add(ids=ids, embeddings=embeds, documents=chunks, metadatas=metadatas)
    _rebuild_bm25()
    return {"doc_id": doc_id, "chunks": len(chunks)}


# ========== 三种检索 ==========
def search_vector(query: str, top_k: int = 5) -> List[Dict]:
    qe = embed([query])[0]
    res = _get_collection().query(query_embeddings=[qe], n_results=top_k)
    return [
        {"id": id_, "content": doc, "metadata": meta, "score": 1.0 / (1.0 + dist), "method": "vector"}
        for id_, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


def search_bm25(query: str, top_k: int = 5) -> List[Dict]:
    if _bm25_index is None or not _bm25_chunks:
        _rebuild_bm25()
    if _bm25_index is None:
        return []
    tokens = tokenize(query)
    scores = _bm25_index.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {
            "id": _bm25_ids[i], "content": _bm25_chunks[i], "metadata": _bm25_metadatas[i],
            "score": float(scores[i]), "method": "bm25",
        }
        for i in top_idx if scores[i] > 0
    ]


def search_hybrid(query: str, top_k: int = 5, k: int = 60) -> List[Dict]:
    vec = search_vector(query, top_k=top_k * 2)
    bm = search_bm25(query, top_k=top_k * 2)
    rrf, chunks = {}, {}
    for rank, r in enumerate(vec):
        rrf[r["id"]] = rrf.get(r["id"], 0) + 1 / (k + rank + 1)
        chunks[r["id"]] = r
    for rank, r in enumerate(bm):
        rrf[r["id"]] = rrf.get(r["id"], 0) + 1 / (k + rank + 1)
        chunks[r["id"]] = r
    sorted_ids = sorted(rrf.keys(), key=lambda x: rrf[x], reverse=True)[:top_k]
    return [
        {
            "id": id_, "content": chunks[id_]["content"], "metadata": chunks[id_]["metadata"],
            "score": rrf[id_], "method": "hybrid",
        }
        for id_ in sorted_ids
    ]


def rerank(query: str, candidates: List[Dict], top_k: int = 3) -> List[Dict]:
    """
    用百炼 gte-rerank 对候选文档精排
    candidates: 召回阶段的 Top-N 候选 (来自 vector/bm25/hybrid)
    返回: 重排后的 Top-K
    """
    if not candidates:
        return []
    require("dashscope")
    dashscope = _lazy_import("dashscope")
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("DASHSCOPE_API_KEY 未设置")
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

    docs = [c["content"] for c in candidates]
    resp = dashscope.TextReRank.call(
        model="gte-rerank",
        query=query,
        documents=docs,
        top_n=top_k,
        return_documents=False,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Rerank failed: {resp.message}")

    # gte-rerank 返回 [{index: 0, relevance_score: 0.98}, ...]
    results = []
    for item in resp.output["results"]:
        idx = item["index"]
        cand = candidates[idx].copy()
        cand["rerank_score"] = item["relevance_score"]
        cand["original_score"] = cand.get("score", 0)
        cand["score"] = item["relevance_score"]
        cand["method"] = cand.get("method", "?") + "+rerank"
        results.append(cand)
    return results


def search(query: str, top_k: int = 3, mode: str = "hybrid", use_rerank: bool = False) -> List[Dict]:
    """
    统一检索入口
    - mode: vector / bm25 / hybrid
    - use_rerank: True 则召回 Top-10 后精排到 top_k
    """
    if use_rerank:
        # 召回更多候选给 Rerank
        recall_k = max(top_k * 3, 10)
        if mode == "vector":
            cands = search_vector(query, recall_k)
        elif mode == "bm25":
            cands = search_bm25(query, recall_k)
        else:
            cands = search_hybrid(query, recall_k)
        return rerank(query, cands, top_k=top_k)

    if mode == "vector":
        return search_vector(query, top_k)
    elif mode == "bm25":
        return search_bm25(query, top_k)
    return search_hybrid(query, top_k)


def clear_all():
    """彻底清空：删 collection + 删所有 ID + 重置内存"""
    global _collection, _chroma_client, _bm25_index, _bm25_chunks, _bm25_ids, _bm25_metadatas
    require("chromadb")
    # 先把当前 collection 里所有 ID 删掉（兜底）
    try:
        coll = _get_collection()
        all_ids = coll.get()["ids"]
        if all_ids:
            coll.delete(ids=all_ids)
    except Exception as e:
        print(f"清理 collection 内容时报错（忽略）: {e}")
    # 再删整个 collection
    try:
        if _chroma_client:
            _chroma_client.delete_collection("knowledge_base")
    except Exception as e:
        print(f"删除 collection 时报错（忽略）: {e}")
    _collection = None
    _bm25_index = None
    _bm25_chunks = []
    _bm25_ids = []
    _bm25_metadatas = []


def stats() -> Dict:
    if _collection is None:
        return {"total_chunks": 0, "bm25_indexed": 0, "initialized": False}
    return {
        "total_chunks": _collection.count(),
        "bm25_indexed": len(_bm25_chunks),
        "initialized": True,
    }
