"""
模块2：向量化与存储
支持带元数据的 Chunk 存储，可按产品类型过滤检索
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 必须在任何 huggingface_hub 相关 import 之前设置镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import chromadb
from config import (
    CHROMA_PERSIST_DIR, EMBEDDING_MODEL, EMBEDDING_DIMENSION, RETRIEVAL_TOP_K,
)

BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文档："

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        if os.environ.get("HF_ENDPOINT") is None:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"  加载 Embedding 模型: {EMBEDDING_MODEL} ...")
        from sentence_transformers import SentenceTransformer
        try:
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
        except Exception as e:
            print(f"  HF镜像加载失败: {e}")
            print(f"  尝试通过 ModelScope 下载模型...")
            # 通过 modelscope 下载到本地缓存
            from modelscope import snapshot_download
            local_path = snapshot_download(
                EMBEDDING_MODEL,
                cache_dir=os.path.join(os.path.dirname(__file__), "model_cache"),
            )
            print(f"  模型已下载到: {local_path}")
            _embedding_model = SentenceTransformer(local_path, trust_remote_code=True)
        print(f"  模型加载完成")
    return _embedding_model


class ChineseEmbeddingFunction:
    """ChromaDB 兼容的中文 Embedding 函数（BGE）"""
    def __init__(self):
        self.model = get_embedding_model()

    def name(self):
        return "bge_zh_v15"

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        emb = self.model.encode(input, normalize_embeddings=True, show_progress_bar=False)
        return emb.tolist()

    def embed_query(self, input):
        if isinstance(input, str):
            input = [input]
        prefixed = [BGE_QUERY_INSTRUCTION + q for q in input]
        emb = self.model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
        return emb.tolist()

    def embed_documents(self, input):
        if isinstance(input, str):
            input = [input]
        emb = self.model.encode(input, normalize_embeddings=True, show_progress_bar=False)
        return emb.tolist()


def create_vectorstore(chunks_list, persist_dir: str = CHROMA_PERSIST_DIR):
    """
    将文本分块（可含元数据）存入 ChromaDB

    参数:
        chunks_list: 可以是:
            - list of str（纯文本，向后兼容）
            - list of Chunk 对象（含 .text 和 .metadata 属性）
            - list of dict（含 "text" 和 "metadata" 键）
    """
    import shutil
    os.makedirs(persist_dir, exist_ok=True)

    # 检测旧库并重建
    sqlite_path = os.path.join(persist_dir, "chroma.sqlite3")
    if os.path.exists(sqlite_path):
        print("  检测到旧的向量数据库，正在备份并重建...")
        backup_dir = persist_dir.rstrip("/\\") + "_bak"
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.move(persist_dir, backup_dir)
        print(f"  旧数据库已备份至: {backup_dir}")
        os.makedirs(persist_dir, exist_ok=True)

    # 统一 chunks_list 到标准格式
    documents = []
    metadatas = []
    for item in chunks_list:
        if isinstance(item, str):
            documents.append(item)
            metadatas.append({})
        elif hasattr(item, 'text') and hasattr(item, 'metadata'):
            documents.append(item.text)
            metadatas.append(item.metadata)
        elif isinstance(item, dict) and 'text' in item:
            documents.append(item['text'])
            metadatas.append(item.get('metadata', {}))
        else:
            documents.append(str(item))
            metadatas.append({})

    print(f"正在向量化并存储 {len(documents)} 个文本块（含元数据）...")

    embedding_fn = ChineseEmbeddingFunction()
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name="rag_documents",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # 分批写入
    batch_size = 128
    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        ids = [f"chunk_{j}" for j in range(i, end)]
        collection.add(
            documents=documents[i:end],
            metadatas=metadatas[i:end],
            ids=ids,
        )
        print(f"  已存储 {end}/{len(documents)} 块")

    # 统计产品分布
    product_counts = {}
    for m in metadatas:
        p = m.get("product", "")
        if p:
            product_counts[p] = product_counts.get(p, 0) + 1
    if product_counts:
        print(f"  产品分布: {dict(sorted(product_counts.items()))}")

    print(f"存储完成: {persist_dir}")

    # 写入模型版本标记
    marker_path = os.path.join(persist_dir, ".model_version")
    with open(marker_path, "w") as f:
        f.write(EMBEDDING_MODEL.replace("/", "_"))

    return collection, embedding_fn


def load_vectorstore(persist_dir: str = CHROMA_PERSIST_DIR):
    """加载已有的向量数据库"""
    embedding_fn = ChineseEmbeddingFunction()
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(name="rag_documents", embedding_function=embedding_fn)
    return collection, embedding_fn


def basic_search(collection, query: str, k: int = RETRIEVAL_TOP_K, filter_dict: dict = None):
    """
    基础相似度检索（支持产品过滤）

    参数:
        filter_dict: ChromaDB 过滤条件，如 {"product": "录音笔"}
    """
    where = filter_dict if filter_dict else None
    results = collection.query(query_texts=[query], n_results=k * 3, where=where)

    # 去重
    if results and results.get("documents"):
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
        dists = results["distances"][0] if results.get("distances") else [0] * len(docs)

        seen = set()
        unique = {"documents": [], "metadatas": [], "distances": []}
        for doc, meta, dist in zip(docs, metas, dists):
            h = hash(doc)
            if h not in seen:
                seen.add(h)
                unique["documents"].append(doc)
                unique["metadatas"].append(meta)
                unique["distances"].append(dist)

        results["documents"] = [unique["documents"]]
        results["metadatas"] = [unique["metadatas"]]
        results["distances"] = [unique["distances"]] if results.get("distances") else None

        if len(unique["documents"]) < len(docs):
            print(f"  [去重] {len(docs)}→{len(unique['documents'])}")

    return results
