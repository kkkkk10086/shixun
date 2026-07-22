"""
模块2：向量化与存储
使用 sentence-transformers 中文 embedding 模型（离线模式）
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import chromadb
from config import CHROMA_PERSIST_DIR


# 全局 embedding 模型
_embedding_model = None


def get_embedding_model():
    """获取中文 embedding 模型"""
    global _embedding_model
    if _embedding_model is None:
        print("  加载中文 Embedding 模型...")
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")
        print("  模型加载完成")
    return _embedding_model


class ChineseEmbeddingFunction:
    """ChromaDB 兼容的中文 Embedding 函数"""

    def __init__(self):
        self.model = get_embedding_model()

    def name(self):
        return "chinese_embedding"

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        embeddings = self.model.encode(input)
        return embeddings.tolist()

    def embed_query(self, input):
        if isinstance(input, str):
            input = [input]
        embeddings = self.model.encode(input)
        return embeddings.tolist()

    def embed_documents(self, input):
        return self(input)


def create_vectorstore(chunks: list, persist_dir: str = CHROMA_PERSIST_DIR):
    """将文本分块存入向量数据库"""
    os.makedirs(persist_dir, exist_ok=True)
    print(f"正在向量化并存储 {len(chunks)} 个文本块...")

    embedding_fn = ChineseEmbeddingFunction()

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name="rag_documents",
        embedding_function=embedding_fn
    )

    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        ids = [f"chunk_{j}" for j in range(i, i + len(batch))]
        collection.add(documents=batch, ids=ids)
        print(f"  已存储 {min(i + batch_size, len(chunks))}/{len(chunks)} 块")

    print(f"存储完成: {persist_dir}")
    return collection, embedding_fn


def load_vectorstore(persist_dir: str = CHROMA_PERSIST_DIR):
    """加载已有的向量数据库"""
    embedding_fn = ChineseEmbeddingFunction()
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(
        name="rag_documents",
        embedding_function=embedding_fn
    )
    return collection, embedding_fn


def basic_search(collection, query: str, k: int = 5):
    """基础相似度检索"""
    results = collection.query(query_texts=[query], n_results=k)
    return results
