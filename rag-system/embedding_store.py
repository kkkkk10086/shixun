"""
模块2：向量化与存储
使用固定的 TF-IDF 词表确保维度一致
"""

import os
import jieba
import jieba.analyse
import chromadb
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from config import CHROMA_PERSIST_DIR


# 固定词表文件路径
VOCAB_PATH = os.path.join(CHROMA_PERSIST_DIR, "tfidf_vocab.pkl")


class TfidfEmbeddingFunction:
    """固定维度的 TF-IDF Embedding"""

    def __init__(self):
        self.vectorizer = None
        self._load_or_create_vectorizer()

    def _load_or_create_vectorizer(self):
        """加载已有词表或创建新的"""
        if os.path.exists(VOCAB_PATH):
            try:
                with open(VOCAB_PATH, "rb") as f:
                    self.vectorizer = pickle.load(f)
                print("  加载已有 TF-IDF 词表")
                return
            except Exception:
                pass

        # 创建固定词表
        print("  创建 TF-IDF 固定词表...")
        self.vectorizer = TfidfVectorizer(
            max_features=512,
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2)
        )

        # 用一个通用语料初始化词表
        init_corpus = [
            "讯飞智能办公本 手写识别 语音转写 会议记录 电子墨水屏",
            "讯飞录音卡 录音 语音转文字 降噪 蓝牙",
            "讯飞翻译机 翻译 多语言 拍照翻译 双屏",
            "产品功能 使用方法 技术参数 规格配置",
            "故障排除 维修方法 常见问题 解决方案",
            "电池容量 处理器 内存 存储 屏幕尺寸 重量",
            "PDF文档 Word文档 文本解析 数据提取 向量化检索",
            "智能分块 语义检索 知识库 问答系统 大语言模型",
        ]
        self.vectorizer.fit(init_corpus)

        # 保存词表
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        with open(VOCAB_PATH, "wb") as f:
            pickle.dump(self.vectorizer, f)
        print("  词表已保存")

    def name(self):
        return "tfidf_fixed"

    def _tokenize(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        result = []
        for text in texts:
            words = list(jieba.cut(text))
            keywords = jieba.analyse.extract_tags(text, topK=10)
            weighted_words = words + keywords * 3
            result.append(" ".join(weighted_words))
        return result

    def __call__(self, input):
        tokenized = self._tokenize(input)
        tfidf_matrix = self.vectorizer.transform(tokenized)
        return tfidf_matrix.toarray().tolist()

    def embed_query(self, input):
        if isinstance(input, str):
            input = [input]
        return self(input)

    def embed_documents(self, input):
        return self(input)


def create_vectorstore(chunks: list, persist_dir: str = CHROMA_PERSIST_DIR):
    """将文本分块存入向量数据库"""
    os.makedirs(persist_dir, exist_ok=True)
    print(f"正在向量化并存储 {len(chunks)} 个文本块...")

    embedding_fn = TfidfEmbeddingFunction()
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
    embedding_fn = TfidfEmbeddingFunction()
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
