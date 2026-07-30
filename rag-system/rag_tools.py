"""
模块：LangChain Tool 封装 + 高级检索
三路检索（关键词 + 向量 + BM25）→ RRF 融合 → Cross-Encoder 重排
"""

import os
import sys
import math
import jieba
import jieba.analyse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 必须在所有 huggingface_hub / sentence_transformers 相关 import 之前设置镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_core.tools import tool
from config import (
    CHROMA_PERSIST_DIR,
    RETRIEVAL_TOP_K,
    RERANK_TOP_K,
    RRF_K,
    CROSS_ENCODER_MODEL,
    CROSS_ENCODER_USE_GPU,
    EMBEDDING_MODEL,
)

# 全局变量，启动时初始化
_collection = None
_bm25_index = None
_bm25_docs = None
_reranker = None


# ============================================================
# BM25 实现
# ============================================================

class BM25:
    """BM25 算法（jieba 分词，k1/b 参数可调）"""

    def __init__(self, documents: list, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.doc_count = len(documents)

        self.doc_lengths = []
        self.avg_doc_length = 0
        self.tf = []
        self.df = {}

        for doc in documents:
            tokens = list(jieba.cut(doc))
            self.doc_lengths.append(len(tokens))

            tf_dict = {}
            for token in tokens:
                tf_dict[token] = tf_dict.get(token, 0) + 1
            self.tf.append(tf_dict)

            for token in set(tokens):
                self.df[token] = self.df.get(token, 0) + 1

        self.avg_doc_length = sum(self.doc_lengths) / self.doc_count if self.doc_count > 0 else 0

    def search(self, query: str, top_k: int = 10) -> list:
        """BM25 搜索，返回 [(score, doc_index), ...]"""
        query_tokens = list(jieba.cut(query))
        scores = []

        for i, doc_tokens_count in enumerate(self.doc_lengths):
            score = 0
            for token in query_tokens:
                if token not in self.tf[i]:
                    continue
                tf = self.tf[i][token]
                df = self.df.get(token, 0)
                idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_tokens_count / self.avg_doc_length)
                score += idf * numerator / denominator
            if score > 0:
                scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]


# ============================================================
# RRF（Reciprocal Rank Fusion）融合算法
# ============================================================

def reciprocal_rank_fusion(
    ranked_lists: list,
    k: int = RRF_K,
    top_k: int = 30,
) -> list:
    """
    RRF 融合多路检索结果

    参数:
        ranked_lists: 多路检索结果列表，每路为 [doc_id, ...] 按相关性降序排列
        k: RRF 常数（默认60），越大越平滑
        top_k: 返回 top-N 个融合结果

    返回: [(rrf_score, doc_id), ...] 降序排列
    """
    rrf_scores = {}
    for rank_list in ranked_lists:
        for rank, doc_id in enumerate(rank_list):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ============================================================
# Cross-Encoder 重排模型
# ============================================================

class CrossEncoderReranker:
    """
    智能重排器

    优先使用真正的 Cross-Encoder 模型（最高精度）
    如果模型不存在（首次运行/网络不可用），自动降级为基于 BGE 的语义重排
    """

    def __init__(self):
        self.model = None
        self.mode = "none"  # "cross_encoder" | "bge_fallback"
        self.model_name = CROSS_ENCODER_MODEL

    @property
    def mode_display(self) -> str:
        """返回可读的重排方式名称"""
        return {"none": "未初始化", "cross_encoder": "Cross-Encoder重排",
                "bge_fallback": "BGE语义重排"}.get(self.mode, "未知")

    def _try_load_cross_encoder(self) -> bool:
        """尝试加载真正的 Cross-Encoder 模型"""
        if self.model is not None:
            return True

        if os.environ.get("HF_ENDPOINT") is None:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        try:
            print(f"  [重排器] 尝试加载 Cross-Encoder: {self.model_name} ...")
            from sentence_transformers import CrossEncoder
            import torch
            device = "cuda" if (CROSS_ENCODER_USE_GPU and torch.cuda.is_available()) else "cpu"
            self.model = CrossEncoder(
                self.model_name,
                device=device,
                trust_remote_code=True,
            )
            self.mode = "cross_encoder"
            print(f"  [重排器] Cross-Encoder 加载成功 (device={device})")
            return True
        except Exception as e:
            print(f"  [重排器] Cross-Encoder 加载失败: {type(e).__name__}: {e}")
            print(f"  [重排器] 尝试通过 ModelScope 下载...")
            try:
                from modelscope import snapshot_download
                import os as _os
                local_path = snapshot_download(
                    self.model_name,
                    cache_dir=_os.path.join(_os.path.dirname(__file__), "model_cache"),
                )
                print(f"  [重排器] 模型已下载到: {local_path}")
                import torch
                device = "cuda" if (CROSS_ENCODER_USE_GPU and torch.cuda.is_available()) else "cpu"
                self.model = CrossEncoder(
                    local_path,
                    device=device,
                    trust_remote_code=True,
                )
                self.mode = "cross_encoder"
                print(f"  [重排器] Cross-Encoder 加载成功 (device={device})")
                return True
            except Exception as e2:
                print(f"  [重排器] ModelScope 也失败: {type(e2).__name__}")
                print(f"  [重排器] 降级为 BGE 语义重排模式")
                self.model = None
                return False

    def _init_bge_fallback(self):
        """降级方案：使用 BGE 模型做语义重排"""
        from embedding_store import get_embedding_model, BGE_QUERY_INSTRUCTION
        self.model = {
            "encoder": get_embedding_model(),
            "instruction": BGE_QUERY_INSTRUCTION,
        }
        self.mode = "bge_fallback"

    def rerank(self, query: str, documents: list, top_k: int = RERANK_TOP_K) -> list:
        """
        对候选文档进行重排

        参数:
            query: 用户查询
            documents: 候选文档列表
            top_k: 返回 top-N 个结果

        返回: [(score, document), ...] 降序排列
        """
        if not documents:
            return []

        if self.model is None:
            # 优先尝试 Cross-Encoder
            if not self._try_load_cross_encoder():
                self._init_bge_fallback()

        if self.mode == "cross_encoder":
            return self._rerank_ce(query, documents, top_k)
        else:
            return self._rerank_bge(query, documents, top_k)

    def _rerank_ce(self, query: str, documents: list, top_k: int) -> list:
        """使用 Cross-Encoder 重排"""
        pairs = [[query, doc] for doc in documents]

        try:
            scores = self.model.predict(pairs, show_progress_bar=False)
        except Exception as e:
            print(f"  [重排器] CE 预测失败: {e}")
            return [(1.0, doc) for doc in documents[:top_k]]

        if isinstance(scores, (float, int)):
            scores = [scores]
        else:
            scores = scores.tolist() if hasattr(scores, 'tolist') else list(scores)

        ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        print(f"  [CE重排] {len(documents)} 候选 → Top-{top_k}")
        return ranked[:top_k]

    def _rerank_bge(self, query: str, documents: list, top_k: int) -> list:
        """
        降级方案：使用 BGE 编码后计算余弦相似度重排
        虽然不是真正的 Cross-Encoder，但 BGE 的语义质量远高于简单关键词匹配
        """
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        encoder = self.model["encoder"]
        instruction = self.model["instruction"]

        # 编码查询（加 BGE 指令前缀）
        query_emb = encoder.encode(
            [instruction + query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # 编码文档（不加前缀）
        doc_embs = encoder.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # 计算余弦相似度
        scores = cosine_similarity(query_emb, doc_embs)[0]

        ranked = sorted(
            zip(scores.tolist(), documents),
            key=lambda x: x[0],
            reverse=True,
        )

        # 打印分数范围
        scores_only = [s for s, _ in ranked]
        if scores_only:
            print(f"  [BGE重排] score范围: {min(scores_only):.4f} ~ {max(scores_only):.4f} | {len(documents)}候选 → Top-{top_k}")

        return ranked[:top_k]


def get_reranker():
    """获取全局 Cross-Encoder 重排器"""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


# ============================================================
# BM25 索引管理
# ============================================================

def build_bm25_index(documents: list):
    """构建 BM25 索引"""
    global _bm25_index, _bm25_docs
    _bm25_docs = documents
    _bm25_index = BM25(documents)
    print(f"  [BM25] 索引构建完成，共 {len(documents)} 个文档")


def set_collection(collection):
    """设置全局向量数据库，并构建 BM25 索引"""
    global _collection
    _collection = collection

    if collection is not None:
        try:
            total = collection.count()
            all_results = collection.get(limit=min(total, 1000))
            all_docs = all_results["documents"] if all_results["documents"] else []
            if all_docs:
                build_bm25_index(all_docs)
        except Exception as e:
            print(f"  [BM25] 索引构建失败: {e}")


# ============================================================
# 产品关键词映射
# ============================================================

PRODUCT_MAP = {
    "学习机": ["学习机", "S90", "S90 Pro"],
    "英语宝": ["英语宝", "EBOX", "EBOX Pro"],
    "录音笔": ["录音笔", "录音设备", "SR702", "SR502", "SR302", "S6 Plus", "S6系列", "S8离线版", "H1 Pro", "Magic", "Pokee"],
    "翻译机": ["翻译机", "双屏翻译机"],
    "办公本": ["办公本", "X2", "X2LAMY", "墨水屏"],
    "词典笔": ["词典笔", "X8", "扫描查词"],
    "录音卡": ["录音卡", "AIR2611"],
    "键盘": ["键盘", "T8"],
    "鼠标": ["鼠标", "AM50"],
}

PRODUCT_NAMES = {
    "录音笔": "讯飞AI录音笔（含11个型号）",
    "翻译机": "讯飞翻译机（双屏翻译机）",
    "办公本": "讯飞智能办公本X2/X2LAMY",
    "词典笔": "科大讯飞AI词典笔X8 Pro",
    "学习机": "科大讯飞AI学习机S90 Pro",
    "录音卡": "讯飞AI录音卡AIR2611",
    "键盘": "科大讯飞AI机械键盘T8星火版",
    "鼠标": "科大讯飞AI鼠标AM50Pro",
    "英语宝": "科大讯飞AI英语宝EBOX Pro二代",
}

LIST_KEYWORDS = ["几款", "有哪些", "什么产品", "产品列表", "全部", "所有",
                 "多少", "总共", "一共", "产品", "几个", "型号"]

SUMMARY_KEYWORDS = ["几款", "多少", "有哪些", "全部", "所有", "总共",
                    "列表", "对比", "区别", "价格", "售价"]


def detect_product(query: str):
    """检测查询涉及的产品，返回 [(product_name, keywords), ...]"""
    for product, keywords in PRODUCT_MAP.items():
        if product in query:
            return [(product, keywords)]
        for kw in keywords:
            if len(kw) >= 2 and kw in query:
                return [(product, keywords)]
    return []


def is_list_query(query: str) -> bool:
    """判断是否为汇总列表类查询"""
    return any(kw in query for kw in LIST_KEYWORDS)


def is_summary_query(query: str) -> bool:
    """判断是否为汇总类问题"""
    return any(kw in query for kw in SUMMARY_KEYWORDS)


# ============================================================
# 核心检索函数（供内部和工具调用）
# ============================================================

def three_way_search(query: str, all_db_docs: list, top_k: int = RETRIEVAL_TOP_K):
    """
    三路检索：关键词 + 向量 + BM25，返回每路的结果列表

    返回: {
        "keyword_ids": [doc_id, ...],
        "vector_ids": [doc_id, ...],
        "bm25_ids": [doc_id, ...],
        "id_to_doc": {id: doc, ...},
        "doc_to_id": {doc: id, ...},
    }
    """
    # 构建 doc ↔ id 映射
    id_to_doc = {i: doc for i, doc in enumerate(all_db_docs)}
    doc_to_id = {doc: i for i, doc in id_to_doc.items()}

    # ---- 1. 关键词检索 ----
    search_terms = [query]
    try:
        search_terms.extend(jieba.analyse.extract_tags(query, topK=10))
    except Exception:
        pass

    keyword_scored = []
    seen = set()
    for doc in all_db_docs:
        score = sum(1 for term in search_terms if term in doc)
        if score >= 1:
            h = hash(doc)
            if h not in seen:
                seen.add(h)
                keyword_scored.append((score, doc))
    keyword_scored.sort(key=lambda x: x[0], reverse=True)
    keyword_ids = [doc_to_id[d] for _, d in keyword_scored[:top_k]]

    # ---- 2. 向量检索 ----
    vector_ids = []
    try:
        vector_results = _collection.query(query_texts=[query], n_results=top_k)
        if vector_results and vector_results["documents"]:
            seen_docs = set()
            for doc in vector_results["documents"][0]:
                if doc in doc_to_id:
                    h = hash(doc)
                    if h not in seen_docs:
                        seen_docs.add(h)
                        vector_ids.append(doc_to_id[doc])
    except Exception as e:
        print(f"  [三路检索] 向量检索失败: {e}")

    # ---- 3. BM25 检索 ----
    bm25_ids = []
    try:
        if _bm25_index is not None:
            bm25_results = _bm25_index.search(query, top_k=top_k)
            bm25_ids = [idx for _, idx in bm25_results]
    except Exception as e:
        print(f"  [三路检索] BM25检索失败: {e}")

    return {
        "keyword_ids": keyword_ids,
        "vector_ids": vector_ids,
        "bm25_ids": bm25_ids,
        "id_to_doc": id_to_doc,
        "doc_to_id": doc_to_id,
    }


def rrf_retrieve(query: str, all_db_docs: list, top_k: int = 30) -> list:
    """
    三路检索 → RRF 融合 → 返回文档列表（已去重，按 RRF 分数排序）

    返回: [(rrf_score, doc_text), ...]
    """
    result = three_way_search(query, all_db_docs, top_k=RETRIEVAL_TOP_K)
    id_to_doc = result["id_to_doc"]

    # RRF 融合三路排名
    ranked_lists = [result["keyword_ids"], result["vector_ids"], result["bm25_ids"]]
    fusion_results = reciprocal_rank_fusion(ranked_lists, k=RRF_K, top_k=top_k)

    # 转换为 (score, doc) 格式
    final = []
    for rrf_score, doc_id in fusion_results:
        if doc_id in id_to_doc:
            final.append((rrf_score, id_to_doc[doc_id]))

    return final


def _filter_short_chunks(documents: list, min_length: int = 50) -> list:
    """过滤掉过短的标题块，它们对重排和生成有害无益"""
    filtered = [d for d in documents if len(d.strip()) >= min_length]
    if not filtered:
        return documents  # 防止全被过滤掉
    return filtered


def rerank_documents(query: str, documents: list, top_k: int = RERANK_TOP_K) -> list:
    """
    Cross-Encoder 重排候选文档

    返回: [(score, doc_text), ...] 按相关性降序
    """
    if is_summary_query(query):
        # 汇总类问题不需要重排，直接返回全部
        print(f"  [重排] 汇总类查询，跳过重排，保留全部 {len(documents)} 个")
        return [(1.0, d) for d in documents]

    reranker = get_reranker()
    return reranker.rerank(query, documents, top_k=top_k)


# ============================================================
# LangChain Tools
# ============================================================

@tool
def search_knowledge_base(query: str) -> str:
    """从讯飞智能硬件产品知识库中检索相关信息。输入用户的问题，返回相关的文档内容。"""
    global _collection
    if _collection is None:
        return "知识库未初始化"

    try:
        # 获取所有文档
        total = _collection.count()
        all_results = _collection.get(limit=min(total, 500))
        all_db_docs = all_results["documents"] if all_results["documents"] else []

        # ---- 列表类查询：直接返回产品名称 ----
        matched_products = detect_product(query)
        if is_list_query(query) and not matched_products:
            result = "\n".join([f"- {name}" for name in PRODUCT_NAMES.values()])
            print(f"  [知识库] 列表类查询，返回产品名称列表")
            return result

        # ---- 特定产品查询：三路检索(RRF) + 产品补充 ----
        if matched_products:
            # 使用三路检索（向量+关键词+BM25），替代纯关键词匹配
            rrf_results = rrf_retrieve(query, all_db_docs, top_k=50)
            merged = {hash(doc): doc for _, doc in rrf_results}

            # 补充该产品的其他文档（三路检索可能漏掉的产品相关文档）
            for product, keywords in matched_products:
                for doc in all_db_docs:
                    if any(kw in doc for kw in keywords):
                        h = hash(doc)
                        if h not in merged:
                            merged[h] = doc

            merged_docs = list(merged.values())
            print(f"  [知识库] 产品 '{','.join(p for p,_ in matched_products)}' 三路检索+产品补充 → {len(merged_docs)} 个候选")

            # 过滤短标题块
            merged_docs = _filter_short_chunks(merged_docs)

            # Cross-Encoder 重排
            reranked = rerank_documents(query, merged_docs, top_k=RERANK_TOP_K * 2)
            final_docs = [doc for _, doc in reranked]

            return "\n".join([f"[文档{i+1}] {d}" for i, d in enumerate(final_docs[:50])])

        # ---- 普通查询：RRF 三路融合 + Cross-Encoder 重排 ----
        print(f"  [知识库] 三路检索 → RRF融合 → Cross-Encoder重排")

        # RRF 融合
        rrf_results = rrf_retrieve(query, all_db_docs, top_k=50)
        rrf_docs = [doc for _, doc in rrf_results]

        if not rrf_docs:
            # 兜底
            rrf_docs = all_db_docs[:20]

        print(f"  [知识库] RRF 融合后 {len(rrf_docs)} 个候选")

        # 过滤短标题块
        rrf_docs = _filter_short_chunks(rrf_docs)

        # Cross-Encoder 重排
        reranked = rerank_documents(query, rrf_docs, top_k=RERANK_TOP_K * 2)
        final_docs = [doc for _, doc in reranked]

        # 最终去重（按内容）
        seen = set()
        unique_docs = []
        for doc in final_docs:
            h = hash(doc)
            if h not in seen:
                seen.add(h)
                unique_docs.append(doc)
        if len(unique_docs) < len(final_docs):
            print(f"  [知识库] 最终去重移除 {len(final_docs) - len(unique_docs)} 个重复")

        # 如果重排后文档太少，补充一些（也要去重）
        if len(unique_docs) < 5:
            for doc in rrf_docs:
                h = hash(doc)
                if h not in seen:
                    seen.add(h)
                    unique_docs.append(doc)
                    if len(unique_docs) >= 5:
                        break

        print(f"  [知识库] RRF({len(rrf_docs)}) → CE重排({len(unique_docs)}) → 返回")
        return "\n".join([f"[文档{i+1}] {d}" for i, d in enumerate(unique_docs)])

    except Exception as e:
        return f"检索出错: {e}"


@tool
def list_all_products() -> str:
    """列出知识库中所有讯飞产品及其基本信息。输入为空，返回完整产品清单。"""
    global _collection
    if _collection is None:
        return "知识库未初始化"

    try:
        total = _collection.count()
        all_results = _collection.get(limit=min(total, 500))
        all_docs = all_results["documents"] if all_results["documents"] else []

        products = {}
        for doc in all_docs:
            for product_name, keywords in PRODUCT_MAP.items():
                if any(kw in doc for kw in keywords):
                    if product_name not in products:
                        products[product_name] = []
                    products[product_name].append(doc)
                    break

        output = "讯飞产品清单：\n"
        for i, (name, docs) in enumerate(products.items(), 1):
            all_info = " ".join(docs)[:500]
            output += f"{i}. {name}：{all_info}\n"

        return output
    except Exception as e:
        return f"获取产品列表出错: {e}"


@tool
def get_product_specs(product_name: str) -> str:
    """查询特定产品的详细规格参数。输入产品名称，返回该产品的规格信息。"""
    global _collection
    if _collection is None:
        return "知识库未初始化"

    try:
        results = _collection.query(query_texts=[product_name], n_results=10)
        docs = results["documents"][0] if results["documents"] else []

        specs = [d for d in docs if any(kw in d for kw in ["参数", "规格", "配置", "屏幕", "电池", "内存"])]
        if not specs:
            specs = docs

        return "\n".join([f"[{i+1}] {s[:300]}" for i, s in enumerate(specs[:5])])
    except Exception as e:
        return f"查询产品规格出错: {e}"


# 所有工具列表
RAG_TOOLS = [search_knowledge_base, list_all_products, get_product_specs]
