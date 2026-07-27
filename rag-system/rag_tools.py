"""
模块：LangChain Tool 封装
将 RAG 检索功能封装为标准的 LangChain Tool
支持三路检索：关键词匹配 + 向量检索 + BM25
"""

import os
import sys
import math
import jieba

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from config import CHROMA_PERSIST_DIR


# 全局变量，启动时初始化
_collection = None
_bm25_index = None
_bm25_docs = None


class BM25:
    """简易 BM25 实现"""

    def __init__(self, documents: list, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.doc_count = len(documents)

        # 分词并计算
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.tf = []  # 词频
        self.df = {}  # 文档频率

        for doc in documents:
            tokens = list(jieba.cut(doc))
            self.doc_lengths.append(len(tokens))

            # 计算词频
            tf_dict = {}
            for token in tokens:
                tf_dict[token] = tf_dict.get(token, 0) + 1
            self.tf.append(tf_dict)

            # 计算文档频率
            for token in set(tokens):
                self.df[token] = self.df.get(token, 0) + 1

        self.avg_doc_length = sum(self.doc_lengths) / self.doc_count if self.doc_count > 0 else 0

    def search(self, query: str, top_k: int = 10) -> list:
        """BM25 搜索，返回 (score, doc_index) 列表"""
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

    # 构建 BM25 索引
    if collection is not None:
        try:
            total = collection.count()
            all_results = collection.get(limit=min(total, 500))
            all_docs = all_results["documents"] if all_results["documents"] else []
            if all_docs:
                build_bm25_index(all_docs)
        except Exception as e:
            print(f"  [BM25] 索引构建失败: {e}")


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

        # 提取搜索关键词
        search_terms = [query]
        try:
            import jieba.analyse
            search_terms.extend(jieba.analyse.extract_tags(query, topK=10))
        except Exception:
            pass

        # 产品关键词映射（按优先级排序，避免误匹配）
        product_map = {
            "学习机": ["学习机", "S90", "S90 Pro"],
            "英语宝": ["英语宝", "EBOX", "EBOX Pro"],
            "录音笔": ["录音笔", "录音设备", "SR702", "SR502", "SR302", "S6 Plus", "S6系列", "S8离线版", "H1 Pro", "Magic", "Pokee"],
            "翻译机": ["翻译机", "双屏翻译机"],
            "办公本": ["办公本", "X2", "X2LAMY", "墨水屏"],
            "词典笔": ["词典笔", "X8", "扫描查词"],
            "录音卡": ["录音卡", "AIR2611"],
            "键盘": ["键盘", "T8"],
            "鼠标": ["鼠标", "AM50"]
        }

        # 检测查询涉及哪些产品（精确匹配，避免"学习"同时匹配学习机和英语宝）
        matched_products = []
        for product, keywords in product_map.items():
            # 精确匹配：查询必须包含完整的产品名或完整关键词
            if product in query:
                matched_products.append((product, keywords))
                break  # 只取最精确的那个产品
            for kw in keywords:
                if len(kw) >= 2 and kw in query:  # 关键词至少2个字符
                    matched_products.append((product, keywords))
                    break
            if matched_products:
                break

        # 汇总类查询：直接返回产品名称列表
        list_keywords = ["几款", "有哪些", "什么产品", "产品列表", "全部", "所有",
                         "多少", "总共", "一共", "产品", "几个", "型号"]
        if any(kw in query for kw in list_keywords) and not matched_products:
            product_names = {
                "录音笔": "讯飞AI录音笔（含11个型号）",
                "翻译机": "讯飞翻译机（双屏翻译机）",
                "办公本": "讯飞智能办公本X2/X2LAMY",
                "词典笔": "科大讯飞AI词典笔X8 Pro",
                "学习机": "科大讯飞AI学习机S90 Pro",
                "录音卡": "讯飞AI录音卡AIR2611",
                "键盘": "科大讯飞AI机械键盘T8星火版",
                "鼠标": "科大讯飞AI鼠标AM50Pro",
                "英语宝": "科大讯飞AI英语宝EBOX Pro二代"
            }
            result = "\n".join([f"- {name}" for name in product_names.values()])
            print(f"  [知识库] 列表类查询，返回产品名称列表")
            return result

        # 先用关键词匹配筛选相关文档
        scored = []
        seen = set()
        for doc in all_db_docs:
            score = sum(1 for term in search_terms if term in doc)
            if score >= 1:
                h = hash(doc)
                if h not in seen:
                    seen.add(h)
                    scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        # 有特定产品：关键词匹配 + 产品文档补充
        if matched_products:
            # 关键词匹配的文档优先
            keyword_docs = [d for _, d in scored]
            # 补充该产品的其他文档（确保覆盖）
            product_only = []
            for product, keywords in matched_products:
                for doc in all_db_docs:
                    if any(kw in doc for kw in keywords):
                        h = hash(doc)
                        if h not in seen:
                            seen.add(h)
                            product_only.append(doc)
            # 合并：关键词匹配的 + 产品相关的
            combined = keyword_docs + product_only
            # 去重
            final = []
            seen2 = set()
            for d in combined:
                h = hash(d)
                if h not in seen2:
                    seen2.add(h)
                    final.append(d)

            print(f"  [知识库] 产品 '{','.join(p for p,_ in matched_products)}' 检索到 {len(final)} 个文档")
            return "\n".join([f"[文档{i+1}] {d}" for i, d in enumerate(final[:30])])

            return "\n".join([f"[文档{i+1}] {d}" for i, d in enumerate(result_docs[:30])])

        # 普通查询：关键词匹配 + 向量检索 + BM25 三路合并
        keyword_docs = [d for _, d in scored[:10]]

        # 向量检索（语义相似性）
        vector_docs = []
        try:
            vector_results = _collection.query(query_texts=[query], n_results=5)
            if vector_results and vector_results["documents"]:
                vector_docs = vector_results["documents"][0]
        except Exception as e:
            print(f"  [知识库] 向量检索失败: {e}")

        # BM25 检索（词频+逆文档频率）
        bm25_docs = []
        try:
            if _bm25_index is not None:
                bm25_results = _bm25_index.search(query, top_k=5)
                bm25_docs = [_bm25_docs[idx] for score, idx in bm25_results]
        except Exception as e:
            print(f"  [知识库] BM25检索失败: {e}")

        # 合并三路结果（去重）
        combined = []
        seen = set()
        for d in keyword_docs + vector_docs + bm25_docs:
            h = hash(d)
            if h not in seen:
                seen.add(h)
                combined.append(d)

        # 限制最多15个文档，避免淹没LLM
        combined = combined[:15]

        # 兜底
        if len(combined) < 3:
            combined = all_db_docs[:15]

        print(f"  [知识库] 关键词({len(keyword_docs)})+向量({len(vector_docs)})+BM25({len(bm25_docs)}) → 合并 {len(combined)} 个文档")
        return "\n".join([f"[文档{i+1}] {d}" for i, d in enumerate(combined)])
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

        # 提取产品信息（收集每个产品的所有文档块）
        products = {}
        product_keywords = {
            "录音笔": ["录音笔", "录音设备", "SR", "H1", "S6", "S8", "Magic", "Pokee"],
            "翻译机": ["翻译机", "翻译设备", "双屏"],
            "办公本": ["办公本", "X2", "X2LAMY"],
            "录音卡": ["录音卡"],
            "键盘": ["键盘", "T8"],
            "鼠标": ["鼠标", "AM50"],
            "词典笔": ["词典笔", "X8"],
            "学习机": ["学习机", "S90"],
            "英语宝": ["英语宝", "EBOX"]
        }

        for doc in all_docs:
            for product_name, keywords in product_keywords.items():
                if any(kw in doc for kw in keywords):
                    if product_name not in products:
                        products[product_name] = []
                    products[product_name].append(doc)
                    break

        # 构建输出（包含每个产品的主要信息）
        output = "讯飞产品清单：\n"
        for i, (name, docs) in enumerate(products.items(), 1):
            # 合并所有文档的关键信息
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
