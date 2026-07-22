"""
模块：LangChain Tool 封装
将 RAG 检索功能封装为标准的 LangChain Tool
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from config import CHROMA_PERSIST_DIR


# 全局变量，启动时初始化
_collection = None


def set_collection(collection):
    """设置全局向量数据库"""
    global _collection
    _collection = collection


@tool
def search_knowledge_base(query: str) -> str:
    """从讯飞智能硬件产品知识库中检索相关信息。输入用户的问题，返回相关的文档内容。"""
    global _collection
    if _collection is None:
        return "知识库未初始化"

    try:
        all_docs = []
        seen = set()

        # 获取所有文档
        total = _collection.count()
        all_results = _collection.get(limit=min(total, 200))
        all_db_docs = all_results["documents"] if all_results["documents"] else []

        # 提取所有搜索词
        search_terms = [query]
        try:
            import jieba.analyse
            keywords = jieba.analyse.extract_tags(query, topK=10)
            search_terms.extend(keywords)
        except Exception:
            pass

        # 直接在文档内容中搜索
        for doc in all_db_docs:
            match_count = sum(1 for term in search_terms if term in doc)
            if match_count >= 1:
                h = hash(doc)
                if h not in seen:
                    seen.add(h)
                    all_docs.append((match_count, doc))

        all_docs.sort(key=lambda x: x[0], reverse=True)

        # 过滤
        noise_words = ["隐私", "政策", "注销", "快递", "物流", "顺丰", "个人信息", "JWT", "private String"]
        product_docs = [(s, d) for s, d in all_docs if not any(nw in d for nw in noise_words)]
        if not product_docs:
            product_docs = all_docs

        docs_only = [d for _, d in product_docs]
        return "\n".join([f"[文档{i+1}] {d[:400]}" for i, d in enumerate(docs_only[:20])])
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
        all_results = _collection.get(limit=min(total, 200))
        all_docs = all_results["documents"] if all_results["documents"] else []

        # 提取产品信息
        products = {}
        product_keywords = {
            "录音笔": ["录音笔", "录音设备", "SR", "H1"],
            "翻译机": ["翻译机", "翻译设备"],
            "办公本": ["办公本", "X2", "X2LAMY"],
            "录音卡": ["录音卡"],
            "键盘": ["键盘", "T8"],
            "鼠标": ["鼠标"],
            "词典笔": ["词典笔", "X8"],
            "学习机": ["学习机", "S90"],
            "英语宝": ["英语宝", "EBOX"]
        }

        for doc in all_docs:
            for product_name, keywords in product_keywords.items():
                if any(kw in doc for kw in keywords):
                    if product_name not in products:
                        products[product_name] = doc[:200]
                    break

        # 构建输出
        output = "讯飞产品清单：\n"
        for i, (name, info) in enumerate(products.items(), 1):
            output += f"{i}. {name}：{info[:100]}...\n"

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
