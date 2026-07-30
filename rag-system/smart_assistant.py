"""
模块：综合智能助手
结合 RAG + Agent + Memory 的完整系统
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from rag_tools import RAG_TOOLS, set_collection, search_knowledge_base
from conversation_memory import memory
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from decorators import timer_and_log


class SmartAssistant:
    """综合智能助手：RAG + Agent + Memory"""

    def __init__(self, collection=None):
        self.llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self.collection = collection
        if collection:
            set_collection(collection)

    @timer_and_log
    def chat(self, query: str, session_id: str = "default", product: str = "") -> str:
        """
        完整的对话流程：
        1. 获取对话历史
        2. 检索知识库（按产品过滤）
        3. 结合历史 + 知识库 + 问题，生成回答
        4. 保存对话记录
        """
        # 1. 获取对话历史
        history_context = memory.get_context(session_id, last_n=5)

        # 2. 检索知识库（按产品过滤）
        from rag_tools import search_knowledge_base, list_all_products, rrf_retrieve, _filter_short_chunks, rerank_documents
        from config import RERANK_TOP_K
        product_context = "全部产品"
        
        if product:
            product_context = f"当前产品：{product}"
            # 按产品过滤文档
            try:
                results = self.collection.get(where={"product": product}, limit=200)
                product_docs = results["documents"] if results and results["documents"] else []
                if product_docs:
                    rrf_results = rrf_retrieve(query, product_docs, top_k=50)
                    merged = {hash(doc): doc for _, doc in rrf_results}
                    merged_docs = list(merged.values())
                    merged_docs = _filter_short_chunks(merged_docs)
                    reranked = rerank_documents(query, merged_docs, top_k=RERANK_TOP_K * 2)
                    final_docs = [doc for _, doc in reranked]
                    if final_docs:
                        kb_result = "\n".join([f"[文档{i+1}] {d}" for i, d in enumerate(final_docs[:50])])
                        print(f"  [产品限定] {product} → 三路检索 → {len(final_docs)} 个文档")
                    else:
                        # RRF 检索无结果，降级到该产品的全部文档（不跨产品搜索）
                        print(f"  [产品限定] {product} → RRF 检索无结果，使用该产品全部文档")
                        kb_result = "\n".join([f"[文档{i+1}] {d}" for i, d in enumerate(product_docs[:50])])
                else:
                    # 该产品没有文档，降级到全局搜索
                    print(f"  [产品限定] {product} → 无产品文档，降级到全局搜索")
                    kb_result = search_knowledge_base.invoke(query)
            except Exception as e:
                print(f"  [产品限定] 检索失败: {e}，降级到该产品全部文档")
                # 异常时也优先使用产品文档，而不是全局搜索
                if product_docs:
                    kb_result = "\n".join([f"[文档{i+1}] {d}" for i, d in enumerate(product_docs[:50])])
                else:
                    kb_result = search_knowledge_base.invoke(query)
        else:
            list_keywords = ["几款", "有哪些", "什么产品", "产品列表", "全部", "所有", "多少"]
            if any(kw in query for kw in list_keywords):
                kb_result = list_all_products.invoke("")
            else:
                kb_result = search_knowledge_base.invoke(query)

        # ===== 兜底策略：检索结果为空时获取全部文档 =====
        if not kb_result or kb_result.count("[文档") == 0:
            print(f"  [智能对话] 检索结果为空，兜底获取全部文档")
            try:
                total = self.collection.count()
                if product:
                    all_results = self.collection.get(where={"product": product}, limit=min(total, 200))
                else:
                    all_results = self.collection.get(limit=min(total, 200))
                all_docs = all_results["documents"] if all_results and all_results["documents"] else []
                if all_docs:
                    noise_words = ["隐私", "政策", "注销", "快递", "物流", "顺丰", "个人信息", "JWT", "private String"]
                    all_docs = [d for d in all_docs if not any(nw in d for nw in noise_words)]
                    kb_result = "\n".join([f"[文档{i+1}] {d}" for i, d in enumerate(all_docs[:50])])
                    print(f"  [智能对话] 兜底获取到 {len(all_docs)} 个文档块")
            except Exception as e:
                print(f"  [智能对话] 兜底失败: {e}")

        # 3. 生成回答（Jinja2模板）
        from prompt_templates import template_manager
        prompt = template_manager.render("chat",
            history_context=history_context,
            kb_result=kb_result,
            query=query,
            product_context=product_context
        )

        response = self.llm.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.3
        )

        answer = response.choices[0].message.content.strip()

        # 4. 保存对话记录
        memory.add_user_message(query, session_id)
        memory.add_ai_message(answer, session_id)

        return answer

    def clear_history(self, session_id: str = "default"):
        """清空对话历史"""
        memory.clear(session_id)

    def get_history(self, session_id: str = "default"):
        """获取对话历史"""
        return memory.get_messages(session_id)


# 全局助手实例
assistant = None


def init_assistant(collection):
    """初始化助手"""
    global assistant
    assistant = SmartAssistant(collection)
    return assistant


def chat_with_assistant(query: str, session_id: str = "default", product: str = "") -> str:
    """与助手对话"""
    global assistant
    if assistant is None:
        return "助手未初始化"
    return assistant.chat(query, session_id, product=product)
