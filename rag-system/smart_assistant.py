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
    def chat(self, query: str, session_id: str = "default") -> str:
        """
        完整的对话流程：
        1. 获取对话历史
        2. 检索知识库
        3. 结合历史 + 知识库 + 问题，生成回答
        4. 保存对话记录
        """
        # 1. 获取对话历史
        history_context = memory.get_context(session_id, last_n=5)

        # 2. 检索知识库（产品列表类问题获取所有产品）
        from rag_tools import search_knowledge_base, list_all_products, get_product_specs
        list_keywords = ["几款", "有哪些", "什么产品", "产品列表", "全部", "所有", "多少"]
        if any(kw in query for kw in list_keywords):
            kb_result = list_all_products.invoke("")
        else:
            kb_result = search_knowledge_base.invoke(query)

        # 3. 生成回答（Jinja2模板）
        from prompt_templates import template_manager
        prompt = template_manager.render("chat",
            history_context=history_context,
            kb_result=kb_result,
            query=query
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


def chat_with_assistant(query: str, session_id: str = "default") -> str:
    """与助手对话"""
    global assistant
    if assistant is None:
        return "助手未初始化"
    return assistant.chat(query, session_id)
