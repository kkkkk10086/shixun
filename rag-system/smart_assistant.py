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


class SmartAssistant:
    """综合智能助手：RAG + Agent + Memory"""

    def __init__(self, collection=None):
        self.llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self.collection = collection
        if collection:
            set_collection(collection)

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

        # 3. 生成回答
        prompt = f"""你是讯飞智能硬件产品的智能助手。

对话历史：
{history_context if history_context else "（这是对话开始）"}

知识库信息：
{kb_result}

用户问题：{query}

回答规则：
1. 如果知识库有信息，必须基于知识库回答，列出所有相关产品
2. 如果是产品列表问题，必须把知识库中提到的所有产品都列出来，不要遗漏
3. 如果知识库没有相关信息，用通用知识补充
4. 回答要准确、简洁、有条理

直接回答："""

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
