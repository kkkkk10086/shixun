"""
模块：LangGraph Agent
用 LangGraph 实现复杂工作流的 Agent
适配 DeepSeek API（不完全支持 tool_calls）
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from rag_tools import RAG_TOOLS, set_collection, search_knowledge_base
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


def create_agent(collection):
    """创建 LangGraph Agent（适配 DeepSeek）"""
    set_collection(collection)
    return True


def run_agent_query(query: str) -> str:
    """
    运行 Agent 查询（简化版，适配 DeepSeek）
    流程：检索知识库 → 发给 LLM 分析 → 返回回答
    """
    # 1. 检索知识库
    kb_result = search_knowledge_base.invoke(query)

    # 2. 调用 LLM 分析
    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.3
    )

    prompt = f"""你是讯飞产品知识库的智能助手。请根据以下知识库信息回答用户问题。

知识库信息：
{kb_result}

用户问题：{query}

回答规则：
1. 基于知识库信息回答
2. 如果知识库没有相关信息，用通用知识补充
3. 回答要准确、简洁

直接回答："""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


if __name__ == "__main__":
    print("LangGraph Agent 模块")
    print("需要先初始化向量数据库")
