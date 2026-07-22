"""
模块：对话记忆
实现多轮对话记忆功能
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from typing import Dict, List


class ConversationMemory:
    """对话记忆管理器"""

    def __init__(self):
        self.histories: Dict[str, InMemoryChatMessageHistory] = {}
        self.max_history = 20  # 最多保留20轮对话

    def get_history(self, session_id: str = "default") -> InMemoryChatMessageHistory:
        """获取指定会话的历史记录"""
        if session_id not in self.histories:
            self.histories[session_id] = InMemoryChatMessageHistory()
        return self.histories[session_id]

    def add_user_message(self, message: str, session_id: str = "default"):
        """添加用户消息"""
        history = self.get_history(session_id)
        history.add_user_message(message)
        self._trim_history(session_id)

    def add_ai_message(self, message: str, session_id: str = "default"):
        """添加 AI 回复"""
        history = self.get_history(session_id)
        history.add_ai_message(message)
        self._trim_history(session_id)

    def get_messages(self, session_id: str = "default") -> List:
        """获取消息列表"""
        history = self.get_history(session_id)
        return history.messages

    def get_context(self, session_id: str = "default", last_n: int = 5) -> str:
        """获取对话上下文字符串"""
        messages = self.get_messages(session_id)
        recent = messages[-last_n * 2:] if len(messages) > last_n * 2 else messages

        context = []
        for msg in recent:
            if isinstance(msg, HumanMessage):
                context.append(f"用户: {msg.content}")
            elif isinstance(msg, AIMessage):
                context.append(f"助手: {msg.content[:200]}")

        return "\n".join(context)

    def clear(self, session_id: str = "default"):
        """清空会话历史"""
        if session_id in self.histories:
            self.histories[session_id].clear()

    def _trim_history(self, session_id: str):
        """裁剪历史记录，保留最近的对话"""
        history = self.get_history(session_id)
        messages = history.messages
        if len(messages) > self.max_history * 2:
            history.messages = messages[-self.max_history * 2:]


# 全局记忆实例
memory = ConversationMemory()
