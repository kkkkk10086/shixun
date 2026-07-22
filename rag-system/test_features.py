"""
功能测试文件
测试四项新功能：LangChain Tool、LangGraph Agent、对话记忆、综合智能助手
"""

import os
import sys

# 加载 .env 文件
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CHROMA_PERSIST_DIR


def test_langchain_tools():
    """测试1：LangChain Tool"""
    print("\n" + "=" * 60)
    print("【测试1】LangChain Tool（@tool 装饰器）")
    print("=" * 60)

    from rag_tools import RAG_TOOLS, search_knowledge_base, get_product_list, get_product_specs

    print(f"工具数量：{len(RAG_TOOLS)} 个")
    for t in RAG_TOOLS:
        print(f"  - {t.name}: {t.description[:50]}...")

    # 测试工具调用
    from embedding_store import load_vectorstore
    collection, _ = load_vectorstore()
    from rag_tools import set_collection
    set_collection(collection)

    print("\n测试 search_knowledge_base:")
    result = search_knowledge_base.invoke("智能办公本")
    print(f"  返回 {len(result)} 字符")
    print(f"  预览：{result[:150]}...")

    print("\n测试 get_product_list:")
    result = get_product_list.invoke("")
    print(f"  返回 {len(result)} 字符")
    print(f"  预览：{result[:150]}...")

    print("\n测试 get_product_specs:")
    result = get_product_specs.invoke("翻译机")
    print(f"  返回 {len(result)} 字符")
    print(f"  预览：{result[:150]}...")

    print("\n✅ LangChain Tool 测试通过")
    return True


def test_langgraph_agent():
    """测试2：LangGraph Agent"""
    print("\n" + "=" * 60)
    print("【测试2】LangGraph Agent（StateGraph 工作流）")
    print("=" * 60)

    from embedding_store import load_vectorstore
    collection, _ = load_vectorstore()

    from langgraph_agent import create_agent, run_agent_query

    print("创建 LangGraph Agent...")
    create_agent(collection)
    print("  Agent 创建成功")

    print("\n测试查询：讯飞有几款产品？")
    answer = run_agent_query("讯飞有几款产品？")
    print(f"  回答：{answer[:200]}...")

    print("\n✅ LangGraph Agent 测试通过")
    return True


def test_conversation_memory():
    """测试3：对话记忆"""
    print("\n" + "=" * 60)
    print("【测试3】Conversation Memory（多轮对话记忆）")
    print("=" * 60)

    from conversation_memory import ConversationMemory

    mem = ConversationMemory()
    session = "test_session"

    # 模拟多轮对话
    print("模拟多轮对话：")
    mem.add_user_message("你好，我想了解讯飞产品", session)
    mem.add_ai_message("你好！讯飞有智能办公本、翻译机、录音卡等产品。", session)
    mem.add_user_message("翻译机多少钱？", session)
    mem.add_ai_message("讯飞双屏翻译机2.0售价3499元。", session)
    mem.add_user_message("那个办公本呢？", session)
    mem.add_ai_message("讯飞智能办公本X2售价4999元。", session)

    print(f"  对话轮数：{len(mem.get_messages(session)) // 2} 轮")
    print(f"  消息总数：{len(mem.get_messages(session))} 条")

    print("\n测试获取上下文：")
    context = mem.get_context(session, last_n=3)
    print(f"  {context}")

    print("\n测试清空历史：")
    mem.clear(session)
    print(f"  清空后消息数：{len(mem.get_messages(session))} 条")

    print("\n✅ Conversation Memory 测试通过")
    return True


def test_smart_assistant():
    """测试4：综合智能助手"""
    print("\n" + "=" * 60)
    print("【测试4】Smart Assistant（RAG + Agent + Memory）")
    print("=" * 60)

    from embedding_store import load_vectorstore
    collection, _ = load_vectorstore()

    from smart_assistant import SmartAssistant

    print("初始化智能助手...")
    assistant = SmartAssistant(collection)
    print("  助手创建成功")

    session = "test_assistant"

    # 多轮对话测试
    print("\n--- 第1轮对话 ---")
    answer = assistant.chat("你好，讯飞有什么产品？", session)
    print(f"  问：讯飞有什么产品？")
    print(f"  答：{answer[:200]}...")

    print("\n--- 第2轮对话 ---")
    answer = assistant.chat("翻译机多少钱？", session)
    print(f"  问：翻译机多少钱？")
    print(f"  答：{answer[:200]}...")

    print("\n--- 第3轮对话（依赖上下文） ---")
    answer = assistant.chat("那个办公本怎么使用？", session)
    print(f"  问：那个办公本怎么使用？")
    print(f"  答：{answer[:200]}...")

    print(f"\n对话历史：{len(assistant.get_history(session))} 条消息")

    print("\n✅ Smart Assistant 测试通过")
    return True


def main():
    """运行所有测试"""
    print("=" * 60)
    print("四项新功能测试")
    print("=" * 60)

    results = {}

    try:
        results["LangChain Tool"] = test_langchain_tools()
    except Exception as e:
        print(f"\n❌ LangChain Tool 测试失败: {e}")
        results["LangChain Tool"] = False

    try:
        results["LangGraph Agent"] = test_langgraph_agent()
    except Exception as e:
        print(f"\n❌ LangGraph Agent 测试失败: {e}")
        results["LangGraph Agent"] = False

    try:
        results["Conversation Memory"] = test_conversation_memory()
    except Exception as e:
        print(f"\n❌ Conversation Memory 测试失败: {e}")
        results["Conversation Memory"] = False

    try:
        results["Smart Assistant"] = test_smart_assistant()
    except Exception as e:
        print(f"\n❌ Smart Assistant 测试失败: {e}")
        results["Smart Assistant"] = False

    # 汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n通过 {passed}/{total} 项测试")


if __name__ == "__main__":
    main()
