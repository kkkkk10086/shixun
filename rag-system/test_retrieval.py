"""
RAG 系统功能测试
测试 5 种检索模式
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CHROMA_PERSIST_DIR
from embedding_store import load_vectorstore, basic_search
from query_enhancer import enhanced_retrieval


# 测试用例
TEST_QUERIES = [
    "智能办公本怎么使用？",
    "讯飞有几款产品？价格分别是多少？",
    "翻译机支持哪些语言？",
    "录音卡有什么功能？",
    "办公本和翻译机有什么区别？"
]


def test_basic_search(collection, query):
    """测试1：基础检索"""
    print(f"\n{'='*50}")
    print(f"【测试1】基础检索")
    print(f"查询：{query}")
    print(f"{'='*50}")

    results = basic_search(collection, query, k=3)
    docs = results["documents"][0] if results["documents"] else []

    print(f"返回 {len(docs)} 个结果：")
    for i, doc in enumerate(docs):
        print(f"  {i+1}. {doc[:100]}...")

    return docs


def test_enhanced_retrieval(collection, query):
    """测试2：增强检索"""
    print(f"\n{'='*50}")
    print(f"【测试2】增强检索（Query重写 + HyDE + 多扩展 + 重排 + 生成）")
    print(f"查询：{query}")
    print(f"{'='*50}")

    docs, answer = enhanced_retrieval(query, collection, top_k=3)

    print(f"\n检索到 {len(docs)} 个文档")
    print(f"\n智能回答：{answer}")

    return docs, answer


def get_search_tool(collection):
    """创建增强版检索工具（直接复用增强检索策略）"""
    def search_knowledge(query_text: str) -> str:
        """从知识库检索，使用完整的增强检索策略"""
        try:
            from query_enhancer import rewrite_query, hyde_query, expand_queries, keyword_filter

            rewritten = rewrite_query(query_text)
            hyde_doc = hyde_query(query_text, collection)
            expanded = expand_queries(query_text, n=3)

            all_queries = [query_text, rewritten, hyde_doc] + expanded
            all_docs = []
            seen = set()

            for q in all_queries:
                try:
                    results = collection.query(query_texts=[q], n_results=15)
                    docs = results["documents"][0] if results["documents"] else []
                    for d in docs:
                        h = hash(d)
                        if h not in seen:
                            seen.add(h)
                            all_docs.append(d)
                except Exception:
                    pass

            filtered = keyword_filter(query_text, all_docs, min_match=1)

            noise_words = ["隐私", "政策", "注销", "快递", "物流", "顺丰", "个人信息"]
            product_docs = [d for d in filtered if not any(nw in d for nw in noise_words)]
            if not product_docs:
                product_docs = filtered

            print(f"  [搜索工具] {query_text} → {rewritten} → {len(product_docs)} 个结果")
            return "\n".join([f"[文档{i+1}] {d[:400]}" for i, d in enumerate(product_docs[:15])]) if product_docs else "未找到相关信息"
        except Exception as e:
            return f"检索出错: {e}"

    return {"search_knowledge": search_knowledge}


def test_react_agent(collection, query):
    """测试3：ReAct Agent"""
    print(f"\n{'='*50}")
    print(f"【测试3】ReAct Agent（推理+行动循环）")
    print(f"查询：{query}")
    print(f"{'='*50}")

    from agent_paradigms import react_agent
    tools = get_search_tool(collection)
    answer = react_agent(query, tools)

    print(f"\n最终回答：{answer}")
    return answer


def test_plan_and_solve(collection, query):
    """测试4：Plan-and-Solve Agent"""
    print(f"\n{'='*50}")
    print(f"【测试4】Plan-and-Solve Agent（规划+执行）")
    print(f"查询：{query}")
    print(f"{'='*50}")

    from agent_paradigms import plan_and_solve_agent
    tools = get_search_tool(collection)
    answer = plan_and_solve_agent(query, tools)

    print(f"\n最终回答：{answer}")
    return answer


def test_reflection_agent(collection, query):
    """测试5：Reflection Agent"""
    print(f"\n{'='*50}")
    print(f"【测试5】Reflection Agent（生成→反思→改进）")
    print(f"查询：{query}")
    print(f"{'='*50}")

    from agent_paradigms import reflection_agent
    tools = get_search_tool(collection)
    answer = reflection_agent(query, tools)

    print(f"\n最终回答：{answer}")
    return answer


def main():
    """运行所有测试"""
    print("=" * 60)
    print("RAG 系统功能测试")
    print("测试 5 种检索模式")
    print("=" * 60)

    # 加载向量数据库
    print("\n加载向量数据库...")
    try:
        collection, _ = load_vectorstore()
        print(f"数据库加载成功，共 {collection.count()} 个文档块")
    except Exception as e:
        print(f"加载失败: {e}")
        return

    # 测试查询
    query = TEST_QUERIES[0]  # 使用第一个查询

    # 测试1：基础检索
    test_basic_search(collection, query)

    # 测试2：增强检索
    test_enhanced_retrieval(collection, query)

    # 测试3：ReAct Agent
    test_react_agent(collection, query)

    # 测试4：Plan-and-Solve Agent
    test_plan_and_solve(collection, query)

    # 测试5：Reflection Agent
    test_reflection_agent(collection, query)

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
