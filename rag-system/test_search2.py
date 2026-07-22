"""测试：检查 Agent 搜索是否能找到鼠标文档"""
import os, sys
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embedding_store import load_vectorstore

print("加载数据库...")
collection, _ = load_vectorstore()

# 模拟 search_knowledge 的逻辑
import jieba.analyse

query = "讯飞鼠标有什么功能"
print(f"\n查询: {query}")

# 获取所有文档
total = collection.count()
all_results = collection.get(limit=min(total, 200))
all_db_docs = all_results["documents"]
print(f"数据库总文档数: {len(all_db_docs)}")

# 提取关键词
keywords = jieba.analyse.extract_tags(query, topK=10)
print(f"提取的关键词: {keywords}")

# 关键词匹配
search_terms = [query] + keywords
matched = []
for doc in all_db_docs:
    match_count = sum(1 for term in search_terms if term in doc)
    if match_count >= 1:
        matched.append((match_count, doc))

matched.sort(key=lambda x: x[0], reverse=True)
print(f"\n关键词匹配到: {len(matched)} 个文档")
for i, (score, d) in enumerate(matched[:10]):
    has_mouse = "鼠标" in d
    print(f"  {i+1}. [分数:{score}] {'✅含鼠标' if has_mouse else '❌不含'} {d[:80]}...")

# 检查所有文档中是否有鼠标
mouse_in_all = [d for d in all_db_docs if "鼠标" in d]
print(f"\n所有文档中包含'鼠标'的: {len(mouse_in_all)} 个")
for i, d in enumerate(mouse_in_all):
    print(f"  {i+1}. {d[:100]}...")
