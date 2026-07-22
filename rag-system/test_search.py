"""快速测试：检查搜索能否找到鼠标文档"""
import os, sys
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embedding_store import load_vectorstore

print("加载数据库...")
collection, _ = load_vectorstore()

# 测试1：直接获取所有文档
print("\n【测试1】获取所有文档")
all_results = collection.get(limit=200)
all_docs = all_results["documents"]
print(f"  总文档数: {len(all_docs)}")

# 检查哪些文档包含"鼠标"
mouse_docs = [d for d in all_docs if "鼠标" in d]
print(f"  包含'鼠标'的文档: {len(mouse_docs)} 个")
for i, d in enumerate(mouse_docs):
    print(f"    {i+1}. {d[:100]}...")

# 测试2：用关键词匹配
print("\n【测试2】关键词匹配搜索")
search_terms = ["鼠标", "AM50", "AM50Pro"]
matched = []
for doc in all_docs:
    for term in search_terms:
        if term in doc:
            matched.append(doc)
            break
print(f"  匹配到: {len(matched)} 个文档")
for i, d in enumerate(matched[:3]):
    print(f"    {i+1}. {d[:100]}...")

# 测试3：搜索 "讯飞鼠标有什么功能"
print("\n【测试3】搜索: 讯飞鼠标有什么功能")
query = "讯飞鼠标有什么功能"
import jieba.analyse
keywords = jieba.analyse.extract_tags(query, topK=10)
print(f"  提取的关键词: {keywords}")

matched2 = []
for doc in all_docs:
    match_count = sum(1 for kw in keywords if kw in doc)
    if match_count >= 1:
        matched2.append((match_count, doc))
matched2.sort(key=lambda x: x[0], reverse=True)
print(f"  匹配到: {len(matched2)} 个文档")
for i, (score, d) in enumerate(matched2[:5]):
    print(f"    {i+1}. [分数:{score}] {d[:80]}...")
