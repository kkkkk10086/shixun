"""
模块3：查询增强（优化版）
1. HyDE（假设性文档）- DeepSeek API
2. Query 重写 - DeepSeek API
3. 多扩展查询 - DeepSeek API
4. 候选文档重排 - 关键词过滤 + DeepSeek 智能重排
"""

import jieba
import jieba.analyse
from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


def get_llm():
    """获取 DeepSeek LLM 客户端"""
    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL
    )


# ========== 关键词匹配预过滤 ==========

def keyword_filter(query: str, documents: list, min_match: int = 1) -> list:
    """
    用关键词匹配过滤掉完全不相关的文档
    对于汇总类问题，不过滤（保留所有候选）
    """
    # 汇总类问题不过滤，保留所有文档
    summary_keywords = ["几款", "多少", "有哪些", "全部", "所有", "总共", "列表", "对比", "区别", "价格", "售价"]
    if any(kw in query for kw in summary_keywords):
        print(f"[过滤] 汇总类问题，跳过过滤，保留全部 {len(documents)} 个文档")
        return documents

    keywords = jieba.analyse.extract_tags(query, topK=8)
    if not keywords:
        return documents

    filtered = []
    for doc in documents:
        match_count = sum(1 for kw in keywords if kw in doc)
        if match_count >= min_match:
            filtered.append(doc)

    return filtered if filtered else documents


# ========== 功能1：HyDE（假设性文档） ==========

def hyde_query(query: str, collection) -> str:
    """用 DeepSeek 生成假设性文档，再用它去检索"""
    llm = get_llm()

    prompt = f"""请根据以下问题，写一段详细的回答（约150字）。
要求：回答要具体、包含产品名称和技术参数。只输出回答内容。

问题：{query}

回答："""

    try:
        response = llm.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.5
        )
        hypothetical_doc = response.choices[0].message.content.strip()
        print(f"[HyDE] {hypothetical_doc[:80]}...")
        return hypothetical_doc
    except Exception as e:
        print(f"[HyDE] 失败: {e}")
        return query


# ========== 功能2：Query 重写 ==========

def rewrite_query(query: str) -> str:
    """用 DeepSeek 将口语化问题重写为更适合检索的形式"""
    llm = get_llm()

    prompt = f"""将以下问题重写为更适合知识库检索的形式。
要求：保留核心语义，补充产品全称，使用书面语。只输出重写后的查询。

问题：{query}

重写："""

    try:
        response = llm.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.3
        )
        rewritten = response.choices[0].message.content.strip()
        print(f"[重写] {query} → {rewritten}")
        return rewritten
    except Exception as e:
        print(f"[重写] 失败: {e}")
        return query


# ========== 功能3：多扩展查询 ==========

def expand_queries(query: str, n: int = 3) -> list:
    """用 DeepSeek 将一个问题扩展为多个不同角度的查询"""
    llm = get_llm()

    prompt = f"""请将以下问题改写为{n}个不同的检索查询，每行一个。
要求：围绕同一主题，从不同角度表述。只输出查询，每行一个。

问题：{query}

{n}个查询："""

    try:
        response = llm.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7
        )
        content = response.choices[0].message.content.strip()
        queries = [q.strip().lstrip("0123456789.、）) ") for q in content.split("\n") if q.strip()]
        queries = [q for q in queries if q and q != query and len(q) > 3][:n]

        print(f"[多扩展] {len(queries)} 个")
        return queries if queries else [query]
    except Exception as e:
        print(f"[多扩展] 失败: {e}")
        return [query]


# ========== 功能4：DeepSeek 智能重排 ==========

def llm_rerank(query: str, documents: list, top_k: int = 3) -> list:
    """
    用 DeepSeek 对候选文档进行智能重排
    让 LLM 判断每个文档与查询的相关性，过滤掉不相关的
    """
    if not documents:
        return []

    llm = get_llm()

    # 构造重排 prompt
    docs_text = ""
    for i, doc in enumerate(documents):
        docs_text += f"[文档{i+1}] {doc[:250]}\n\n"

    prompt = f"""判断以下文档与查询的相关性，只输出保留的文档编号。

查询：{query}

候选文档：
{docs_text}

规则：
- 只要文档包含与查询相关的信息就保留
- 只输出保留的文档编号，用逗号分隔
- 格式示例：2,5,1

保留的文档编号："""

    try:
        response = llm.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1
        )
        rank_text = response.choices[0].message.content.strip()
        print(f"[LLM重排] {rank_text}")

        if "无" in rank_text:
            print("[LLM重排] LLM认为都不相关，使用关键词过滤结果")
            return documents[:top_k]

        # 解析排序结果 - 支持多种格式
        import re
        # 提取所有数字
        numbers = re.findall(r'\d+', rank_text)
        rank_ids = []
        for num in numbers:
            idx = int(num) - 1  # 转为0-based索引
            if 0 <= idx < len(documents):
                rank_ids.append(idx)

        reranked = [documents[i] for i in rank_ids if i < len(documents)]
        print(f"[LLM重排] 保留 {len(reranked)}/{len(documents)} 个文档")

        return reranked[:top_k]

    except Exception as e:
        print(f"[LLM重排] 失败: {e}")
        return documents[:top_k]


# ========== 组合增强检索 ==========

def enhanced_retrieval(query: str, collection, top_k: int = 3):
    """
    两阶段检索策略：
    阶段1：精确检索，看能否直接找到答案
    阶段2：如果检索结果不理想，扩大范围查看所有文档
    阶段3：综合所有信息生成回答
    """
    print("\n" + "=" * 50)
    print("增强检索开始")
    print("=" * 50)

    # ===== 阶段1：精确检索 =====
    print("[阶段1] 精确检索...")

    rewritten = rewrite_query(query)
    hyde_doc = hyde_query(query, collection)
    expanded = expand_queries(query, n=3)

    all_queries = [query, rewritten, hyde_doc] + expanded
    stage1_docs = []

    for q in all_queries:
        try:
            results = collection.query(query_texts=[q], n_results=10)
            docs = results["documents"][0] if results["documents"] else []
            stage1_docs.extend(docs)
        except Exception as e:
            print(f"  检索失败: {e}")

    # 去重
    seen = set()
    unique_docs = []
    for doc in stage1_docs:
        doc_hash = hash(doc)
        if doc_hash not in seen:
            seen.add(doc_hash)
            unique_docs.append(doc)

    # 关键词过滤
    filtered_docs = keyword_filter(query, unique_docs, min_match=1)
    print(f"[阶段1] 找到 {len(filtered_docs)} 个候选")

    # ===== 阶段2：判断是否需要扩大范围 =====
    # 如果候选太少，获取所有文档
    total_in_db = collection.count()
    print(f"[阶段2] 数据库共 {total_in_db} 个文档块")

    if len(filtered_docs) < 5 and total_in_db > 5:
        print("[阶段2] 候选不足，扩大检索范围...")
        try:
            # 获取更多文档（最多50个）
            fetch_limit = min(total_in_db, 50)
            all_results = collection.get(limit=fetch_limit)
            all_docs = all_results["documents"] if all_results["documents"] else []
            print(f"[阶段2] 获取到 {len(all_docs)} 个文档块")

            # 用原始查询重新过滤
            for doc in all_docs:
                doc_hash = hash(doc)
                if doc_hash not in seen:
                    seen.add(doc_hash)
                    unique_docs.append(doc)

            filtered_docs = keyword_filter(query, unique_docs, min_match=1)
            print(f"[阶段2] 扩大后 {len(filtered_docs)} 个候选")
        except Exception as e:
            print(f"[阶段2] 扩大检索失败: {e}")

    # ===== 阶段3：重排 + 生成回答 =====
    print("[阶段3] 重排与生成...")

    # 对于汇总类问题，保留更多文档
    is_summary_query = any(kw in query for kw in ["几款", "多少", "有哪些", "全部", "所有", "总共", "列表", "对比", "区别", "价格", "售价"])

    if is_summary_query:
        # 汇总类问题：不过滤，直接返回所有文档
        print(f"[阶段3] 汇总模式，跳过LLM重排，直接返回全部 {len(filtered_docs)} 个文档")
        reranked_docs = filtered_docs[:15]
    else:
        rerank_top_k = min(top_k * 2, 10)
        print(f"[阶段3] 精确模式，保留 Top-{rerank_top_k}")
        reranked_docs = llm_rerank(query, filtered_docs, top_k=rerank_top_k)
    answer = generate_answer(query, reranked_docs, collection=collection)

    print(f"\n[完成] 返回生成回答 + {len(reranked_docs)} 个参考文档")
    print("=" * 50)

    return reranked_docs, answer


# ========== LLM 生成回答 ==========

def generate_answer(query: str, documents: list, collection=None) -> str:
    """
    根据数据库中所有文档，用 DeepSeek 生成回答
    优先使用检索结果，再补充其他文档
    """
    llm = get_llm()

    # 收集所有文档：检索结果 + 数据库其他文档
    all_docs = list(documents) if documents else []
    seen_hashes = set(hash(d) for d in all_docs)

    if collection is not None:
        try:
            total = collection.count()
            all_results = collection.get(limit=min(total, 200))
            db_docs = all_results["documents"] if all_results["documents"] else []

            # 补充数据库中未在检索结果中的文档
            for doc in db_docs:
                doc_hash = hash(doc)
                if doc_hash not in seen_hashes:
                    seen_hashes.add(doc_hash)
                    all_docs.append(doc)

            print(f"[生成] 检索结果 {len(documents)} 个 + 数据库补充 {len(all_docs) - len(documents)} 个 = 共 {len(all_docs)} 个文档")
        except Exception as e:
            print(f"[生成] 获取全部文档失败: {e}")
    else:
        print(f"[生成] 使用检索结果 {len(all_docs)} 个文档")

    if not all_docs:
        return "知识库中暂无相关信息。"

    # 拼接所有文档（全部发送，不截断）
    context = "\n\n".join([f"【文档{i+1}】{doc}" for i, doc in enumerate(all_docs)])

    prompt = f"""你是讯飞产品知识库的智能助手。请根据以下所有文档内容回答用户问题。

回答策略：
1. 优先基于知识库文档回答
2. 如果知识库中有相关信息，直接使用
3. 如果知识库中没有完全匹配的信息，但有相关产品信息，结合已有信息回答
4. 如果知识库中完全没有相关信息，使用你自己的知识补充回答，并说明"以下为通用知识补充"

知识库全部文档（共{len(all_docs)}个）：
{context}

用户问题：{query}

直接回答（不要说"好的"、"作为助手"等开场白）："""

    try:
        response = llm.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.3
        )
        answer = response.choices[0].message.content.strip()
        print(f"[生成] 回答长度: {len(answer)} 字符")
        return answer
    except Exception as e:
        print(f"[生成] 失败: {e}")
        # 降级：直接返回检索结果摘要
        return f"检索到 {len(documents)} 条相关信息，但生成回答时出错。请查看下方检索结果。"
