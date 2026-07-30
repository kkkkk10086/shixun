"""
模块3：查询增强（优化版）
1. HyDE（假设性文档）- DeepSeek API
2. Query 重写 - DeepSeek API
3. 多扩展查询 - DeepSeek API
4. RRF 融合多路检索结果
5. Cross-Encoder 智能重排（替代原有 LLM 重排）
"""

import jieba
import jieba.analyse
from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from config import RETRIEVAL_TOP_K, RERANK_TOP_K
from cache import query_cache, retrieval_cache, rerank_cache, answer_cache, cached, make_cache_key

# 汇总类查询关键词
SUMMARY_KEYWORDS = ["几款", "多少", "有哪些", "全部", "所有", "总共", "列表", "对比", "区别", "价格", "售价"]


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
    if any(kw in query for kw in SUMMARY_KEYWORDS):
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

@cached(query_cache, "hyde", ttl=1800)
def hyde_query(query: str, collection=None) -> str:
    """用 DeepSeek 生成假设性文档，再用它去检索（结果缓存30分钟）"""
    llm = get_llm()

    from prompt_templates import template_manager
    prompt = template_manager.render("hyde", query=query)

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

@cached(query_cache, "rewrite", ttl=1800)
def rewrite_query(query: str) -> str:
    """用 DeepSeek 将口语化问题重写为更适合检索的形式（结果缓存30分钟）"""
    llm = get_llm()

    from prompt_templates import template_manager
    prompt = template_manager.render("query_rewrite", query=query)

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

@cached(query_cache, "expand", ttl=1800)
def expand_queries(query: str, n: int = 3) -> list:
    """用 DeepSeek 将一个问题扩展为多个不同角度的查询（结果缓存30分钟）"""
    llm = get_llm()

    from prompt_templates import template_manager
    prompt = template_manager.render("query_expand", query=query, n=n)

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


# ========== 功能4：Cross-Encoder 重排（替代原有 LLM 重排） ==========

def cross_encoder_rerank(query: str, documents: list, top_k: int = RERANK_TOP_K) -> list:
    """
    使用 Cross-Encoder 模型对候选文档进行重排
    比 LLM 重排更快、更稳定、更便宜
    结果缓存30分钟
    """
    if not documents:
        return []

    # 过滤短标题块（< 50字符的纯标题对重排有害无益）
    filtered = [d for d in documents if len(d.strip()) >= 50]
    if not filtered:
        filtered = documents
    elif len(filtered) < len(documents):
        print(f"  [过滤] 移除 {len(documents) - len(filtered)} 个短标题块")
    documents = filtered

    # 用文档内容hash做缓存键
    docs_hash = str(hash(tuple(hash(d) for d in documents)))
    cache_key = make_cache_key("rerank", query, f"k={top_k}", docs_hash)
    cached_result = rerank_cache.get(cache_key)
    if cached_result is not None:
        print(f"  [缓存] 命中重排结果 ({len(cached_result)} 个文档)")
        return cached_result

    try:
        from rag_tools import get_reranker
        reranker = get_reranker()
        reranked = reranker.rerank(query, documents, top_k=top_k)
        result = [doc for _, doc in reranked]
        rerank_cache.set(cache_key, result, ttl=1800)
        return result
    except Exception as e:
        print(f"[CE重排] 失败: {e}，降级为保留原始顺序")
        return documents[:top_k]


# ========== 组合增强检索 ==========

def enhanced_retrieval(query: str, collection, top_k: int = 3, product: str = ""):
    """
    增强检索策略（三路融合版）：
    阶段1：Query重写 + HyDE + 多扩展查询 → 三路检索（关键词+向量+BM25）→ RRF融合
    阶段2：候选不足时自动扩大检索范围（扩大 n_results + 降低关键词阈值 + BM25兜底）
    阶段3：再次不足时获取全部文档做兜底
    阶段4：Cross-Encoder 重排 → LLM 生成回答
    """
    from rag_tools import three_way_search, reciprocal_rank_fusion, get_reranker as get_ce_reranker

    print("\n" + "=" * 50)
    print("增强检索开始（三路融合：BGE-large + 关键词 + BM25 + RRF + Cross-Encoder）")
    if product:
        print(f"【产品限定】只检索: {product}")
    print("=" * 50)

    # ===== 阶段1：多角度查询生成 =====
    print("[阶段1] 生成多角度查询...")

    rewritten = rewrite_query(query)
    hyde_doc = hyde_query(query, collection)
    expanded = expand_queries(query, n=5)  # 增加到5个扩展查询

    all_queries = [query, rewritten, hyde_doc] + expanded
    print(f"[阶段1] 共 {len(all_queries)} 个查询维度")

    # ===== 阶段1：按产品过滤获取数据库文档 =====
    total_in_db = collection.count() if collection else 0
    print(f"[阶段1] 数据库共 {total_in_db} 个文档块")

    try:
        # 如果指定了产品，按 metadata 过滤
        where_filter = {"product": product} if product else None
        all_results = collection.get(limit=min(total_in_db, 500), where=where_filter)
        all_db_docs = all_results["documents"] if all_results["documents"] else []
        if product and all_db_docs:
            print(f"[阶段1] 产品 '{product}' 过滤后 → {len(all_db_docs)} 个文档块")
    except Exception as e:
        print(f"[阶段1] 获取{'产品过滤后' if product else '全库'}文档失败: {e}")
        # 降级：获取全部文档
        if product:
            print(f"[阶段1] 产品过滤失败，降级为全库检索")
            all_results = collection.get(limit=min(total_in_db, 500))
            all_db_docs = all_results["documents"] if all_results["documents"] else []
        else:
            all_db_docs = []

    # 构建 id↔doc 映射
    id_to_doc = {i: doc for i, doc in enumerate(all_db_docs)}
    doc_to_id = {doc: i for i, doc in id_to_doc.items()}

    # ===== 阶段1：对每个查询做三路检索 → RRF 融合 → 合并所有结果 =====
    stage1_docs = []
    seen = set()

    for qi, q in enumerate(all_queries):
        cache_key = make_cache_key("threeway_rrf", q)
        cached_docs = retrieval_cache.get(cache_key)
        if cached_docs is not None:
            print(f"  [缓存] 命中三路检索: {q[:40]}...({len(cached_docs)} 个)")
            rrf_results = cached_docs
        else:
            try:
                # 三路检索
                three_result = three_way_search(q, all_db_docs, top_k=RETRIEVAL_TOP_K)
                # RRF 融合
                ranked_lists = [
                    three_result["keyword_ids"],
                    three_result["vector_ids"],
                    three_result["bm25_ids"],
                ]
                fusion = reciprocal_rank_fusion(ranked_lists, k=RRF_K, top_k=30)
                rrf_results = [(score, id_to_doc.get(doc_id, "")) for score, doc_id in fusion if doc_id in id_to_doc]
                retrieval_cache.set(cache_key, rrf_results, ttl=600)
            except Exception as e:
                print(f"  三路检索失败 [{q[:40]}]: {e}")
                # 降级为纯向量检索
                try:
                    results = collection.query(query_texts=[q], n_results=RETRIEVAL_TOP_K)
                    docs = results["documents"][0] if results["documents"] else []
                    rrf_results = [(1.0 / (i + 1), d) for i, d in enumerate(docs)]
                except Exception:
                    rrf_results = []

        # 去重合并
        for score, doc in rrf_results:
            if doc:
                doc_hash = hash(doc)
                if doc_hash not in seen:
                    seen.add(doc_hash)
                    stage1_docs.append(doc)

    # 关键词过滤
    filtered_docs = keyword_filter(query, stage1_docs, min_match=1)
    print(f"[阶段1] 三路融合后 {len(filtered_docs)} 个候选")

    # ===== 阶段2：候选不足时，扩大检索范围 =====
    if len(filtered_docs) < 5 and total_in_db > 5:
        print("[阶段2] 候选不足（< 5条），扩大检索范围...")
        try:
            # 策略A：扩大向量检索 n_results
            expanded_n = RETRIEVAL_TOP_K * 3
            for q in all_queries[:3]:
                try:
                    results = collection.query(query_texts=[q], n_results=expanded_n)
                    docs = results["documents"][0] if results["documents"] else []
                    for doc in docs:
                        doc_hash = hash(doc)
                        if doc_hash not in seen:
                            seen.add(doc_hash)
                            filtered_docs.append(doc)
                except Exception as e:
                    print(f"  扩大检索失败 [{q[:30]}]: {e}")

            # 策略B：BM25 兜底检索
            if len(filtered_docs) < 5:
                print("[阶段2] 向量扩大仍不足，使用BM25兜底...")
                try:
                    from rag_tools import BM25
                    bm25 = BM25(all_db_docs)
                    bm25_results = bm25.search(query, top_k=30)
                    for _, idx in bm25_results:
                        if idx < len(all_db_docs):
                            doc = all_db_docs[idx]
                            doc_hash = hash(doc)
                            if doc_hash not in seen:
                                seen.add(doc_hash)
                                filtered_docs.append(doc)
                except Exception as e:
                    print(f"  BM25兜底失败: {e}")

            # 策略C：降低关键词匹配阈值到0
            if len(filtered_docs) < 5:
                print("[阶段2] 仍不足，放宽关键词过滤...")
                filtered_docs = keyword_filter(query, filtered_docs, min_match=0)

            # 策略D：最终兜底，获取全部文档
            if len(filtered_docs) < 5:
                print("[阶段2] 最终兜底：获取全部文档块...")
                try:
                    fetch_limit = min(total_in_db, 100)
                    all_results = collection.get(limit=fetch_limit)
                    all_docs = all_results["documents"] if all_results["documents"] else []
                    for doc in all_docs:
                        doc_hash = hash(doc)
                        if doc_hash not in seen:
                            seen.add(doc_hash)
                            filtered_docs.append(doc)
                except Exception as e:
                    print(f"[阶段2] 兜底获取失败: {e}")

            print(f"[阶段2] 扩大后 {len(filtered_docs)} 个候选")
        except Exception as e:
            print(f"[阶段2] 扩大检索失败: {e}")

    # ===== 阶段3：Cross-Encoder 重排 + 生成回答 =====
    rerank_method = "Cross-Encoder重排"
    try:
        r = get_ce_reranker()
        if r.mode:
            rerank_method = r.mode_display
    except Exception:
        pass
    print(f"[阶段3] {rerank_method}...")

    is_summary = any(kw in query for kw in SUMMARY_KEYWORDS)

    if is_summary:
        print(f"[阶段3] 汇总模式，跳过重排，直接返回全部 {len(filtered_docs)} 个文档")
        reranked_docs = filtered_docs[:20]
    else:
        # 重排更多候选，提高召回
        rerank_top_k = min(top_k * 3, 15)
        print(f"[阶段3] {rerank_method}，重排 {len(filtered_docs)} 候选 → 保留 Top-{rerank_top_k}")
        reranked_docs = cross_encoder_rerank(query, filtered_docs, top_k=rerank_top_k)

    # LLM 生成回答（传递产品上下文）
    answer = generate_answer(query, reranked_docs, collection=collection, product=product)

    print(f"\n[完成] 三路融合检索 → Cross-Encoder重排({len(reranked_docs)}个) → LLM生成回答")
    print("=" * 50)

    return reranked_docs, answer


# ========== LLM 生成回答 ==========

def generate_answer(query: str, documents: list, collection=None, product: str = "") -> str:
    """
    根据数据库中所有文档，用 DeepSeek 生成回答
    优先使用检索结果，再补充其他文档
    """
    # 缓存键：基于 query + 文档数量 + 首尾文档hash
    docs_hash = str(hash(tuple(hash(d) for d in documents[:10])))
    cache_key = make_cache_key("answer", query, f"n={len(documents)}", docs_hash)
    cached_answer = answer_cache.get(cache_key)
    if cached_answer is not None:
        print(f"  [缓存] 命中已回答 ({len(cached_answer)} 字符)")
        return cached_answer

    llm = get_llm()

    all_docs = list(documents) if documents else []
    seen_hashes = set(hash(d) for d in all_docs)

    if collection is not None:
        try:
            total = collection.count()
            # 如果指定了产品，只补充该产品的文档
            where_filter = {"product": product} if product else None
            all_results = collection.get(limit=min(total, 200), where=where_filter)
            db_docs = all_results["documents"] if all_results["documents"] else []

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

    product_context = f"当前产品：{product}" if product else "全部产品"
    context = "\n\n".join([f"【文档{i+1}】{doc}" for i, doc in enumerate(all_docs)])

    from prompt_templates import template_manager
    prompt = template_manager.render("answer", doc_count=len(all_docs), context=context, query=query, product_context=product_context)

    try:
        response = llm.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.3
        )
        answer = response.choices[0].message.content.strip()
        # 缓存生成结果（1小时）
        answer_cache.set(cache_key, answer, ttl=3600)
        print(f"[生成] 回答长度: {len(answer)} 字符 (已缓存)")
        return answer
    except Exception as e:
        print(f"[生成] 失败: {e}")
        return f"检索到 {len(documents)} 条相关信息，但生成回答时出错。请查看下方检索结果。"
