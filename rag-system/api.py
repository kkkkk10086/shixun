"""
RAG 智能检索系统 - 后端 API
FastAPI 提供 RESTful 接口
"""

import os
import sys
import json
import uuid

# 必须在所有 huggingface_hub 相关 import 之前设置镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 项目根目录（rag-system 目录），确保所有相对路径正确
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)  # 切换工作目录，防止从父目录启动时找不到 frontend/ 等

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from config import DOCS_DIR, CHROMA_PERSIST_DIR, RETRIEVAL_TOP_K, RERANK_TOP_K, OUTPUT_DIR, EMBEDDING_MODEL
from document_processor import process_documents, save_chunks, save_markdown
from embedding_store import create_vectorstore, load_vectorstore, basic_search
from query_enhancer import enhanced_retrieval
from mysql_store import init_database, save_chunks_to_mysql, get_stats
from video_knowledgebase import video_store, Video, VideoChapter, init_demo_videos
from document_classifier import classify_document, DOCUMENT_TYPES, get_document_type_stats
from video_topic_pool import topic_pool, VideoTopic, init_demo_topics
from video_pipeline import pipeline, PipelineTask, get_product_template, PRODUCT_LINE_TEMPLATES
from review_center import review_center, ReviewRecord, REVIEW_TYPES, init_demo_reviews

app = FastAPI(title="RAG 智能检索系统")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
collection = None
embedding_fn = None


def get_product_docs(product: str, max_docs: int = 200):
    """
    按产品筛选文档块
    返回筛选后的文档列表，空列表表示获取全部
    """
    global collection
    if not product or not collection:
        return []

    try:
        # 从 ChromaDB 按 metadata 过滤
        results = collection.get(where={"product": product}, limit=max_docs)
        docs = results["documents"] if results and results["documents"] else []
        if docs:
            print(f"  [产品过滤] {product} → {len(docs)} 个文档块")
        return docs
    except Exception as e:
        print(f"  [产品过滤] 失败: {e}")
        return []


class QueryRequest(BaseModel):
    query: str
    mode: str = "enhanced"
    top_k: int = 3
    agent_mode: str = "none"  # none / react / plan_and_solve / reflection
    product: str = ""  # 选中的产品类别，为空表示全部产品


class EvaluateRequest(BaseModel):
    query: str
    ground_truth: Optional[str] = None
    top_k: int = 5


class BatchEvaluateRequest(BaseModel):
    test_cases: List[dict]  # [{"query": "...", "ground_truth": "..."}, ...]
    top_k: int = 5


class QueryResponse(BaseModel):
    query: str
    mode: str
    results: List[dict]
    enhanced_info: Optional[dict] = None


# ========== 视频知识库 API 模型 ==========

class VideoCreateRequest(BaseModel):
    video_id: str = ""
    product_line: str = ""
    product_model: str = ""
    firmware_version: str = ""
    title: str = ""
    duration: int = 0
    video_url: str = ""
    thumbnail_url: str = ""
    applicable_questions: List[str] = []
    chapters: List[dict] = []
    review_status: str = "draft"


class VideoUpdateRequest(BaseModel):
    product_line: Optional[str] = None
    product_model: Optional[str] = None
    firmware_version: Optional[str] = None
    title: Optional[str] = None
    duration: Optional[int] = None
    video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    applicable_questions: Optional[List[str]] = None
    review_status: Optional[str] = None


class ChapterCreateRequest(BaseModel):
    start_time: int
    end_time: int
    title: str


class ChapterUpdateRequest(BaseModel):
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    title: Optional[str] = None


@app.on_event("startup")
async def startup():
    global collection, embedding_fn

    print("正在初始化 RAG 系统...")

    # 检测 ChromaDB 是否存在且是否是旧模型版本
    model_marker_path = os.path.join(CHROMA_PERSIST_DIR, ".model_version")
    doc_hash_path = os.path.join(CHROMA_PERSIST_DIR, ".doc_hash")
    current_model = EMBEDDING_MODEL.replace("/", "_")

    # 收集所有文档路径，用于检测文档变更
    all_doc_paths = list(DOCS_DIR) if DOCS_DIR else []
    uploads_dir = "./uploads"
    if os.path.exists(uploads_dir):
        for f in os.listdir(uploads_dir):
            file_path = os.path.join(uploads_dir, f)
            if os.path.isfile(file_path):
                all_doc_paths.append(file_path)

    # 计算当前文档集合的指纹（基于文件路径+修改时间）
    import hashlib
    doc_fingerprints = []
    for dp in sorted(all_doc_paths):
        if os.path.exists(dp):
            mtime = os.path.getmtime(dp)
            size = os.path.getsize(dp)
            doc_fingerprints.append(f"{dp}:{mtime}:{size}")
    current_doc_hash = hashlib.md5("|".join(doc_fingerprints).encode()).hexdigest() if doc_fingerprints else ""

    need_rebuild = False

    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        # 检查模型版本标记
        stored_model = ""
        if os.path.exists(model_marker_path):
            with open(model_marker_path, "r") as f:
                stored_model = f.read().strip()

        if stored_model != current_model:
            print(f"检测到 Embedding 模型变更: {stored_model} → {current_model}")
            need_rebuild = True
        else:
            # 检查文档是否变更
            stored_doc_hash = ""
            if os.path.exists(doc_hash_path):
                with open(doc_hash_path, "r") as f:
                    stored_doc_hash = f.read().strip()

            if stored_doc_hash != current_doc_hash:
                print(f"检测到文档变更（新增/修改），需要重建向量数据库...")
                need_rebuild = True
            else:
                # 模型没变，文档没变，尝试加载
                print("加载已有向量数据库...")
                try:
                    collection, embedding_fn = load_vectorstore()
                    test = collection.count()
                    print(f"加载成功，共 {test} 个文档块")
                    return
                except Exception as e:
                    print(f"加载失败 ({e})，重新构建...")
                    need_rebuild = True

    if not need_rebuild and not os.path.exists(CHROMA_PERSIST_DIR):
        need_rebuild = True

    if not need_rebuild:
        return

    # 重建向量数据库
    print("处理文档...")

    all_chunks = process_documents(all_doc_paths)

    # 同时加载爬虫数据
    output_dir = "./output"
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            if f.endswith(".txt") and "chunks" not in f:
                file_path = os.path.join(output_dir, f)
                try:
                    with open(file_path, "r", encoding="utf-8") as fp:
                        text = fp.read()
                    if text.strip():
                        all_chunks[f"淘宝_{f}"] = [text]
                except Exception:
                    pass

    if not all_chunks:
        print("未找到文档")
        return

    save_chunks(all_chunks)

    # 构建带元数据的块列表（Chunk 对象或纯文本均可）
    all_texts_with_meta = []
    for chunks in all_chunks.values():
        all_texts_with_meta.extend(chunks)

    collection, embedding_fn = create_vectorstore(all_texts_with_meta)

    # 保存文档指纹，下次启动时检测文档变更
    if current_doc_hash:
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        with open(doc_hash_path, "w") as f:
            f.write(current_doc_hash)

    # 尝试存入 MySQL
    try:
        init_database()
        save_chunks_to_mysql(all_chunks)
    except Exception as e:
        print(f"MySQL 存储跳过: {e}")

    # 初始化视频示例数据
    try:
        init_demo_videos()
    except Exception as e:
        print(f"视频示例数据初始化跳过: {e}")

    # 初始化选题库示例数据
    try:
        init_demo_topics()
    except Exception as e:
        print(f"选题库示例数据初始化跳过: {e}")

    # 初始化审核中心示例数据
    try:
        init_demo_reviews()
    except Exception as e:
        print(f"审核中心示例数据初始化跳过: {e}")

    print("初始化完成！")


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")


@app.get("/evaluate")
async def evaluate_page():
    return FileResponse("frontend/evaluate.html")


@app.get("/video_admin")
async def video_admin_page():
    return FileResponse("frontend/video_admin.html")


@app.get("/admin")
async def admin_page():
    return FileResponse("frontend/admin.html")


@app.post("/api/query")
async def query(request: QueryRequest):
    global collection

    if collection is None:
        raise HTTPException(status_code=400, detail="向量数据库未初始化，请先处理文档")

    # 如果选择了 Agent 模式，使用 Agent 范式
    if request.agent_mode and request.agent_mode != "none":
        from agent_paradigms import run_agent

        # 创建产品过滤后的检索上下文
        agent_product = request.product

        # 工具：从知识库检索相关信息（支持产品过滤）
        def search_knowledge(query_text: str) -> str:
            """从知识库检索相关信息"""
            from rag_tools import set_collection, rrf_retrieve, _filter_short_chunks, rerank_documents
            from config import RERANK_TOP_K
            set_collection(collection)

            if agent_product:
                # 产品过滤：只检索该产品的文档
                product_docs = get_product_docs(agent_product)
                if product_docs:
                    rrf_results = rrf_retrieve(query_text, product_docs, top_k=50)
                    merged = {hash(doc): doc for _, doc in rrf_results}
                    merged_docs = list(merged.values())
                    merged_docs = _filter_short_chunks(merged_docs)
                    reranked = rerank_documents(query_text, merged_docs, top_k=RERANK_TOP_K * 2)
                    final_docs = [doc for _, doc in reranked]
                    if final_docs:
                        print(f"  [Agent检索] 产品 '{agent_product}' → 三路检索 → {len(final_docs)} 个文档")
                        return "\n".join([f"[文档{i+1}] {d}" for i, d in enumerate(final_docs[:50])])
                    else:
                        # RRF 无结果，使用该产品的全部文档（不跨产品搜索）
                        print(f"  [Agent检索] 产品 '{agent_product}' → RRF 无结果，使用该产品全部文档")
                        return "\n".join([f"[文档{i+1}] {d}" for i, d in enumerate(product_docs[:50])])
                else:
                    print(f"  [Agent检索] 产品 '{agent_product}' 无产品限定文档，降级到全局搜索")

            # 降级：全部产品检索
            from rag_tools import search_knowledge_base
            return search_knowledge_base.invoke(query_text)

        # 工具2：发给 DeepSeek 分析（强制基于文档）
        def analyze_with_llm(query_and_docs: str) -> str:
            """将问题和文档发给 DeepSeek 分析，严格基于文档内容"""
            try:
                from openai import OpenAI
                from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
                llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

                product_prefix = f"【产品范围】当前产品：{agent_product}\n\n" if agent_product else ""
                prompt = f"""你是讯飞产品知识库的智能助手。请严格基于以下知识库文档回答问题。

{product_prefix}重要规则：
1. 必须基于文档中实际存在的信息回答，逐个检查每个文档
2. 如果文档中有相关信息，必须详细提取并整理成回答
3. 如果文档中有部分内容与问题相关，也要整理出来，不要因为不完全匹配就忽略
4. 如果文档中确实没有相关信息，列出文档中与查询最相关的片段
5. 绝对不要编造产品参数、功能、价格等信息
6. 回答要详细、完整，不要过于简短

{query_and_docs}

基于文档内容回答："""

                response = llm.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"DeepSeek分析出错: {e}"

        # 工具3：直接获取所有文档（供 plan_and_solve 和 reflection 使用）
        def get_all_docs(query_text: str) -> str:
            """直接获取知识库所有文档内容，供分析使用"""
            try:
                total = collection.count()
                all_docs = []

                if agent_product:
                    # 先尝试按产品过滤
                    where_filter = {"product": agent_product}
                    all_results = collection.get(limit=min(total, 200), where=where_filter)
                    all_docs = all_results["documents"] if all_results["documents"] else []
                    if all_docs:
                        print(f"  [Agent文档] 产品 '{agent_product}' 过滤后 → {len(all_docs)} 个文档块")
                    else:
                        # 产品过滤无结果，降级到获取全部文档
                        print(f"  [Agent文档] 产品 '{agent_product}' 过滤无结果，降级到全部文档")
                        all_results = collection.get(limit=min(total, 200))
                        all_docs = all_results["documents"] if all_results["documents"] else []
                else:
                    all_results = collection.get(limit=min(total, 200))
                    all_docs = all_results["documents"] if all_results["documents"] else []

                # 过滤无关内容
                noise_words = ["隐私", "政策", "注销", "快递", "物流", "顺丰", "个人信息", "JWT", "private String"]
                all_docs = [d for d in all_docs if not any(nw in d for nw in noise_words)]

                print(f"  [Agent文档] 获取 {len(all_docs)} 个文档块")
                return "\n".join([f"[文档{i+1}] {d[:400]}" for i, d in enumerate(all_docs)]) if all_docs else "未找到相关文档"
            except Exception as e:
                return f"获取文档出错: {e}"

        tools = {"search_knowledge": search_knowledge, "analyze_with_llm": analyze_with_llm, "get_all_docs": get_all_docs}

        answer = run_agent(request.query, mode=request.agent_mode, tools=tools, product_context=agent_product)
        return QueryResponse(
            query=request.query,
            mode=f"agent_{request.agent_mode}",
            results=[],
            enhanced_info={
                "steps": [f"Agent模式: {request.agent_mode}"],
                "answer": answer
            }
        )

    if request.mode == "basic":
        filter_dict = {"product": request.product} if request.product else None
        results = basic_search(collection, request.query, k=request.top_k, filter_dict=filter_dict)
        docs = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []

        # 如果检索结果不够，扩大检索范围
        if len(docs) < 3:
            print(f"[基础检索] 结果不足（{len(docs)} 条），扩大检索范围...")
            try:
                # 策略A：扩大 n_results 重新检索
                expanded_results = basic_search(collection, request.query, k=RETRIEVAL_TOP_K * 3, filter_dict=filter_dict)
                expanded_docs = expanded_results["documents"][0] if expanded_results["documents"] else []
                if expanded_docs:
                    seen = set(hash(d) for d in docs)
                    for d in expanded_docs:
                        if hash(d) not in seen:
                            seen.add(hash(d))
                            docs.append(d)
                    print(f"  [基础检索] 扩大后 {len(docs)} 条")
            except Exception as e:
                print(f"  [基础检索] 扩大检索失败: {e}")

            # 策略B：仍不足，获取全部文档（按产品过滤）
            if len(docs) < 3:
                print(f"[基础检索] 仍不足（{len(docs)} 条），获取全部文档")
                all_results = collection.get(limit=200, where=filter_dict)
                all_docs = all_results["documents"] if all_results["documents"] else []
                seen = set(hash(d) for d in docs)
                for d in all_docs:
                    if hash(d) not in seen:
                        seen.add(hash(d))
                        docs.append(d)
                print(f"  [基础检索] 最终 {len(docs)} 条")

        from query_enhancer import generate_answer
        answer = generate_answer(request.query, docs, collection=collection, product=request.product)

        return QueryResponse(
            query=request.query,
            mode="basic",
            results=[
                {"content": doc, "score": round(1.0 / (1.0 + dist), 4)}
                for doc, dist in zip(docs[:5], distances[:5]) if distances
            ],
            enhanced_info={
                "steps": ["基础相似度检索", "LLM生成回答"],
                "answer": answer
            }
        )
    else:
        reranked_docs, answer = enhanced_retrieval(
            request.query, collection, top_k=request.top_k, product=request.product
        )

        return QueryResponse(
            query=request.query,
            mode="enhanced",
            results=[
                {"content": doc, "score": 0}
                for doc in reranked_docs
            ],
            enhanced_info={
                "steps": ["Query重写", "HyDE假设性文档", "多扩展查询", "关键词过滤", "Cross-Encoder重排", "LLM生成回答"],
                "total_candidates": len(reranked_docs),
                "answer": answer
            }
        )


@app.get("/api/stats")
async def stats():
    result = {
        "vector_db_status": "ready" if collection else "not_ready",
        "vector_db_count": collection.count() if collection else 0
    }

    # 尝试获取产品分布
    if collection:
        try:
            all_data = collection.get(limit=1000)
            metas = all_data.get("metadatas", [])
            products = {}
            for m in metas:
                p = m.get("product", "") if m else ""
                if p:
                    products[p] = products.get(p, 0) + 1
            result["product_distribution"] = products
        except Exception:
            pass

    try:
        mysql_stats = get_stats()
        result["mysql"] = mysql_stats
    except Exception:
        result["mysql"] = {"status": "not_connected"}

    return result


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"
    product: str = ""  # 选中的产品类别，为空表示全部产品


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """智能助手对话接口（带记忆）"""
    global collection

    if collection is None:
        raise HTTPException(status_code=400, detail="向量数据库未初始化")

    from smart_assistant import init_assistant, chat_with_assistant
    from conversation_memory import memory

    # 初始化助手
    init_assistant(collection)

    # 对话
    answer = chat_with_assistant(request.query, request.session_id, product=request.product)

    # 获取对话历史
    history = memory.get_messages(request.session_id)
    history_list = []
    for msg in history[-10:]:
        if hasattr(msg, "content"):
            role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
            history_list.append({"role": role, "content": msg.content[:200]})

    return {
        "answer": answer,
        "session_id": request.session_id,
        "history": history_list
    }


@app.post("/api/chat/clear")
async def clear_chat(session_id: str = "default"):
    """清空对话历史"""
    from conversation_memory import memory
    memory.clear(session_id)
    return {"status": "cleared"}


@app.post("/api/chat_stream")
async def chat_stream(request: ChatRequest):
    """流式输出接口（SSE 打字机效果）"""
    global collection

    if collection is None:
        raise HTTPException(status_code=400, detail="向量数据库未初始化")

    from smart_assistant import init_assistant
    init_assistant(collection)

    def generate():
        try:
            # ===== 产品过滤：如果用户选择了产品，只检索该产品的文档 =====
            product_docs = get_product_docs(request.product)
            product_context = ""
            if product_docs:
                product_context = f"当前产品：{request.product}"
                # 使用产品限定的文档进行检索
                from rag_tools import set_collection, search_knowledge_base, list_all_products, rrf_retrieve, _filter_short_chunks, rerank_documents
                from config import RERANK_TOP_K
                set_collection(collection)
                
                # 用产品文档做三路检索
                rrf_results = rrf_retrieve(request.query, product_docs, top_k=50)
                merged = {hash(doc): doc for _, doc in rrf_results}
                merged_docs = list(merged.values())
                merged_docs = _filter_short_chunks(merged_docs)
                reranked = rerank_documents(request.query, merged_docs, top_k=RERANK_TOP_K * 2)
                final_docs = [doc for _, doc in reranked]
                if final_docs:
                    kb_result = "\n".join([f"[文档{i+1}] {d}" for i, d in enumerate(final_docs[:50])])
                    print(f"  [产品限定] {request.product} → 三路检索 → {len(final_docs)} 个文档")
                else:
                    # RRF 检索无结果，降级到该产品的全部文档（不跨产品搜索）
                    print(f"  [产品限定] {request.product} → RRF 检索无结果，使用该产品全部文档")
                    kb_result = "\n".join([f"[文档{i+1}] {d}" for i, d in enumerate(product_docs[:50])])
            else:
                product_context = "全部产品"
                # 获取检索结果
                list_keywords = ["几款", "有哪些", "什么产品", "产品列表", "全部", "所有", "多少"]
                if any(kw in request.query for kw in list_keywords):
                    kb_result = list_all_products.invoke("")
                else:
                    kb_result = search_knowledge_base.invoke(request.query)

            # ===== 兜底策略：检索结果不足时自动扩大范围 =====
            doc_count = kb_result.count("[文档") if kb_result else 0
            if doc_count < 3 and "检索出错" not in kb_result and "知识库未初始化" not in kb_result:
                print(f"[聊天-兜底] 检索结果不足（{doc_count} 条），扩大检索范围...")
                try:
                    # 策略A：获取更多文档（优先按产品过滤）
                    total = collection.count()
                    if request.product:
                        all_results = collection.get(limit=min(total, 200), where={"product": request.product})
                    else:
                        all_results = collection.get(limit=min(total, 200))
                    all_docs = all_results["documents"] if all_results["documents"] else []
                    # 过滤噪音文档
                    from document_processor import _is_valid_chunk
                    valid_docs = [d for d in all_docs if _is_valid_chunk(d)]
                    if valid_docs:
                        print(f"  [聊天-兜底] 获取到 {len(valid_docs)} 个有效文档块")
                        # 简单关键词匹配排序
                        import jieba
                        keywords = list(jieba.analyse.extract_tags(request.query, topK=10))
                        if keywords:
                            scored = []
                            for doc in valid_docs:
                                score = sum(1 for kw in keywords if kw in doc)
                                if score > 0:
                                    scored.append((score, doc))
                            scored.sort(key=lambda x: x[0], reverse=True)
                            valid_docs = [d for _, d in scored][:30]
                        kb_result = "\n".join([f"[文档{i+1}] {d}" for i, d in enumerate(valid_docs[:30])])
                except Exception as e:
                    print(f"  [聊天-兜底] 扩大检索失败: {e}")

            # 获取对话历史
            from conversation_memory import memory
            history_context = memory.get_context(request.session_id, last_n=5)

            yield f"data: {__import__('json').dumps({'type': 'status', 'content': '正在分析...'})}\n\n"

            # 流式调用 DeepSeek
            from openai import OpenAI
            from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
            llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

            from prompt_templates import template_manager
            prompt = template_manager.render("chat",
                history_context=history_context,
                kb_result=kb_result,
                query=request.query,
                product_context=product_context
            )

            response = llm.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3,
                stream=True
            )

            full_answer = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_answer += text
                    yield f"data: {__import__('json').dumps({'type': 'chunk', 'content': text})}\n\n"

            # 保存对话记录
            memory.add_user_message(request.query, request.session_id)
            memory.add_ai_message(full_answer, request.session_id)

            # ===== 视频推荐 =====
            try:
                video_product = request.product
                video_model = ""
                # 从查询中提取产品上下文
                recommended_videos = video_store.recommend_videos(
                    query=request.query,
                    product_line=video_product,
                    product_model=video_model,
                    top_k=3
                )
                if recommended_videos:
                    video_list = [v.to_dict() for v in recommended_videos]
                    yield f"data: {__import__('json').dumps({'type': 'videos_recommended', 'videos': video_list})}\n\n"
            except Exception as e:
                print(f"  [视频推荐] 失败: {e}")

            yield f"data: {__import__('json').dumps({'type': 'done', 'content': ''})}\n\n"

        except Exception as e:
            yield f"data: {__import__('json').dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/evaluate")
async def evaluate_query(request: EvaluateRequest):
    """单个查询评估"""
    global collection

    if collection is None:
        raise HTTPException(status_code=400, detail="向量数据库未初始化")

    from ragas_evaluator import get_evaluator

    evaluator = get_evaluator(collection)
    result = evaluator.evaluate_single(
        query=request.query,
        ground_truth=request.ground_truth,
        top_k=request.top_k
    )

    return result


@app.post("/api/evaluate/batch")
async def evaluate_batch(request: BatchEvaluateRequest):
    """批量评估"""
    global collection

    if collection is None:
        raise HTTPException(status_code=400, detail="向量数据库未初始化")

    from ragas_evaluator import get_evaluator, DEFAULT_TEST_CASES

    evaluator = get_evaluator(collection)

    # 如果没有提供测试用例，使用默认用例
    test_cases = request.test_cases if request.test_cases else DEFAULT_TEST_CASES

    result = evaluator.evaluate_batch(test_cases, top_k=request.top_k)

    # 保存报告
    report_path = evaluator.save_report(result)
    json_path = evaluator.save_results_json(result)

    return {
        **result,
        "report_path": report_path,
        "json_path": json_path
    }


@app.post("/api/evaluate/auto-generate")
async def auto_generate_testcases():
    """AI根据知识库自动生成测试用例"""
    global collection

    if collection is None:
        raise HTTPException(status_code=400, detail="向量数据库未初始化，请先上传文档")

    try:
        # 获取知识库所有文档摘要
        total = collection.count()
        all_results = collection.get(limit=min(total, 100))
        all_docs = all_results["documents"] if all_results["documents"] else []

        if not all_docs:
            raise HTTPException(status_code=400, detail="知识库为空，请先上传文档")

        # 提取文档摘要（每个文档取前300字符）
        doc_summaries = []
        seen = set()
        for doc in all_docs:
            short = doc[:300]
            h = hash(short)
            if h not in seen:
                seen.add(h)
                doc_summaries.append(short)

        docs_text = "\n".join([f"[文档{i+1}] {s}" for i, s in enumerate(doc_summaries[:30])])

        # 用 DeepSeek 生成测试用例
        from openai import OpenAI
        from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
        llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

        prompt = f"""你是测试工程师。根据以下知识库文档内容，生成5个测试用例，用于验证RAG检索系统的质量。

知识库文档摘要：
{docs_text}

要求：
1. 每个测试用例包含：query（测试问题）、ground_truth（标准答案）
2. 问题要多样化：有的问具体产品参数、有的问功能对比、有的问产品列表、有的问使用方法
3. 标准答案必须基于文档中的实际内容，不要编造
4. 输出严格的JSON数组格式，不要包含任何其他文字

只输出JSON数组：
[
  {{"query": "测试问题1", "ground_truth": "标准答案1"}},
  {{"query": "测试问题2", "ground_truth": "标准答案2"}},
  {{"query": "测试问题3", "ground_truth": "标准答案3"}},
  {{"query": "测试问题4", "ground_truth": "标准答案4"}},
  {{"query": "测试问题5", "ground_truth": "标准答案5"}}
]"""

        response = llm.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.3
        )

        result_text = response.choices[0].message.content.strip()
        print(f"[AI用例] 原始返回: {result_text[:200]}...")

        # 清理 markdown 标记
        if result_text.startswith("```"):
            result_text = result_text.split("\n", 1)[1]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            result_text = result_text.strip()

        import json
        test_cases = json.loads(result_text)

        if not isinstance(test_cases, list) or len(test_cases) == 0:
            raise Exception("AI返回的不是有效的JSON数组")

        print(f"[AI用例] 成功生成 {len(test_cases)} 个测试用例")
        return {"test_cases": test_cases, "source_docs_count": len(doc_summaries)}

    except json.JSONDecodeError as e:
        print(f"[AI用例] JSON解析失败: {e}")
        print(f"[AI用例] 原始文本: {result_text[:500]}")
        raise HTTPException(status_code=500, detail=f"AI返回的内容无法解析为JSON: {result_text[:200]}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[AI用例] 生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成测试用例失败: {str(e)}")


@app.get("/api/evaluate/reports")
async def list_reports():
    """获取所有评估报告列表"""
    output_dir = OUTPUT_DIR

    if not os.path.exists(output_dir):
        return {"reports": []}

    report_files = sorted(
        [f for f in os.listdir(output_dir) if f.startswith("ragas_report_")],
        reverse=True
    )

    reports = []
    for f in report_files:
        filepath = os.path.join(output_dir, f)
        stat = os.stat(filepath)
        # 从文件名提取时间: ragas_report_20260722_141513.md
        time_str = f.replace("ragas_report_", "").replace(".md", "")
        try:
            from datetime import datetime
            dt = datetime.strptime(time_str, "%Y%m%d_%H%M%S")
            display_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            display_time = time_str

        reports.append({
            "filename": f,
            "display_time": display_time,
            "size": f"{stat.st_size / 1024:.1f}KB"
        })

    return {"reports": reports}


@app.get("/api/evaluate/report")
async def get_report(filename: str = None):
    """获取指定评估报告内容，不传则获取最新"""
    output_dir = OUTPUT_DIR

    if not os.path.exists(output_dir):
        return {"error": "无评估报告"}

    if filename:
        # 获取指定报告
        report_path = os.path.join(output_dir, filename)
        if not os.path.exists(report_path):
            return {"error": f"报告 {filename} 不存在"}
    else:
        # 获取最新报告
        report_files = sorted([f for f in os.listdir(output_dir) if f.startswith("ragas_report_")])
        if not report_files:
            return {"error": "无评估报告"}
        filename = report_files[-1]
        report_path = os.path.join(output_dir, filename)

    with open(report_path, "r", encoding="utf-8") as f:
        report_content = f.read()

    return {
        "filename": filename,
        "content": report_content
    }


@app.get("/api/files")
async def list_files():
    """获取已上传的文件列表"""
    upload_dir = "./uploads"
    output_dir = "./output"

    files = []
    total_chunks = 0

    # 扫描 uploads 目录
    if os.path.exists(upload_dir):
        for f in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, f)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                ext = os.path.splitext(f)[1].lower().replace(".", "")
                files.append({
                    "name": f,
                    "type": ext,
                    "size": f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/1024/1024:.1f}MB",
                    "source": "uploads"
                })

    # 统计分块数
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            if f.endswith("_chunks.txt"):
                chunk_path = os.path.join(output_dir, f)
                try:
                    with open(chunk_path, "r", encoding="utf-8") as fp:
                        content = fp.read()
                        total_chunks += content.count("--- Chunk ")
                except Exception:
                    pass

    # 统计向量数据库
    if collection:
        total_chunks = collection.count()

    return {"files": files, "chunks": total_chunks}


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传文档并处理"""
    upload_dir = "./uploads"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    chunks = process_documents([file_path])

    if not chunks:
        raise HTTPException(status_code=400, detail="文档处理失败")

    # 保存 Markdown 文件
    save_markdown(chunks)

    all_texts = list(chunks.values())[0] if chunks else []

    global collection, embedding_fn
    if collection is None:
        collection, embedding_fn = create_vectorstore(all_texts)
    else:
        batch_size = 500
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i + batch_size]
            ids = [f"upload_{file.filename}_{j}" for j in range(i, i + len(batch))]
            collection.add(documents=batch, ids=ids)

    return {
        "status": "success",
        "filename": file.filename,
        "chunks_count": len(all_texts)
    }


# ========== 视频知识库 API ==========


@app.get("/api/videos")
async def list_videos(
    product_line: str = "",
    product_model: str = "",
    status: str = ""
):
    """列出视频，支持按产品线/型号/状态过滤"""
    videos = video_store.list_videos(
        product_line=product_line,
        product_model=product_model,
        status=status
    )
    return {
        "total": len(videos),
        "videos": [v.to_dict() for v in videos]
    }


@app.get("/api/videos/search")
async def search_videos(query: str = "", product_line: str = ""):
    """按关键词搜索视频"""
    if not query:
        return {"total": 0, "videos": []}
    videos = video_store.search_videos(query, product_line=product_line)
    return {
        "total": len(videos),
        "videos": [v.to_dict() for v in videos]
    }


@app.get("/api/videos/recommend")
async def recommend_videos(
    query: str = "",
    product_line: str = "",
    product_model: str = "",
    top_k: int = 3
):
    """根据用户问题推荐相关视频（12.3节召回规则）"""
    if not query:
        return {"total": 0, "videos": []}
    videos = video_store.recommend_videos(
        query=query,
        product_line=product_line,
        product_model=product_model,
        top_k=top_k
    )
    return {
        "total": len(videos),
        "videos": [v.to_dict() for v in videos]
    }


@app.get("/api/videos/stats")
async def video_stats():
    """视频库统计"""
    return video_store.get_stats()


@app.get("/api/videos/{video_id}")
async def get_video(video_id: str):
    """获取视频详情"""
    video = video_store.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="视频不存在")
    return video.to_dict()


@app.post("/api/videos")
async def create_video(request: VideoCreateRequest):
    """添加新视频"""
    chapters = [
        VideoChapter(c["start_time"], c["end_time"], c["title"])
        for c in request.chapters
    ] if request.chapters else []

    video = Video(
        video_id=request.video_id,
        product_line=request.product_line,
        product_model=request.product_model,
        firmware_version=request.firmware_version,
        title=request.title,
        duration=request.duration,
        video_url=request.video_url,
        thumbnail_url=request.thumbnail_url,
        applicable_questions=request.applicable_questions,
        chapters=chapters,
        review_status=request.review_status,
    )
    video_store.add_video(video)
    return video.to_dict()


@app.put("/api/videos/{video_id}")
async def update_video(video_id: str, request: VideoUpdateRequest):
    """更新视频信息"""
    # 只传非空字段
    updates = {k: v for k, v in request.dict().items() if v is not None}
    video = video_store.update_video(video_id, updates)
    if not video:
        raise HTTPException(status_code=404, detail="视频不存在")
    return video.to_dict()


@app.delete("/api/videos/{video_id}")
async def delete_video(video_id: str):
    """删除视频"""
    success = video_store.delete_video(video_id)
    if not success:
        raise HTTPException(status_code=404, detail="视频不存在")
    return {"status": "deleted", "video_id": video_id}


# ---- 章节管理 ----

@app.post("/api/videos/{video_id}/chapters")
async def add_chapter(video_id: str, request: ChapterCreateRequest):
    """添加视频章节"""
    chapter = VideoChapter(request.start_time, request.end_time, request.title)
    video = video_store.add_chapter(video_id, chapter)
    if not video:
        raise HTTPException(status_code=404, detail="视频不存在")
    return video.to_dict()


@app.put("/api/videos/{video_id}/chapters/{chapter_index}")
async def update_chapter(video_id: str, chapter_index: int, request: ChapterUpdateRequest):
    """更新视频章节"""
    # 获取当前章节
    video = video_store.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="视频不存在")
    if chapter_index < 0 or chapter_index >= len(video.chapters):
        raise HTTPException(status_code=404, detail="章节索引无效")

    current = video.chapters[chapter_index]
    updated_chapter = VideoChapter(
        start_time=request.start_time if request.start_time is not None else current.start_time,
        end_time=request.end_time if request.end_time is not None else current.end_time,
        title=request.title if request.title is not None else current.title,
    )
    video = video_store.update_chapter(video_id, chapter_index, updated_chapter)
    return video.to_dict()


@app.delete("/api/videos/{video_id}/chapters/{chapter_index}")
async def delete_chapter(video_id: str, chapter_index: int):
    """删除视频章节"""
    video = video_store.delete_chapter(video_id, chapter_index)
    if not video:
        raise HTTPException(status_code=404, detail="视频或章节不存在")
    return video.to_dict()


# ========== OpenMAIC 视频教程集成 API ==========

OPENMAIC_BASE_URL = "http://localhost:3001"


class GenerateTutorialRequest(BaseModel):
    query: str
    product: str = ""
    product_model: str = ""
    scene_type: str = "hardware_action"  # 场景模板类型（10.4节）
    use_sop: bool = False  # 是否使用 SOP 结构化生成


@app.post("/api/generate-tutorial")
async def generate_tutorial(request: GenerateTutorialRequest):
    """
    调用 OpenMAIC 生成产品使用教程视频
    支持两种模式：
    1. 基础模式（use_sop=False）：直接传 requirement
    2. SOP 模式（use_sop=True）：先生成 SOP，再构建场景序列
    """
    global collection

    if collection is None:
        raise HTTPException(status_code=400, detail="向量数据库未初始化")

    # 构建产品描述
    product_desc = request.product
    if request.product_model:
        product_desc += f" {request.product_model}"

    # 从知识库检索该产品的相关信息
    product_docs = get_product_docs(request.product)
    doc_context = ""
    if product_docs:
        import jieba
        keywords = list(jieba.analyse.extract_tags(request.query, topK=10))
        scored = []
        for doc in product_docs:
            score = sum(1 for kw in keywords if kw in doc)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_docs = [d for _, d in scored[:10]] if scored else product_docs[:10]
        doc_context = "\n".join([f"[参考{i+1}] {d[:500]}" for i, d in enumerate(top_docs)])

    # 构建 OpenMAIC 请求体
    if request.use_sop and doc_context:
        # SOP 模式：生成结构化 SOP → 场景序列
        try:
            from sop_generator import generate_sop_from_docs, build_openmaic_request
            sop = generate_sop_from_docs(request.query, product_desc, doc_context)
            openmaic_body = build_openmaic_request(sop, scene_type=request.scene_type)
            print(f"  [SOP模式] 生成 SOP: {sop.sop_id} → {len(sop.steps)} 个步骤")
        except Exception as e:
            print(f"  [SOP模式] 失败，降级为基础模式: {e}")
            openmaic_body = {
                "requirement": f"生成{product_desc}的使用教程。用户问题：{request.query}\n\n产品知识库参考信息：\n{doc_context}",
                "enableVideoGeneration": False,
                "enableTTS": False,
                "enableImageGeneration": True,
                "enableWebSearch": False,
            }
    else:
        # 基础模式
        requirement = f"生成{product_desc}的使用教程。用户问题：{request.query}"
        if doc_context:
            requirement += f"\n\n产品知识库参考信息：\n{doc_context}"
        openmaic_body = {
            "requirement": requirement,
            "enableVideoGeneration": False,
            "enableTTS": False,
            "enableImageGeneration": True,
            "enableWebSearch": False,
        }

    # 调用 OpenMAIC API
    try:
        import httpx
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{OPENMAIC_BASE_URL}/api/generate-classroom",
                json=openmaic_body,
            )
            result = resp.json()

            if not result.get("success"):
                raise HTTPException(
                    status_code=500,
                    detail=f"OpenMAIC 调用失败: {result.get('error', '未知错误')}"
                )

            return {
                "job_id": result["jobId"],
                "status": result["status"],
                "poll_url": result["pollUrl"],
                "poll_interval_ms": result.get("pollIntervalMs", 5000),
                "product": request.product,
                "product_model": request.product_model,
                "query": request.query,
            }
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"无法连接到 OpenMAIC 服务 ({OPENMAIC_BASE_URL}): {str(e)}"
        )


@app.get("/api/generate-tutorial/{job_id}")
async def get_tutorial_status(job_id: str):
    """查询视频教程生成任务状态"""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{OPENMAIC_BASE_URL}/api/generate-classroom/{job_id}"
            )
            return resp.json()
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"无法连接到 OpenMAIC 服务: {str(e)}"
        )


# ========== 反馈收集 API ==========

FEEDBACK_FILE = os.path.join(BASE_DIR, "feedback_data.json")


class FeedbackRequest(BaseModel):
    session_id: str = ""
    query: str = ""
    answer: str = ""
    rating: int  # 1 = 赞, -1 = 踩
    comment: str = ""


@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """提交用户反馈（点赞/点踩）"""
    try:
        # 加载现有反馈数据
        feedbacks = []
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                feedbacks = json.load(f)

        # 添加新反馈
        feedback_entry = {
            "id": f"FB-{uuid.uuid4().hex[:8].upper()}",
            "session_id": request.session_id,
            "query": request.query[:200],
            "answer": request.answer[:500],
            "rating": request.rating,
            "comment": request.comment,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        feedbacks.append(feedback_entry)

        # 保存
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(feedbacks, f, ensure_ascii=False, indent=2)

        return {"status": "success", "id": feedback_entry["id"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存反馈失败: {str(e)}")


@app.get("/api/feedback/stats")
async def get_feedback_stats():
    """获取反馈统计"""
    try:
        if not os.path.exists(FEEDBACK_FILE):
            return {"total": 0, "positive": 0, "negative": 0, "rate": 0}
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            feedbacks = json.load(f)
        total = len(feedbacks)
        positive = sum(1 for fb in feedbacks if fb.get("rating") == 1)
        negative = sum(1 for fb in feedbacks if fb.get("rating") == -1)
        rate = round(positive / total * 100, 1) if total > 0 else 0
        return {"total": total, "positive": positive, "negative": negative, "rate": rate}
    except Exception as e:
        return {"total": 0, "positive": 0, "negative": 0, "rate": 0, "error": str(e)}


# ========== 文档分类标签 API（6.1节） ==========

@app.get("/api/doc-types")
async def get_doc_types():
    """获取文档类型列表"""
    return {"types": [{"name": k, "description": v} for k, v in DOCUMENT_TYPES.items()]}


@app.post("/api/classify")
async def classify_doc(request: dict):
    """分类文档内容"""
    if "text" not in request:
        raise HTTPException(status_code=400, detail="缺少text字段")
    result = classify_document(request["text"], request.get("title", ""))
    return {"document_type": result}


# ========== 选题库 API（7.1/7.2节） ==========

@app.get("/api/topics")
async def list_topics(product_line: str = "", status: str = "", sort_by: str = "priority"):
    """列出选题"""
    topics = topic_pool.list_topics(product_line, status, sort_by)
    return {"total": len(topics), "topics": [t.to_dict() for t in topics]}


@app.get("/api/topics/stats")
async def topic_stats():
    """选题库统计"""
    return topic_pool.get_stats()


@app.get("/api/topics/{topic_id}")
async def get_topic(topic_id: str):
    """获取选题详情"""
    topic = topic_pool.get_topic(topic_id)
    if not topic:
        raise HTTPException(status_code=404, detail="选题不存在")
    return topic.to_dict()


@app.post("/api/topics")
async def create_topic(request: dict):
    """创建选题"""
    topic = VideoTopic(
        title=request.get("title", ""),
        product_line=request.get("product_line", ""),
        product_model=request.get("product_model", ""),
        source=request.get("source", "manual_input"),
        source_detail=request.get("source_detail", ""),
        query_count=request.get("query_count", 0),
        difficulty=request.get("difficulty", "中"),
        estimated_duration=request.get("estimated_duration", 60),
    )
    return topic_pool.add_topic(topic).to_dict()


@app.put("/api/topics/{topic_id}")
async def update_topic(topic_id: str, request: dict):
    """更新选题"""
    topic = topic_pool.update_topic(topic_id, request)
    if not topic:
        raise HTTPException(status_code=404, detail="选题不存在")
    return topic.to_dict()


@app.delete("/api/topics/{topic_id}")
async def delete_topic(topic_id: str):
    """删除选题"""
    if not topic_pool.delete_topic(topic_id):
        raise HTTPException(status_code=404, detail="选题不存在")
    return {"status": "deleted"}


# ========== 产品线视频模板 API（3.1~3.4节） ==========

@app.get("/api/product-templates")
async def get_product_templates(product_line: str = ""):
    """获取产品线视频模板"""
    if product_line:
        tmpl = get_product_template(product_line)
        if not tmpl:
            raise HTTPException(status_code=404, detail="产品线不存在")
        return {"product_line": product_line, "template": tmpl}
    return {"templates": PRODUCT_LINE_TEMPLATES}


# ========== SOP 生成 API（8.1/8.2节） ==========

@app.get("/api/sop/scene-templates")
async def get_sop_scene_templates():
    """获取场景模板列表"""
    from sop_generator import SCENE_TEMPLATES
    return {"templates": {k: v for k, v in SCENE_TEMPLATES.items()}}


@app.post("/api/sop/generate")
async def generate_sop(request: dict):
    """从知识库文档生成 SOP"""
    from sop_generator import generate_sop_from_docs, sop_to_openmaic_scenes, build_openmaic_request
    query = request.get("query", "")
    product_desc = request.get("product_desc", "")
    doc_context = request.get("doc_context", "")
    scene_type = request.get("scene_type", "hardware_action")

    if not query or not product_desc:
        raise HTTPException(status_code=400, detail="缺少 query 或 product_desc 字段")

    sop = generate_sop_from_docs(query, product_desc, doc_context)
    scenes = sop_to_openmaic_scenes(sop, scene_type)
    openmaic_body = build_openmaic_request(sop, scene_type)

    return {
        "sop": sop.to_dict(),
        "scenes": scenes,
        "openmaic_request": openmaic_body,
    }


@app.post("/api/sop/convert")
async def convert_sop_to_scenes(request: dict):
    """将 SOP 转换为 OpenMAIC 场景序列"""
    from sop_generator import SOPStep, SOP, SCENE_TEMPLATES, sop_to_openmaic_scenes
    sop_data = request.get("sop", {})
    scene_type = request.get("scene_type", "hardware_action")

    # 重建 SOP 对象
    steps = []
    for s in sop_data.get("steps", []):
        steps.append(SOPStep(
            step_no=s.get("step_no", 1),
            action=s.get("action", ""),
            operation_target=s.get("operation_target", ""),
            expected_result=s.get("expected_result", ""),
            visual_requirement=s.get("visual_requirement", ""),
            source_document=s.get("source_document", ""),
            source_page=s.get("source_page", 0),
            warning=s.get("warning", ""),
        ))

    sop = SOP(
        sop_id=sop_data.get("sop_id", ""),
        product_line=sop_data.get("product_line", ""),
        product_model=sop_data.get("product_model", ""),
        title=sop_data.get("title", ""),
        applicable_version=sop_data.get("applicable_version", ""),
        prerequisites=sop_data.get("prerequisites", []),
        warnings=sop_data.get("warnings", []),
        steps=steps,
        completion_check=sop_data.get("completion_check", []),
        common_errors=sop_data.get("common_errors", []),
        created_at=sop_data.get("created_at", ""),
        updated_at=sop_data.get("updated_at", ""),
    )

    scenes = sop_to_openmaic_scenes(sop, scene_type)
    return {"scenes": scenes, "scene_count": len(scenes)}


# ========== 流水线 API（2.3节） ==========

@app.get("/api/pipeline")
async def list_pipeline_tasks(status: str = "", product_line: str = ""):
    """列出流水线任务"""
    tasks = pipeline.list_tasks(status, product_line)
    return {"total": len(tasks), "tasks": [t.to_dict() for t in tasks]}


@app.get("/api/pipeline/stats")
async def pipeline_stats():
    """流水线统计"""
    return pipeline.get_stats()


@app.get("/api/pipeline/{task_id}")
async def get_pipeline_task(task_id: str):
    """获取流水线任务详情"""
    task = pipeline.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task.to_dict()


@app.post("/api/pipeline")
async def create_pipeline_task(request: dict):
    """创建流水线任务"""
    task = pipeline.create_task(
        title=request.get("title", ""),
        product_line=request.get("product_line", ""),
        product_model=request.get("product_model", ""),
        query=request.get("query", ""),
        topic_id=request.get("topic_id", ""),
    )
    return task.to_dict()


@app.put("/api/pipeline/{task_id}/advance")
async def advance_pipeline_stage(task_id: str, request: dict):
    """
    推进流水线阶段
    自动执行逻辑：
    - 推进到 sop_generation：自动从知识库检索并生成 SOP
    - 推进到 openmaic_generation：自动调用 OpenMAIC 生成视频
    """
    next_stage = request.get("stage", "")
    updates = request.get("updates", {})

    # 获取当前任务
    task = pipeline.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    # ========== 自动生成 SOP ==========
    if next_stage == "sop_generation":
        try:
            from sop_generator import generate_sop_from_docs
            product_desc = f"{task.product_line} {task.product_model}".strip()
            query = task.query or task.title
            doc_context = task.doc_context or ""

            # 如果还没有文档上下文，从知识库检索
            if not doc_context:
                product_docs = get_product_docs(task.product_line)
                if product_docs:
                    import jieba
                    import jieba.analyse
                    keywords = list(jieba.analyse.extract_tags(query, topK=10))
                    scored = []
                    for doc in product_docs:
                        score = sum(1 for kw in keywords if kw in doc)
                        if score > 0:
                            scored.append((score, doc))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    top_docs = [d for _, d in scored[:10]] if scored else product_docs[:10]
                    doc_context = "\n".join([f"[参考{i+1}] {d[:500]}" for i, d in enumerate(top_docs)])
                    updates["doc_context"] = doc_context

            if doc_context:
                sop = generate_sop_from_docs(query, product_desc, doc_context)
                updates["sop_data"] = sop.to_dict()
                print(f"  [流水线] 自动生成 SOP: {sop.sop_id} → {len(sop.steps)} 个步骤")
            else:
                print(f"  [流水线] 无法获取文档上下文，跳过 SOP 自动生成")
        except Exception as e:
            print(f"  [流水线] SOP 自动生成失败: {e}")

    # ========== 自动调用 OpenMAIC 生成视频 ==========
    elif next_stage == "openmaic_generation":
        try:
            import httpx

            # 根据是否有 SOP 数据选择模式
            if task.sop_data:
                from sop_generator import SOP, SOPStep, build_openmaic_request
                sd = task.sop_data
                steps = []
                for s in sd.get("steps", []):
                    steps.append(SOPStep(
                        step_no=s.get("step_no", 1),
                        action=s.get("action", ""),
                        operation_target=s.get("operation_target", ""),
                        expected_result=s.get("expected_result", ""),
                        visual_requirement=s.get("visual_requirement", ""),
                        source_document=s.get("source_document", ""),
                        source_page=s.get("source_page", 0),
                        warning=s.get("warning", ""),
                    ))
                sop = SOP(
                    sop_id=sd.get("sop_id", ""),
                    product_line=sd.get("product_line", task.product_line),
                    product_model=sd.get("product_model", task.product_model),
                    title=sd.get("title", task.title),
                    applicable_version=sd.get("applicable_version", ""),
                    prerequisites=sd.get("prerequisites", []),
                    warnings=sd.get("warnings", []),
                    steps=steps,
                    completion_check=sd.get("completion_check", []),
                    common_errors=sd.get("common_errors", []),
                )
                openmaic_body = build_openmaic_request(sop)
                # 启用视频生成
                openmaic_body["enableVideoGeneration"] = True
                openmaic_body["enableTTS"] = True
                print(f"  [流水线] SOP 模式构建 OpenMAIC 请求: {sop.sop_id}")
            else:
                # 基础模式
                product_desc = f"{task.product_line} {task.product_model}".strip()
                openmaic_body = {
                    "requirement": f"生成{product_desc}的{task.title}教学视频",
                    "enableVideoGeneration": True,
                    "enableTTS": True,
                    "enableImageGeneration": True,
                    "enableWebSearch": False,
                }
                print(f"  [流水线] 基础模式构建 OpenMAIC 请求")

            # 调用 OpenMAIC API
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{OPENMAIC_BASE_URL}/api/generate-classroom",
                    json=openmaic_body,
                )
                result = resp.json()

                if result.get("success"):
                    updates["openmaic_job_id"] = result.get("jobId", "")
                    updates["openmaic_result"] = {
                        "job_id": result.get("jobId"),
                        "status": result.get("status"),
                        "poll_url": result.get("pollUrl"),
                        "poll_interval_ms": result.get("pollIntervalMs", 5000),
                    }
                    print(f"  [流水线] OpenMAIC 任务已创建: {result.get('jobId')}")
                else:
                    print(f"  [流水线] OpenMAIC 调用失败: {result.get('error', '未知错误')}")
        except httpx.RequestError as e:
            print(f"  [流水线] OpenMAIC 连接失败: {e}")
        except Exception as e:
            print(f"  [流水线] OpenMAIC 自动调用异常: {e}")

    # 推进阶段
    task = pipeline.advance_stage(task_id, next_stage, updates)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task.to_dict()


@app.post("/api/pipeline/{task_id}/complete")
async def complete_pipeline_task(task_id: str, request: dict = None):
    """完成流水线任务"""
    updates = request.dict() if request else None
    task = pipeline.complete_task(task_id, updates)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task.to_dict()


# ========== 审核与发布中心 API ==========

@app.get("/api/reviews")
async def list_reviews(status: str = "", target_type: str = "", product_line: str = ""):
    """列出审核记录"""
    records = review_center.list_records(status, target_type, product_line)
    return {"total": len(records), "records": [r.to_dict() for r in records]}


@app.get("/api/reviews/types")
async def get_review_types():
    """获取审核类型定义"""
    return {"types": REVIEW_TYPES}


@app.get("/api/reviews/stats")
async def review_stats():
    """审核中心统计"""
    return review_center.get_stats()


@app.get("/api/reviews/{review_id}")
async def get_review(review_id: str):
    """获取审核详情"""
    record = review_center.get_record(review_id)
    if not record:
        raise HTTPException(status_code=404, detail="审核记录不存在")
    return record.to_dict()


@app.post("/api/reviews")
async def create_review(request: dict):
    """创建审核任务"""
    record = review_center.create_review(
        target_type=request.get("target_type", ""),
        target_id=request.get("target_id", ""),
        target_title=request.get("target_title", ""),
        product_line=request.get("product_line", ""),
        product_model=request.get("product_model", ""),
    )
    return record.to_dict()


@app.post("/api/reviews/{review_id}/submit")
async def submit_review(review_id: str, request: dict):
    """提交审核结果"""
    record = review_center.submit_review(
        review_id=review_id,
        review_type=request.get("review_type", ""),
        status=request.get("status", ""),
        reviewer=request.get("reviewer", ""),
        comment=request.get("comment", ""),
    )
    if not record:
        raise HTTPException(status_code=404, detail="审核记录不存在")
    return record.to_dict()


@app.post("/api/reviews/{review_id}/publish")
async def publish_review(review_id: str):
    """发布"""
    record = review_center.publish(review_id)
    if not record:
        raise HTTPException(status_code=400, detail="无法发布，可能未通过审核")
    return record.to_dict()


@app.post("/api/reviews/{review_id}/unpublish")
async def unpublish_review(review_id: str):
    """下架"""
    record = review_center.unpublish(review_id)
    if not record:
        raise HTTPException(status_code=400, detail="无法下架")
    return record.to_dict()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
