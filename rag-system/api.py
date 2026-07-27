"""
RAG 智能检索系统 - 后端 API
FastAPI 提供 RESTful 接口
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from config import DOCS_DIR, CHROMA_PERSIST_DIR, RETRIEVAL_TOP_K, RERANK_TOP_K, OUTPUT_DIR
from document_processor import process_documents, save_chunks, save_markdown
from embedding_store import create_vectorstore, load_vectorstore, basic_search
from query_enhancer import enhanced_retrieval
from mysql_store import init_database, save_chunks_to_mysql, get_stats

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


class QueryRequest(BaseModel):
    query: str
    mode: str = "enhanced"
    top_k: int = 3
    agent_mode: str = "none"  # none / react / plan_and_solve / reflection


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


@app.on_event("startup")
async def startup():
    global collection, embedding_fn

    print("正在初始化 RAG 系统...")

    # 检查是否已有向量数据库
    if os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        print("加载已有向量数据库...")
        try:
            collection, embedding_fn = load_vectorstore()
            print("加载成功")
            return
        except Exception:
            print("加载失败，重新构建...")

    # 处理文档并构建向量数据库
    print("处理文档...")

    # 同时处理 DOCS_DIR 和 uploads 目录
    all_doc_paths = list(DOCS_DIR) if DOCS_DIR else []
    uploads_dir = "./uploads"
    if os.path.exists(uploads_dir):
        for f in os.listdir(uploads_dir):
            file_path = os.path.join(uploads_dir, f)
            if os.path.isfile(file_path):
                all_doc_paths.append(file_path)

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

    # 向量化存储
    all_texts = []
    for chunks in all_chunks.values():
        all_texts.extend(chunks)

    collection, embedding_fn = create_vectorstore(all_texts)

    # 尝试存入 MySQL
    try:
        init_database()
        save_chunks_to_mysql(all_chunks)
    except Exception as e:
        print(f"MySQL 存储跳过: {e}")

    print("初始化完成！")


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")


@app.get("/evaluate")
async def evaluate_page():
    return FileResponse("frontend/evaluate.html")


@app.post("/api/query")
async def query(request: QueryRequest):
    global collection

    if collection is None:
        raise HTTPException(status_code=400, detail="向量数据库未初始化，请先处理文档")

    # 如果选择了 Agent 模式，使用 Agent 范式
    if request.agent_mode and request.agent_mode != "none":
        from agent_paradigms import run_agent

        # 工具：直接复用 rag_tools 的检索逻辑（和聊天系统一致）
        def search_knowledge(query_text: str) -> str:
            """从知识库检索相关信息"""
            from rag_tools import set_collection, search_knowledge_base
            set_collection(collection)
            return search_knowledge_base.invoke(query_text)

        # 工具2：发给 DeepSeek 分析（强制基于文档）
        def analyze_with_llm(query_and_docs: str) -> str:
            """将问题和文档发给 DeepSeek 分析，严格基于文档内容"""
            try:
                from openai import OpenAI
                from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
                llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

                prompt = f"""你是讯飞产品知识库的智能助手。请严格基于以下知识库文档回答问题。

重要规则：
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
                all_results = collection.get(limit=min(total, 200))
                all_docs = all_results["documents"] if all_results["documents"] else []

                # 过滤无关内容
                noise_words = ["隐私", "政策", "注销", "快递", "物流", "顺丰", "个人信息", "JWT", "private String"]
                product_docs = [d for d in all_docs if not any(nw in d for nw in noise_words)]
                if not product_docs:
                    product_docs = all_docs

                print(f"  [全部文档] 获取 {len(product_docs)} 个文档块")
                return "\n".join([f"[文档{i+1}] {d[:400]}" for i, d in enumerate(product_docs)])
            except Exception as e:
                return f"获取文档出错: {e}"

        tools = {"search_knowledge": search_knowledge, "analyze_with_llm": analyze_with_llm, "get_all_docs": get_all_docs}

        answer = run_agent(request.query, mode=request.agent_mode, tools=tools)
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
        results = basic_search(collection, request.query, k=request.top_k)
        docs = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []

        # 如果检索结果不够，获取所有文档发给 DeepSeek
        if len(docs) < 2:
            print("[基础检索] 结果不足，获取所有文档")
            all_results = collection.get(limit=200)
            docs = all_results["documents"] if all_results["documents"] else []

        from query_enhancer import generate_answer
        answer = generate_answer(request.query, docs, collection=collection)

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
            request.query, collection, top_k=request.top_k
        )

        return QueryResponse(
            query=request.query,
            mode="enhanced",
            results=[
                {"content": doc, "score": 0}
                for doc in reranked_docs
            ],
            enhanced_info={
                "steps": ["Query重写", "HyDE假设性文档", "多扩展查询", "关键词过滤", "LLM智能重排", "LLM生成回答"],
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

    try:
        mysql_stats = get_stats()
        result["mysql"] = mysql_stats
    except Exception:
        result["mysql"] = {"status": "not_connected"}

    return result


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"


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
    answer = chat_with_assistant(request.query, request.session_id)

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
            # 获取检索结果
            from rag_tools import search_knowledge_base, list_all_products
            list_keywords = ["几款", "有哪些", "什么产品", "产品列表", "全部", "所有", "多少"]
            if any(kw in request.query for kw in list_keywords):
                kb_result = list_all_products.invoke("")
            else:
                kb_result = search_knowledge_base.invoke(request.query)

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
                query=request.query
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



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
