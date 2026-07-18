"""
RAG 智能检索系统 - 后端 API
FastAPI 提供 RESTful 接口
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from config import DOCS_DIR, CHROMA_PERSIST_DIR, RETRIEVAL_TOP_K, RERANK_TOP_K
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


class ScrapeRequest(BaseModel):
    store_url: str = "https://iflytek.tmall.com"
    max_pages: int = 3


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


@app.post("/api/query")
async def query(request: QueryRequest):
    global collection

    if collection is None:
        raise HTTPException(status_code=400, detail="向量数据库未初始化，请先处理文档")

    # 如果选择了 Agent 模式，使用 Agent 范式
    if request.agent_mode and request.agent_mode != "none":
        from agent_paradigms import run_agent

        # 工具1：获取所有文档
        def search_knowledge(query_text: str) -> str:
            """获取知识库所有文档内容"""
            try:
                total = collection.count()
                all_results = collection.get(limit=min(total, 200))
                all_docs = all_results["documents"] if all_results["documents"] else []
                noise_words = ["隐私", "政策", "注销", "快递", "物流", "顺丰", "个人信息"]
                product_docs = [d for d in all_docs if not any(nw in d for nw in noise_words)]
                if not product_docs:
                    product_docs = all_docs
                if not product_docs:
                    product_docs = ["知识库暂无内容"]
                print(f"  [知识库] 获取 {len(product_docs)} 个文档块")
                return "\n".join([f"[文档{i+1}] {d[:300]}" for i, d in enumerate(product_docs[:50])])
            except Exception as e:
                return f"获取文档出错: {e}"

        # 工具2：发给 DeepSeek 分析（带兜底）
        def analyze_with_llm(query_and_docs: str) -> str:
            """将问题和文档发给 DeepSeek 分析，知识库没有时用通用知识补充"""
            try:
                from openai import OpenAI
                from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
                llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

                prompt = f"""你是讯飞产品知识库的智能助手。

回答策略：
1. 优先基于提供的文档回答
2. 如果文档中有相关信息，直接使用
3. 如果文档中没有完全匹配的信息，结合已有信息回答
4. 如果文档中完全没有相关信息，使用你自己的知识补充回答，并说明"以下为通用知识补充"

{query_and_docs}

直接回答："""

                response = llm.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=800,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"DeepSeek分析出错: {e}"

        tools = {"search_knowledge": search_knowledge, "analyze_with_llm": analyze_with_llm}

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


@app.post("/api/scrape")
async def scrape_store(request: ScrapeRequest):
    """爬取淘宝店铺数据并集成到 RAG 系统"""
    global collection, embedding_fn

    try:
        from taobao_scraper import scrape_and_process
        products = scrape_and_process(request.store_url, request.max_pages)
    except Exception as e:
        # 如果爬取失败，使用演示数据
        from taobao_scraper import _get_demo_products, TaobaoScraper
        scraper = TaobaoScraper()
        products = _get_demo_products()
        scraper.products = products
        scraper.save_to_files()

    # 重新处理所有文档并重建向量数据库
    all_chunks = process_documents(DOCS_DIR)

    # 加载爬虫数据
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

    all_texts = []
    for chunks in all_chunks.values():
        all_texts.extend(chunks)

    if all_texts:
        # 直接追加到现有集合，不删除旧数据库
        if collection is not None:
            batch_size = 500
            for i in range(0, len(all_texts), batch_size):
                batch = all_texts[i:i + batch_size]
                ids = [f"scrape_{j}" for j in range(i, i + len(batch))]
                collection.add(documents=batch, ids=ids)
        else:
            collection, embedding_fn = create_vectorstore(all_texts)

    return {
        "status": "success",
        "products_count": len(products),
        "products": [{"name": p["name"], "price": p.get("price", "")} for p in products]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
