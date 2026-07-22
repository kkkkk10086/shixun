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

        # 工具：智能检索（关键词匹配 + TF-IDF 补充）
        def search_knowledge(query_text: str) -> str:
            """从知识库检索相关信息"""
            try:
                all_docs = []
                seen = set()

                # 获取所有文档
                total = collection.count()
                all_results = collection.get(limit=min(total, 200))
                all_db_docs = all_results["documents"] if all_results["documents"] else []

                # 提取搜索词
                search_terms = [query_text]
                try:
                    import jieba.analyse
                    keywords = jieba.analyse.extract_tags(query_text, topK=10)
                    search_terms.extend(keywords)
                except Exception:
                    pass

                # 关键词匹配
                for doc in all_db_docs:
                    match_count = sum(1 for term in search_terms if term in doc)
                    if match_count >= 1:
                        h = hash(doc)
                        if h not in seen:
                            seen.add(h)
                            all_docs.append((match_count, doc))

                all_docs.sort(key=lambda x: x[0], reverse=True)

                # 关键词过滤
                noise_words = ["隐私", "政策", "注销", "快递", "物流", "顺丰", "个人信息", "JWT", "private String"]
                product_docs = [(s, d) for s, d in all_docs if not any(nw in d for nw in noise_words)]

                # 如果过滤后太少，补充所有文档
                if len(product_docs) < 3:
                    for doc in all_db_docs:
                        h = hash(doc)
                        if h not in seen:
                            seen.add(h)
                            product_docs.append((0, doc))

                docs_only = [d for _, d in product_docs]

                # 关键：如果查询包含特定产品名，确保该产品文档排在最前面
                product_map = {
                    "鼠标": ["鼠标", "AM50"],
                    "键盘": ["键盘", "T8"],
                    "翻译机": ["翻译机"],
                    "录音笔": ["录音笔", "录音"],
                    "办公本": ["办公本", "X2"],
                    "词典笔": ["词典笔", "X8"],
                    "学习机": ["学习机", "S90"],
                    "英语宝": ["英语宝", "EBOX"],
                    "录音卡": ["录音卡"]
                }

                for product_name, product_keywords in product_map.items():
                    if product_name in query_text:
                        # 找到包含该产品关键词的文档，排到最前面
                        product_docs_list = [(s, d) for s, d in product_docs if any(kw in d for kw in product_keywords)]
                        other_docs = [(s, d) for s, d in product_docs if not any(kw in d for kw in product_keywords)]
                        product_docs = product_docs_list + other_docs
                        docs_only = [d for _, d in product_docs]
                        print(f"  [知识库] 优先排序 '{product_name}' 相关文档: {len(product_docs_list)} 个")
                        break

                print(f"  [知识库] 检索到 {len(docs_only)} 个相关文档")
                return "\n".join([f"[文档{i+1}] {d[:400]}" for i, d in enumerate(docs_only[:20])])
            except Exception as e:
                return f"检索出错: {e}"

                docs_only = [d for _, d in product_docs]
                print(f"  [知识库] 检索到 {len(docs_only)} 个相关文档")
                return "\n".join([f"[文档{i+1}] {d[:400]}" for i, d in enumerate(docs_only[:20])])
            except Exception as e:
                return f"检索出错: {e}"

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
