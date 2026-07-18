# RAG 智能检索系统 — 项目实现计划

> 项目目标：实现一个带高级检索策略的 RAG 系统
> 技术栈：Python + MarkItDown + LangChain + ChromaDB + 星火大模型 API

---

## 项目结构

```
D:\xjwjj\实训\rag-system\
├── main.py                  # 主入口，串联所有模块
├── requirements.txt         # 依赖清单
├── config.py                # 配置（API Key、模型参数等）
├── document_processor.py    # 模块1：文档处理管线
├── embedding_store.py       # 模块2：向量化与存储
├── retriever.py             # 模块3：检索策略
├── query_enhancer.py        # 模块4：查询增强（HyDE/重写/多扩展/重排）
├── docs/                    # 放测试用的文档（PDF/Word）
├── output/                  # 生成的 Markdown 和分块结果
└── README.md
```

---

## 模块拆解与实现步骤

### 模块1：文档处理管线（document_processor.py）

**功能：** 任意格式文档 → Markdown → 智能分块

**步骤 1.1：MarkItDown 文档转换**
```python
# 依赖：pip install markitdown
from markitdown import MarkItDown

md = MarkItDown()
result = md.convert("docs/测试文档.pdf")
markdown_text = result.text_content
```

**步骤 1.2：智能分块（Chunking）**
```python
# 使用 LangChain 的 RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 每块约500字符
    chunk_overlap=50,      # 块间重叠50字符，保持上下文连续
    separators=["\n\n", "\n", "。", "！", "？", ".", " "]
)
chunks = splitter.split_text(markdown_text)
```

**产出：** 一个将任意文档转为 Markdown 分块列表的函数

---

### 模块2：向量化与存储检索（embedding_store.py）

**功能：** 文本 → 向量 → 存入向量数据库 → 支持检索

**步骤 2.1：选择 Embedding 模型**
```python
# 方案A：使用本地模型（无需API，离线可用）
# pip install sentence-transformers
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

# 方案B：使用星火 Embedding API（需联网）
# 推荐先用方案A，简单可靠
```

**步骤 2.2：存入向量数据库**
```python
# 使用 ChromaDB（轻量级，无需服务器）
# pip install chromadb
from langchain.vectorstores import Chroma

vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"   # 持久化存储
)

# 检索示例
results = vectorstore.similarity_search("如何重置设备", k=3)
```

**产出：** 支持存储和基础相似度检索的向量数据库

---

### 模块3：检索策略（retriever.py）

**功能：** 从向量数据库中检索最相关的文档

**步骤 3.1：基础检索**
```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}    # 返回最相关的5个文档
)
```

**步骤 3.2：混合检索（可选增强）**
```python
# 结合关键词检索 + 向量检索，提高召回率
from langchain.retrievers import EnsembleRetriever
```

---

### 模块4：查询增强（query_enhancer.py）

这是老师重点要求的四个高级功能：

**功能 4.1：假设性文档（HyDE）**
> 思路：先让 LLM 生成一个"假设性答案"，再用这个答案去检索，比直接用原始 query 检索效果更好

```python
def hyde_query(query: str, llm) -> str:
    """用 LLM 生成假设性文档，再用它去检索"""
    prompt = f"请针对以下问题，写一段详细的假设性回答（200字左右）：\n{query}"
    hypothetical_doc = llm.invoke(prompt)
    return hypothetical_doc   # 用这个去向量库检索
```

**功能 4.2：Query 重写**
> 思路：把用户的口语化问题改写为更适合检索的形式

```python
def rewrite_query(query: str, llm) -> str:
    """将用户查询重写为更精确的检索查询"""
    prompt = f"请将以下用户问题重写为更适合知识库检索的形式，保留核心语义：\n{query}"
    rewritten = llm.invoke(prompt)
    return rewritten
```

**功能 4.3：多扩展查询（Multi-Query）**
> 思路：一个问题生成多个不同角度的查询，分别检索后合并结果，提高覆盖率

```python
def expand_queries(query: str, llm, n=3) -> list:
    """一个问题扩展为多个不同角度的查询"""
    prompt = f"请将以下问题从{n}个不同角度改写，每行一个：\n{query}"
    expanded = llm.invoke(prompt)
    return [q.strip() for q in expanded.split("\n") if q.strip()]
```

**功能 4.4：候选文档重排（Reranking）**
> 思路：初步检索出一批文档后，用重排模型对它们重新排序，把最相关的排到前面

```python
# 使用交叉编码器（Cross-Encoder）进行重排
# pip install sentence-transformers
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_documents(query: str, documents: list, top_k=3) -> list:
    """对检索结果进行重排序"""
    pairs = [(query, doc.page_content) for doc in documents]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked[:top_k]]
```

---

## 实现顺序（建议）

```
第1步：搭建项目结构，安装依赖          （30分钟）
第2步：实现文档处理管线（MarkItDown + 分块） （1小时）
第3步：实现向量化与存储（ChromaDB）       （1小时）
第4步：实现基础检索                      （30分钟）
第5步：实现 HyDE                        （30分钟）
第6步：实现 Query 重写                   （20分钟）
第7步：实现多扩展查询                    （20分钟）
第8步：实现文档重排                      （30分钟）
第9步：串联所有模块，写 main.py          （30分钟）
第10步：测试与调试                       （1小时）
```

---

## 依赖清单（requirements.txt）

```
markitdown
langchain
langchain-community
chromadb
sentence-transformers
openai
jieba
```

---

## 当前进度

- [ ] 第1步：搭建项目结构
- [ ] 第2步：文档处理管线
- [ ] 第3步：向量化与存储
- [ ] 第4步：基础检索
- [ ] 第5步：HyDE
- [ ] 第6步：Query 重写
- [ ] 第7步：多扩展查询
- [ ] 第8步：文档重排
- [ ] 第9步：串联模块
- [ ] 第10步：测试调试
