# RAG 智能检索系统 — 技术文档

> 讯飞智能硬件产品助理 · 产品数据底座搭建

---

## 1. 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    前端 (index.html)                      │
│              HTML + JavaScript + CSS                     │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP API
┌───────────────────────▼─────────────────────────────────┐
│                  后端 (FastAPI)                           │
│                    api.py                                │
├─────────┬──────────┬──────────┬──────────┬──────────────┤
│ 文档处理 │ 向量存储  │ 查询增强  │ 结构化提取│ 爬虫模块     │
│模块1     │ 模块2     │ 模块3    │ 模块5    │ 模块4        │
│doc_proc  │embed_store│query_enh│struct_ext│taobao_scrape │
└────┬────┴─────┬────┴────┬────┴──────────┴──────────────┘
     │          │         │
     ▼          ▼         ▼
 MarkItDown  ChromaDB   DeepSeek
 + 分块     (向量数据库)  (LLM API)
                │
                ▼
            MySQL
          (结构化存储)
```

---

## 2. 模块说明

### 模块1：文档处理管线 (document_processor.py)

**功能：** 任意格式文档 → MarkItDown → Markdown → 智能分块

| 步骤 | 技术 | 说明 |
|------|------|------|
| 文档转换 | MarkItDown | 支持 PDF、Word、TXT、Markdown |
| 智能分块 | RecursiveCharacterTextSplitter | 块大小1000字符，重叠100字符 |
| 并发处理 | asyncio + ThreadPoolExecutor | 4线程并发处理多个文件 |

**核心函数：**
- `process_documents(doc_paths)` — 同步处理
- `process_documents_async(doc_paths)` — 异步并发处理
- `save_markdown(chunks)` — 保存Markdown文件

### 模块2：向量化存储 (embedding_store.py)

**功能：** 文本 → TF-IDF向量化 → ChromaDB存储

| 组件 | 说明 |
|------|------|
| TfidfEmbeddingFunction | 自定义Embedding，jieba分词 + TF-IDF + 关键词加权 |
| ChromaDB | 轻量级向量数据库，持久化存储 |
| 二元组特征 | ngram_range=(1,2)，提升中文检索效果 |

### 模块3：查询增强 (query_enhancer.py)

**功能：** 4个高级检索策略

| 策略 | 技术 | 说明 |
|------|------|------|
| HyDE | DeepSeek API | 生成假设性文档，再用它检索 |
| Query重写 | DeepSeek API | 口语化问题 → 检索友好查询 |
| 多扩展查询 | DeepSeek API | 一个问题 → 多角度查询 |
| 文档重排 | DeepSeek API + TF-IDF | LLM智能评分 + 关键词过滤 |

**完整流程：**
```
用户查询
  ↓ Query重写（DeepSeek）
  ↓ HyDE假设性文档（DeepSeek）
  ↓ 多扩展查询（DeepSeek）
  ↓ 合并检索结果（ChromaDB）
  ↓ 关键词预过滤（jieba）
  ↓ LLM智能重排（DeepSeek）
最终结果
```

### 模块4：淘宝爬虫 (taobao_scraper.py)

**功能：** 从讯飞淘宝店铺爬取产品数据

| 状态 | 说明 |
|------|------|
| 当前 | 内置4款讯飞产品演示数据 |
| 待升级 | 需安装Playwright实现真实爬取 |

### 模块5：结构化数据提取 (structured_extractor.py)

**功能：** 将文档内容转为标准JSON格式

**输出格式：**
```json
{
  "product_name": "讯飞智能办公本X2",
  "brand": "讯飞",
  "category": "智能办公本",
  "price": "4999元",
  "specs": { "屏幕": "10.3英寸", "电池": "4000mAh" },
  "features": ["语音转写", "手写识别", "会议纪要"],
  "target_users": "商务人士",
  "highlights": ["墨水屏护眼", "AI会议纪要"]
}
```

### 模块6：Jinja2提示词模板 (prompt_templates.py)

**功能：** 管理文档清洗、数据提取、FAQ生成的模板

| 模板 | 用途 |
|------|------|
| document_cleanup | 文档清洗 |
| product_extraction | 产品信息提取 |
| faq_generation | FAQ自动生成 |
| troubleshooting | 排障步骤提取 |
| query_enhance | 查询增强 |

---

## 3. API 接口

### POST /api/query
智能检索接口

**请求：**
```json
{
  "query": "智能办公本有什么功能？",
  "mode": "enhanced",
  "top_k": 3
}
```

**响应：**
```json
{
  "query": "智能办公本有什么功能？",
  "mode": "enhanced",
  "results": [
    { "content": "...", "score": 0.85 }
  ],
  "enhanced_info": {
    "steps": ["Query重写", "HyDE", "多扩展查询", "关键词过滤", "LLM重排"]
  }
}
```

### POST /api/upload
文档上传接口

### POST /api/scrape
数据爬取接口

### GET /api/stats
系统状态查询

---

## 4. 文件结构

```
rag-system/
├── api.py                  # FastAPI 后端
├── config.py               # 配置文件
├── main.py                 # 启动入口
├── document_processor.py   # 模块1：文档处理
├── embedding_store.py      # 模块2：向量存储
├── query_enhancer.py       # 模块3：查询增强
├── taobao_scraper.py       # 模块4：淘宝爬虫
├── structured_extractor.py # 模块5：结构化提取
├── prompt_templates.py     # 模块6：提示词模板
├── mysql_store.py          # MySQL存储
├── requirements.txt        # 依赖清单
├── frontend/
│   └── index.html          # 前端界面
├── output/                 # 输出目录
│   ├── *.md               # Markdown文件
│   ├── *_chunks.txt       # 分块文件
│   └── structured_*.json  # 结构化数据
├── chroma_db/              # 向量数据库
└── uploads/                # 上传文件
```

---

## 5. 技术栈

| 组件 | 技术 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| 文档转换 | MarkItDown |
| 文本分块 | LangChain TextSplitter |
| 向量数据库 | ChromaDB |
| Embedding | TF-IDF + jieba |
| LLM API | DeepSeek (deepseek-chat) |
| 数据库 | MySQL + PyMySQL |
| 提示词模板 | Jinja2 |
| 前端 | HTML + CSS + JavaScript |
