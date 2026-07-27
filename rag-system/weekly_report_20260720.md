# 周报：RAG 智能检索系统开发（2026-07-20 ~ 2026-07-25）

---

## 一、项目背景

基于 RAG（检索增强生成）架构的讯飞智能硬件产品知识问答系统，使用 FastAPI 后端 + ChromaDB 向量数据库 + DeepSeek LLM，实现9类讯飞产品的智能检索与问答。项目命名为"讯飞产品智库"。

---

## 二、本周完成工作

### 1. RAGAS 评估系统搭建
- 实现5维度评估模块：上下文精度、上下文召回、忠实度、答案相关性、引用准确性
- 支持单个查询评估和批量评估
- 自动生成 Markdown 评估报告
- 前端评估界面：单个评估、批量评估、报告列表浏览
- AI 自动生成测试用例（调用 DeepSeek 根据知识库生成用例并执行）
- 主页新增"AI 测试"入口，一键生成用例并评估

### 2. 三路检索系统
- 关键词匹配 + 向量检索（sentence-transformers）+ BM25 三路合并检索
- BM25 余弦相似度算法实现，服务启动时自动构建索引
- 产品名精确匹配优化（修复"学习机/英语宝"误匹配问题）
- 列表查询去重修复（每个产品独立匹配，无去重）

### 3. 参数调优
- Temperature/Top-P 系统性测试（6种参数组合 × 4个测试场景）
- 生成调优报告 `PARAMETER_TUNING_REPORT.md`
- 推荐配置：参数查询 T=0.1、对话生成 T=0.3、HyDE T=0.5

### 4. 定向切分策略
- 表格内容整表保留（正则提取 Markdown 表格）
- 排障步骤按 `## 故障/问题` 分割
- 普通文本超过3000字符时递归分割

### 5. Agent 模式修复
- ReAct：增加轮数（5→8）、放宽 Final Answer 检测格式
- Reflection：max_tokens 从500提升到1500，支持完整列表输出
- 统一检索逻辑：Agent 工具直接复用 `search_knowledge_base`

### 6. 数据提取优化
- 修复文档分组 bug（`product_name.split("讯飞")` 产生空字符串误匹配）
- 重新提取9个产品参数、FAQ、排障步骤
- 参数准确率87.1%、FAQ准确率96.2%
- 验证脚本 `verify_accuracy.py`
- 运行 `structured_extractor.py` 生成结构化数据集

### 7. 交付物补全

| 文件 | 说明 |
|------|------|
| `PARAMETER_TUNING_REPORT.md` | Temperature/Top-P 调优报告 |
| `GIT_WORKFLOW.md` | Git 分支策略与提交规范 |
| `output/structured_products.json` | 7个产品的结构化JSON数据 |
| `output/products.jsonl` | JSONL 格式（适合训练） |
| `verify_accuracy.py` | 数据提取准确率验证脚本 |

---

## 三、问题修复汇总

| 问题 | 原因 | 修复 |
|------|------|------|
| Agent 检索不到产品 | 使用独立旧版检索逻辑 | 统一复用 `search_knowledge_base` |
| ReAct 超时无答案 | max_tokens 偏小、检测格式严格 | 轮数5→8，支持中文 Final Answer |
| 数据提取参数raw格式 | JSON 清理逻辑不完善 | 新增 `clean_json()` 统一处理 |
| 产品检测误匹配 | "学习"同时匹配学习机和英语宝 | 精确匹配产品名，优先排序 |
| 列表查询遗漏 | `seen` 去重导致产品跳过 | 每个产品独立匹配，无去重 |
| 忠实度评估为0 | 文档截断200字符，评估LLM看不到完整内容 | 改为500字符 |
| 数据提取分组bug | `split("讯飞")` 产生空字符串 | 改用独立关键词列表 |
| Reflection 回答截断 | max_tokens=500 偏小 | 改为1500 |
| RAGAS 报告生成报错 | citations 类型检查缺失 | 添加 `isinstance` 判断 |

---

## 四、当前系统状态

### 系统指标（RAGAS 最佳成绩）

| 指标 | 分数 | 说明 |
|------|------|------|
| 忠实度 | 100% | 回答严格基于文档 |
| 答案相关性 | 64% | 回答与问题相关 |
| 引用准确性 | 100% | 引用来源正确 |
| 上下文召回 | 44% | 部分关键信息遗漏 |
| 上下文精度 | 4% | 检索到部分不相关文档 |
| **综合评分** | **62.4%** | |

### 知识库

- 文档块：163个
- 产品数：9类（录音笔11个型号 + 翻译机 + 办公本 + 词典笔 + 学习机 + 录音卡 + 键盘 + 鼠标 + 英语宝）
- 向量模型：shibing624/text2vec-base-chinese（离线模式）
- 分块大小：2000字符

### 前端功能

6种查询模式 + SSE 流式输出：
- 智能对话（带多轮记忆）
- 增强检索（HyDE + Query重写 + 多扩展 + LLM重排）
- 基础检索
- ReAct Agent
- Plan-and-Solve Agent
- Reflection Agent

### 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| LLM | DeepSeek API（deepseek-chat） |
| 向量数据库 | ChromaDB |
| Embedding | sentence-transformers（text2vec-base-chinese） |
| 文档解析 | MarkItDown（MinerU 降级） |
| 文本分块 | LangChain RecursiveCharacterTextSplitter |
| Agent框架 | LangChain + LangGraph |
| 提示词模板 | Jinja2 |
| 数据库 | MySQL + PyMySQL |
| 前端 | HTML + CSS + JavaScript（原生） |
| 分词 | jieba |
| 评估框架 | RAGAS（自研实现） |

---

## 五、Git 提交记录

```
1ad3991 feat: RAGAS评估系统 + 三路检索 + 参数调优 + 定向切分 + 数据提取优化
23c13d6 feat: 完成数据提取优化和交付物
5e11713 RAG智能检索系统完整版：LangChain Tool + LangGraph Agent + RAGAS评估 + Jinja2模板
```

---

## 六、下周计划

- 继续优化检索召回率（目标85%+）
- 补充详细产品规格文档
- 完善前端交互体验
- MinerU API 修复
