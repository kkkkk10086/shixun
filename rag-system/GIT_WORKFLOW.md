# Git 工作流文档

> RAG 智能检索系统 · Git 版本管理规范

---

## 1. 仓库信息

| 项目 | 信息 |
|------|------|
| 仓库地址 | https://github.com/kkkkk10086/shixun.git |
| 主分支 | main |
| 开发分支 | dev |
| 用户名 | KKKKK10086 |

---

## 2. 分支策略

```
main (生产分支)
  ↑ 合并
dev (开发分支)
  ↑ 合并
feature/xxx (功能分支)
```

- **main**：生产分支，只接受 dev 分支的合并，不直接提交
- **dev**：开发分支，日常开发在此分支进行
- **feature/xxx**：功能分支，每个新功能创建独立分支

---

## 3. 工作流程

### 3.1 日常开发流程

```bash
# 1. 切换到 dev 分支
git checkout dev

# 2. 拉取最新代码
git pull origin dev

# 3. 创建功能分支
git checkout -b feature/ragas-evaluation

# 4. 开发完成后提交
git add .
git commit -m "feat: 新增RAGAS评估系统"

# 5. 推送功能分支
git push origin feature/ragas-evaluation

# 6. 在 GitHub 上创建 Pull Request 合并到 dev
```

### 3.2 发布流程

```bash
# 1. dev 分支测试通过后，合并到 main
git checkout main
git merge dev

# 2. 打标签
git tag -a v1.0.0 -m "发布版本1.0.0"

# 3. 推送
git push origin main --tags
```

---

## 4. 提交规范

### 4.1 提交信息格式

```
<类型>(<范围>): <描述>

<可选正文>

<可选脚注>
```

### 4.2 类型说明

| 类型 | 说明 | 示例 |
|------|------|------|
| feat | 新功能 | feat: 新增RAGAS评估系统 |
| fix | 修复Bug | fix: 修复检索精度问题 |
| docs | 文档更新 | docs: 更新技术文档 |
| style | 代码格式 | style: 格式化代码 |
| refactor | 重构 | refactor: 优化检索逻辑 |
| test | 测试 | test: 添加单元测试 |
| chore | 构建/工具 | chore: 更新依赖 |

### 4.3 示例

```bash
git commit -m "feat: 新增RAGAS评估系统，支持单个查询和批量评估"
git commit -m "fix: 修复Agent模式检索不到产品信息的问题"
git commit -m "docs: 更新技术文档，添加评估系统说明"
```

---

## 5. .gitignore 规则

```
.venv/              # Python虚拟环境
__pycache__/        # 编译缓存
*.pyc               # Python编译文件
.env                # 环境变量（含API密钥）
chroma_db/          # 向量数据库（运行时生成）
uploads/            # 上传文件
output/*.md         # Markdown输出文件
output/*_chunks.txt # 分块文件
app.log             # 日志文件
.DS_Store           # macOS系统文件
.idea/              # IDE配置
query               # 临时文件
test_*.py           # 测试脚本
```

---

## 6. 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v1.0.0 | 2026-07-23 | RAG系统完整版：LangChain + LangGraph + RAGAS评估 + 数据提取 |

---

## 7. 协作规范

1. 不直接向 main 分支提交代码
2. 每个功能创建独立分支
3. 提交前先 pull 最新代码，解决冲突
4. 提交信息必须遵循规范
5. 敏感信息（API密钥、密码）不得提交到 Git
