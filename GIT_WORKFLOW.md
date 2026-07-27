# Git 协同规范

## 分支模型

```
main        ← 生产环境代码（受保护）
  └── dev   ← 开发集成分支
       ├── feature/xxx    ← 功能分支
       ├── fix/xxx        ← 修复分支
       └── docs/xxx       ← 文档分支
```

## 日常工作流程

### 1. 创建功能分支
```bash
# 从 dev 创建新分支
git checkout dev
git pull origin dev
git checkout -b feature/新增xxx功能
```

### 2. 开发与提交
```bash
# 开发完成后
git add .
git commit -m "feat: 新增xxx功能"
```

**提交规范：**
- `feat:` 新功能
- `fix:` 修复 bug
- `docs:` 文档变更
- `refactor:` 重构
- `chore:` 构建/工具变动

### 3. 推送到远程
```bash
git push origin feature/新增xxx功能
```

### 4. 创建 Pull Request (PR)
1. 打开 GitHub 仓库页面
2. 点击 「Compare & pull request」
3. 填写 PR 标题和描述
4. 选择 `dev` 作为目标分支
5. 点击 「Create pull request」

### 5. 代码审查与合并
1. 至少一人审查通过
2. 点击 「Merge pull request」
3. 合并后删除功能分支

### 6. 同步到 main
```bash
git checkout main
git pull origin main
git merge dev
git push origin main
```

## 常用命令速查

| 操作 | 命令 |
|------|------|
| 查看分支 | `git branch -a` |
| 切换分支 | `git checkout 分支名` |
| 拉取最新 | `git pull origin 当前分支` |
| 推送代码 | `git push origin 当前分支` |
| 合并分支 | `git merge 要合并的分支` |
| 删除本地分支 | `git branch -d 分支名` |
| 删除远程分支 | `git push origin --delete 分支名` |
