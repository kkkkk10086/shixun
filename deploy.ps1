# 部署脚本 - 启动全部服务（无需 Docker）
# 用法: .\deploy.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  讯飞智能硬件产品AI助手 - 部署脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 检查 Python 虚拟环境
$venvPath = "D:\xjwjj\实训\.venv"
if (Test-Path "$venvPath\Scripts\Activate.ps1") {
    Write-Host "[1/4] 激活虚拟环境..." -ForegroundColor Yellow
    & "$venvPath\Scripts\Activate.ps1"
} else {
    Write-Host "[!] 未找到虚拟环境，尝试使用系统 Python" -ForegroundColor Yellow
}

# 1. 启动 MySQL（如果未运行）
Write-Host "[2/4] 检查 MySQL 服务..." -ForegroundColor Yellow
$mysql = Get-Service -Name "MySQL*" -ErrorAction SilentlyContinue
if ($mysql -and $mysql.Status -eq 'Running') {
    Write-Host "  MySQL 服务运行中" -ForegroundColor Green
} else {
    Write-Host "  请确保 MySQL 已启动（配置: localhost:3306, root/YOUR_PASSWORD）" -ForegroundColor Yellow
}

# 2. 启动 OpenMAIC（端口 3001）
Write-Host "[3/4] 启动 OpenMAIC 视频服务..." -ForegroundColor Yellow
$openmaicDir = "D:\xjwjj\实训\openmaic"
if (Test-Path "$openmaicDir\node_modules\.pnpm") {
    Start-Process -WindowStyle Hidden -FilePath "pnpm" -ArgumentList "--dir $openmaicDir dev --port 3001"
    Write-Host "  OpenMAIC 已启动 → http://localhost:3001" -ForegroundColor Green
} else {
    Write-Host "  OpenMAIC 依赖未安装，请先运行: cd $openmaicDir && pnpm install" -ForegroundColor Red
}

# 3. 启动 RAG 系统（端口 8000）
Write-Host "[4/4] 启动 RAG 智能检索系统..." -ForegroundColor Yellow
$ragDir = "D:\xjwjj\实训\rag-system"
$logFile = "$ragDir\server.log"

# 在后台启动 RAG 服务
$job = Start-Job -ScriptBlock {
    param($dir, $log)
    Set-Location $dir
    python api.py *>&1 | Out-File -FilePath $log -Encoding utf8
} -ArgumentList $ragDir, $logFile

Write-Host "  RAG 系统已启动 → http://localhost:8000 (日志: $logFile)" -ForegroundColor Green
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  服务启动完成！" -ForegroundColor Green
Write-Host "  RAG 系统:     http://localhost:8000" -ForegroundColor Green
Write-Host "  OpenMAIC:     http://localhost:3001" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "查看日志: Get-Content $ragDir\server.log -Tail 20 -Wait" -ForegroundColor Gray
Write-Host "停止服务: Get-Job | Stop-Job" -ForegroundColor Gray