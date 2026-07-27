# RAG 系统配置

# DeepSeek API 配置（从环境变量读取，或在此处填写）
import os
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-api-key-here")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"

# MinerU API 配置
MINERU_API_KEY = os.getenv("MINERU_API_KEY", "your-mineru-key-here")

# MySQL 配置
MYSQL_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "Khy050126",
    "database": "rag_system",
    "charset": "utf8mb4"
}

# 分块参数（加大块大小，保留更多上下文，确保产品参数不被拆散）
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

# 检索参数
RETRIEVAL_TOP_K = 5
RERANK_TOP_K = 3

# 文档目录
DOCS_DIR = [
    r"D:\新建文件夹\智能办公本使用说明书（X2及X2LAMY）",
    r"D:\新建文件夹\讯飞ai录音卡.docx",
    r"D:\新建文件夹\讯飞双屏翻译机2.0.docx"
]

# ChromaDB 存储路径
CHROMA_PERSIST_DIR = "./chroma_db"

# 输出目录
OUTPUT_DIR = "./output"

# 服务端口
SERVER_PORT = 8000
