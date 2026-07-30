# RAG 系统配置

# DeepSeek API 配置（从环境变量读取，或在此处填写）
import os
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_API_KEY_HERE")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"

# MinerU API 配置
MINERU_API_KEY = os.getenv("MINERU_API_KEY", "YOUR_MINERU_API_KEY_HERE")

# MySQL 配置
MYSQL_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "YOUR_MYSQL_PASSWORD_HERE",
    "database": "rag_system",
    "charset": "utf8mb4"
}

# 分块参数
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 400

# 语义分块参数
CHUNK_SECTION_SIZE = 1200     # 产品章节最大长度，超长则按子标题拆分
CHUNK_TABLE_MAX_ROWS = 20     # 表格超过此行数则按行组分拆
CHUNK_MIN_SIZE = 100          # 小于此长度的块丢弃（除非是表格）

# 检索参数
RETRIEVAL_TOP_K = 20   # 每路检索取 top-20，RRF融合后更多候选
RERANK_TOP_K = 10      # Cross-Encoder 重排后保留 top-10
RRF_K = 60             # RRF 融合常数（越大越平滑）

# Embedding 模型
# 升级为 bge-large-zh-v1.5（1024维，检索精度提升约5-10%）
# 首次运行需下载 ~1.3GB 模型，之后秒启动
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
EMBEDDING_DIMENSION = 1024

# Cross-Encoder 重排模型（首次运行会自动从 hf-mirror 下载）
CROSS_ENCODER_MODEL = "maidalun1020/bce-reranker-base_v1"
CROSS_ENCODER_USE_GPU = True   # 有 GPU 则设为 True

# 文档目录
DOCS_DIR = [
    r"D:\新建文件夹\智能办公本使用说明书（X2及X2LAMY）.pdf",
    r"D:\新建文件夹\讯飞ai录音卡.docx",
    r"D:\新建文件夹\讯飞双屏翻译机2.0.docx",
]

# ChromaDB 存储路径
CHROMA_PERSIST_DIR = "./chroma_db"

# 输出目录
OUTPUT_DIR = "./output"

# 服务端口
SERVER_PORT = 8000