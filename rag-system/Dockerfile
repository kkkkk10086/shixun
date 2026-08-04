FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 创建必要目录
RUN mkdir -p uploads output video_data chroma_db

# 暴露端口
EXPOSE 8000

# 启动（使用 api.py 避免离线模式限制）
CMD ["python", "-c", "import uvicorn; uvicorn.run('api:app', host='0.0.0.0', port=8000)"]

# 构建时预下载模型（可选，加速首次启动）
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-zh-v1.5')"