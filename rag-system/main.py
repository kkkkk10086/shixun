"""
RAG 智能检索系统 - 主入口
运行此文件启动完整系统（前端 + 后端）
"""

import os
import sys

# ===== 离线模式：所有模型已缓存，禁止联网 =====
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 加载 .env 文件
from dotenv import load_dotenv
load_dotenv()

import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api import app

if __name__ == "__main__":
    print("=" * 60)
    print("RAG 智能检索系统")
    print("讯飞智能硬件产品助理 - 产品数据底座")
    print("=" * 60)
    print("启动服务: http://localhost:8000")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)
