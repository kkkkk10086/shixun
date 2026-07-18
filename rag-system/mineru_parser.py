"""
MinerU 文档解析模块
使用 MinerU 云 API 解析 PDF/Word 等文档，转换为 Markdown
"""

import os
import time
import requests
from config import MINERU_API_KEY, OUTPUT_DIR


# MinerU API 配置
MINERU_BASE_URL = "https://mineru.net/api/v4"


def parse_document(file_path: str) -> str:
    """
    使用 MinerU API 解析文档，返回 Markdown 文本

    流程：
    1. 上传文件创建解析任务
    2. 轮询任务状态
    3. 获取解析结果
    """
    headers = {
        "Authorization": f"Bearer {MINERU_API_KEY}"
    }

    print(f"  [MinerU] 上传文件: {os.path.basename(file_path)}")

    # 步骤1：上传文件创建解析任务
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f)}
        data = {
            "output_format": "markdown",
            "enable_formula": "true",
            "enable_table": "true"
        }

        response = requests.post(
            f"{MINERU_BASE_URL}/extract/task",
            headers=headers,
            files=files,
            data=data,
            timeout=60
        )

    if response.status_code != 200:
        raise Exception(f"MinerU 上传失败: {response.status_code} - {response.text}")

    result = response.json()
    task_id = result.get("data", {}).get("task_id")

    if not task_id:
        raise Exception(f"MinerU 未返回 task_id: {result}")

    print(f"  [MinerU] 任务创建成功: {task_id}")

    # 步骤2：轮询任务状态
    max_wait = 300  # 最多等5分钟
    poll_interval = 3  # 每3秒查一次
    elapsed = 0

    while elapsed < max_wait:
        time.sleep(poll_interval)
        elapsed += poll_interval

        status_response = requests.get(
            f"{MINERU_BASE_URL}/extract/task/{task_id}",
            headers=headers,
            timeout=30
        )

        if status_response.status_code != 200:
            print(f"  [MinerU] 状态查询失败: {status_response.status_code}")
            continue

        status_data = status_response.json()
        task_status = status_data.get("data", {}).get("status", "")

        if task_status == "completed":
            print(f"  [MinerU] 解析完成 ({elapsed}秒)")
            break
        elif task_status == "failed":
            error_msg = status_data.get("data", {}).get("error", "未知错误")
            raise Exception(f"MinerU 解析失败: {error_msg}")
        else:
            print(f"  [MinerU] 处理中... ({elapsed}秒)")

    # 步骤3：获取解析结果
    result_response = requests.get(
        f"{MINERU_BASE_URL}/extract/task/{task_id}/result",
        headers=headers,
        timeout=30
    )

    if result_response.status_code != 200:
        raise Exception(f"MinerU 获取结果失败: {result_response.status_code}")

    result_data = result_response.json()

    # 提取 Markdown 内容
    markdown_content = result_data.get("data", {}).get("markdown", "")

    if not markdown_content:
        # 尝试从其他字段提取
        markdown_content = result_data.get("data", {}).get("content", "")

    print(f"  [MinerU] 获取到 {len(markdown_content)} 字符的 Markdown")

    return markdown_content


def parse_to_file(file_path: str, output_dir: str = OUTPUT_DIR) -> str:
    """解析文档并保存为 Markdown 文件"""
    os.makedirs(output_dir, exist_ok=True)

    markdown = parse_document(file_path)

    # 保存 Markdown
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    md_path = os.path.join(output_dir, f"{base_name}_mineru.md")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"  [MinerU] 已保存: {md_path}")
    return md_path


if __name__ == "__main__":
    print("MinerU 文档解析模块")
    print(f"API Key: {MINERU_API_KEY[:10]}...")
