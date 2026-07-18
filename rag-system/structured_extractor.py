"""
模块5：结构化数据提取
将文档内容提取为讯飞产品标准 JSON 格式
"""

import json
import os
from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, OUTPUT_DIR


def get_llm():
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def extract_product_info(text: str, doc_name: str) -> dict:
    """
    用 DeepSeek 从文本中提取结构化产品信息
    输出讯飞标准产品 JSON 格式
    """
    llm = get_llm()

    prompt = f"""从以下文档中提取产品信息，输出严格的 JSON 格式。

文档内容：
{text[:2000]}

输出格式（只输出JSON，不要其他内容）：
{{
  "product_name": "产品名称",
  "brand": "品牌",
  "category": "产品类别（如：智能办公本/录音设备/翻译设备）",
  "price": "价格（如：4999元）",
  "description": "产品简介（50字以内）",
  "specs": {{
    "屏幕": "屏幕参数",
    "处理器": "处理器信息",
    "内存": "内存信息",
    "电池": "电池信息",
    "重量": "重量信息",
    "特色功能": "核心功能"
  }},
  "features": ["功能1", "功能2", "功能3"],
  "target_users": "目标用户群体",
  "highlights": ["亮点1", "亮点2"]
}}

JSON："""

    try:
        response = llm.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.1
        )
        result = response.choices[0].message.content.strip()

        # 清理可能的 markdown 标记
        if result.startswith("```"):
            result = result.split("\n", 1)[1].rsplit("```", 1)[0]

        product_data = json.loads(result)
        product_data["source_file"] = doc_name
        return product_data

    except Exception as e:
        print(f"  提取失败: {e}")
        return {
            "product_name": doc_name,
            "brand": "讯飞",
            "description": text[:100],
            "specs": {},
            "features": [],
            "source_file": doc_name
        }


def batch_extract(chunks_dict: dict) -> list:
    """
    批量提取结构化数据
    输入：{文件名: [chunk1, chunk2, ...]}
    输出：[product1, product2, ...]
    """
    products = []

    for doc_name, chunks in chunks_dict.items():
        print(f"  提取: {doc_name}")

        # 合并所有 chunks 为完整文本
        full_text = "\n".join(chunks)

        product = extract_product_info(full_text, doc_name)
        products.append(product)

        print(f"    ✓ {product.get('product_name', '未知')}")

    return products


def save_structured_json(products: list, output_dir: str = OUTPUT_DIR):
    """保存结构化 JSON 数据"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存完整数据集
    json_path = os.path.join(output_dir, "structured_products.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)
    print(f"  已保存: {json_path}")

    # 保存为 JSONL 格式（每行一个JSON，适合训练）
    jsonl_path = os.path.join(output_dir, "products.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for product in products:
            f.write(json.dumps(product, ensure_ascii=False) + "\n")
    print(f"  已保存: {jsonl_path}")

    # 保存统计信息
    stats = {
        "total_products": len(products),
        "categories": {},
        "total_features": 0
    }
    for p in products:
        cat = p.get("category", "其他")
        stats["categories"][cat] = stats["categories"].get(cat, 0) + 1
        stats["total_features"] += len(p.get("features", []))

    stats_path = os.path.join(output_dir, "extraction_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  已保存统计: {stats_path}")

    return stats


if __name__ == "__main__":
    print("结构化数据提取模块")
    print("请通过 api.py 调用")
