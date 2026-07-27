"""
模块：数据提取
从知识库中提取产品参数、FAQ、排障步骤
"""

import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, OUTPUT_DIR


def clean_json(text: str) -> str:
    """清理LLM返回的JSON文本，去除markdown标记等"""
    text = text.strip()
    # 去除 ```json ... ``` 标记
    if text.startswith("```"):
        lines = text.split("\n")
        # 去掉第一行（```json 或 ```）
        lines = lines[1:]
        # 去掉最后一行（```）
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    # 去除可能的前后缀文字
    text = text.strip()
    # 找到第一个 { 或 [
    for i, ch in enumerate(text):
        if ch in "{[":
            text = text[i:]
            break
    # 找到最后一个 } 或 ]
    for i in range(len(text) - 1, -1, -1):
        if text[i] in "}]":
            text = text[:i+1]
            break
    return text.strip()


def extract_product_params(doc_text: str, product_name: str) -> dict:
    """从文档中提取产品参数"""
    llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    prompt = f"""从以下文档中提取{product_name}的产品参数，输出严格的JSON格式。不要输出任何其他文字，只输出JSON。

文档内容：
{doc_text[:5000]}

只输出JSON，格式如下：
{{
  "产品名称": "",
  "品牌": "科大讯飞",
  "型号": "",
  "价格": "",
  "核心参数": {{
    "参数1": "值1",
    "参数2": "值2"
  }},
  "功能特点": ["功能1", "功能2"],
  "适用场景": ""
}}"""

    response = llm.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.1
    )

    result = response.choices[0].message.content.strip()
    cleaned = clean_json(result)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"  [警告] JSON解析失败，尝试修复...")
        # 尝试修复常见问题
        try:
            # 修复尾部逗号
            import re
            fixed = re.sub(r',\s*([}\]])', r'\1', cleaned)
            return json.loads(fixed)
        except:
            return {"产品名称": product_name, "raw": result[:500]}


def extract_faq(doc_text: str, product_name: str) -> list:
    """从文档中提取 FAQ"""
    llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    prompt = f"""从以下文档中提取{product_name}的常见问题和答案。

要求：
1. 至少提取5个FAQ，最多10个
2. 问题要多样化：产品参数、使用方法、功能对比、适用人群等
3. 答案必须完全基于文档中的实际内容，不要添加文档中没有的信息
4. 答案中尽量保留文档中的原文表述

文档内容：
{doc_text[:5000]}

只输出JSON数组，格式如下：
[
  {{"question": "问题1", "answer": "答案1"}},
  {{"question": "问题2", "answer": "答案2"}},
  {{"question": "问题3", "answer": "答案3"}},
  {{"question": "问题4", "answer": "答案4"}},
  {{"question": "问题5", "answer": "答案5"}}
]"""

    response = llm.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.1
    )

    result = response.choices[0].message.content.strip()
    cleaned = clean_json(result)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        import re
        try:
            fixed = re.sub(r',\s*([}\]])', r'\1', cleaned)
            return json.loads(fixed)
        except:
            return [{"question": "解析失败", "answer": result[:200]}]


def extract_troubleshooting(doc_text: str, product_name: str) -> list:
    """从文档中提取排障步骤"""
    llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    prompt = f"""从以下文档中提取{product_name}的常见故障和排障步骤。

要求：
1. 至少提取5个常见故障，最多10个
2. 故障要多样化：硬件问题、软件问题、连接问题、使用问题等
3. 每个故障包含：故障描述、可能原因、解决方法
4. 解决方法必须基于文档中的实际操作步骤，不要编造

文档内容：
{doc_text[:5000]}

只输出JSON数组，格式如下：
[
  {{"problem": "故障描述1", "cause": "可能原因1", "solution": "解决方法1"}},
  {{"problem": "故障描述2", "cause": "可能原因2", "solution": "解决方法2"}},
  {{"problem": "故障描述3", "cause": "可能原因3", "solution": "解决方法3"}},
  {{"problem": "故障描述4", "cause": "可能原因4", "solution": "解决方法4"}},
  {{"problem": "故障描述5", "cause": "可能原因5", "solution": "解决方法5"}}
]"""

    response = llm.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.1
    )

    result = response.choices[0].message.content.strip()
    cleaned = clean_json(result)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        import re
        try:
            fixed = re.sub(r',\s*([}\]])', r'\1', cleaned)
            return json.loads(fixed)
        except:
            return [{"problem": "解析失败", "cause": "", "solution": result[:200]}]


def batch_extract(collection):
    """批量提取所有产品的结构化数据"""
    import json

    total = collection.count()
    all_results = collection.get(limit=min(total, 200))
    all_docs = all_results["documents"]

    # 按产品分组（使用独立的关键词，避免空字符串匹配）
    product_keywords = {
        "讯飞AI录音笔": ["录音笔", "录音设备", "SR702", "SR502", "SR302", "S6 Plus", "S6系列", "S8离线版", "H1 Pro", "Magic", "Pokee"],
        "讯飞翻译机": ["翻译机", "双屏翻译机", "翻译设备"],
        "智能办公本": ["办公本", "X2", "X2LAMY", "墨水屏"],
        "讯飞词典笔": ["词典笔", "X8", "扫描查词"],
        "讯飞学习机": ["学习机", "S90"],
        "讯飞录音卡": ["录音卡", "AIR2611"],
        "讯飞机械键盘": ["键盘", "T8"],
        "讯飞鼠标": ["鼠标", "AM50"],
        "讯飞英语宝": ["英语宝", "EBOX"]
    }

    product_groups = {name: [] for name in product_keywords}

    for doc in all_docs:
        for product_name, keywords in product_keywords.items():
            if any(kw in doc for kw in keywords):
                product_groups[product_name].append(doc)
                break  # 每个文档只归入一个产品组

    # 提取每个产品的数据
    results = {}
    for product_name, docs in product_groups.items():
        if not docs:
            continue

        print(f"\n提取: {product_name} ({len(docs)} 个文档块)")
        full_text = "\n\n".join(docs)  # 使用所有文档块，不再限制前5个

        # 提取产品参数
        params = extract_product_params(full_text, product_name)
        print(f"  参数提取完成")

        # 提取 FAQ
        faq = extract_faq(full_text, product_name)
        print(f"  FAQ 提取完成: {len(faq)} 条")

        # 提取排障步骤
        troubleshooting = extract_troubleshooting(full_text, product_name)
        print(f"  排障步骤提取完成: {len(troubleshooting)} 条")

        results[product_name] = {
            "params": params,
            "faq": faq,
            "troubleshooting": troubleshooting
        }

    # 保存结果
    output_path = os.path.join(OUTPUT_DIR, "extracted_data.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n提取完成，共 {len(results)} 个产品")
    print(f"保存到: {output_path}")

    return results


if __name__ == "__main__":
    from embedding_store import load_vectorstore
    collection, _ = load_vectorstore()
    batch_extract(collection)
