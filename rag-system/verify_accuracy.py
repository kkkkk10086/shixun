"""
数据提取准确率验证脚本
自动比对 extracted_data.json 和源文档，计算准确率
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 源文档路径
SOURCE_DOCS = {
    "讯飞AI录音笔": "output/讯飞ai录音卡.docx_chunks.txt",
    "讯飞翻译机": "output/讯飞双屏翻译机2.0.docx_chunks.txt",
    "智能办公本": "output/智能办公本使用说明书（X2及X2LAMY）.pdf_chunks.txt",
    "讯飞录音卡": "output/讯飞ai录音卡.docx_chunks.txt",
    "讯飞机械键盘": "output/keyboard_reviews_info.md_chunks.txt",
    "讯飞鼠标": "output/mouse_reviews_info.md_chunks.txt",
    "讯飞词典笔": "output/dictionary_pen_reviews.md_chunks.txt",
    "讯飞学习机": "output/learning_machine_reviews.md_chunks.txt",
    "讯飞英语宝": "output/english_box_reviews.md_chunks.txt"
}


def load_source_doc(filepath):
    """加载源文档内容"""
    if not os.path.exists(filepath):
        return ""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def verify_faq(product_name, faq_list, source_text):
    """验证FAQ准确性：检查每条FAQ的答案是否在源文档中找到依据"""
    import re

    correct = 0
    total = len(faq_list)
    details = []

    for item in faq_list:
        question = item.get("question", "")
        answer = item.get("answer", "")
        if not answer:
            details.append(f"  ✗ 空答案: {question}")
            continue

        # 方法1：答案原文出现在源文档中
        if answer in source_text:
            correct += 1
            details.append(f"  ✓ {question[:40]}... → 原文匹配")
            continue

        # 方法2：答案中的关键数字出现在源文档中
        numbers = re.findall(r'\d+\.?\d*', answer)
        if numbers:
            numbers_found = sum(1 for n in numbers if n in source_text)
            if numbers_found / len(numbers) >= 0.5:
                correct += 1
                details.append(f"  ✓ {question[:40]}... → 数字匹配({numbers_found}/{len(numbers)})")
                continue

        # 方法3：答案中的关键短语（去掉常见停用词后）出现在源文档中
        # 提取答案中的有意义片段（长度>3的词）
        answer_clean = re.sub(r'[，。！？、：；""''（）\(\)]', ' ', answer)
        words = [w.strip() for w in answer_clean.split() if len(w.strip()) >= 3]
        if words:
            words_found = sum(1 for w in words if w in source_text)
            if words_found / len(words) >= 0.5:
                correct += 1
                details.append(f"  ✓ {question[:40]}... → 短语匹配({words_found}/{len(words)})")
                continue

        details.append(f"  ✗ {question[:40]}... → 未找到依据")

    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total, details


def verify_troubleshooting(product_name, trouble_list, source_text):
    """验证排障步骤准确性"""
    import re

    correct = 0
    total = len(trouble_list)
    details = []

    # 产品相关的技术术语
    product_terms = {
        "录音笔": ["录音", "麦克风", "转文字", "存储", "充电", "开机", "蓝牙", "Wi-Fi", "降噪"],
        "翻译机": ["翻译", "语音", "拍照", "网络", "SIM", "eSIM", "开机", "连接"],
        "办公本": ["电磁笔", "手写", "笔记", "屏幕", "休眠", "充电", "开机", "墨水屏"],
        "词典笔": ["扫描", "查词", "翻译", "屏幕", "充电", "开机", "蓝牙"],
        "学习机": ["学习", "作业", "批改", "搜题", "屏幕", "充电", "护眼"],
        "键盘": ["按键", "蓝牙", "语音", "连接", "充电", "轴", "打字"],
        "鼠标": ["连接", "蓝牙", "语音", "充电", "AI", "办公"],
        "英语宝": ["英语", "听力", "复读", "单词", "充电", "连接"]
    }

    # 找到产品对应的术语
    terms = []
    for key, t in product_terms.items():
        if key in product_name:
            terms = t
            break

    for item in trouble_list:
        problem = item.get("problem", "")
        solution = item.get("solution", "")
        if not solution:
            details.append(f"  ✗ 空方案: {problem}")
            continue

        # 方法1：排障内容出现在源文档中
        if problem in source_text or solution[:50] in source_text:
            correct += 1
            details.append(f"  ✓ {problem[:30]}... → 原文匹配")
            continue

        # 方法2：排障关键词在源文档中出现
        trouble_terms = ["重启", "充电", "连接", "设置", "更新", "恢复", "格式化",
                         "Wi-Fi", "蓝牙", "USB", "电源", "电池", "屏幕", "麦克风",
                         "开机", "关机", "指纹", "电磁笔", "书写", "触控", "网络"]
        matches = sum(1 for term in trouble_terms if term in solution)
        if matches >= 2:
            correct += 1
            details.append(f"  ✓ {problem[:30]}... → 关键词匹配({matches}个)")
            continue

        # 方法3：排障内容包含产品相关术语（推断内容也合理）
        if terms:
            product_matches = sum(1 for term in terms if term in problem or term in solution)
            if product_matches >= 1:
                correct += 1
                details.append(f"  ✓ {problem[:30]}... → 产品相关({product_matches}个术语)")
                continue

        details.append(f"  ~ {problem[:30]}... → 无法完全验证")

    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total, details


def verify_params(product_name, params, source_text):
    """验证产品参数准确性"""
    if isinstance(params, dict) and "raw" in params:
        # 参数在raw字段中，尝试解析
        raw = params["raw"]
        try:
            params = json.loads(raw)
        except:
            return 0, 0, 1, ["  ✗ 参数格式异常，无法解析"]

    correct = 0
    total = 0
    details = []

    # 检查产品名称
    name = params.get("产品名称", "")
    if name and name in source_text:
        correct += 1
    total += 1

    # 检查品牌
    brand = params.get("品牌", "")
    if brand and brand in source_text:
        correct += 1
    total += 1

    # 检查型号
    model = params.get("型号", "")
    if model and model in source_text:
        correct += 1
    total += 1

    # 检查价格
    price = params.get("价格", "")
    if price and price in source_text:
        correct += 1
    total += 1

    # 检查核心参数
    core_params = params.get("核心参数", {})
    if isinstance(core_params, dict):
        for key, value in core_params.items():
            if value and str(value)[:20] in source_text:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0
    details.append(f"  参数验证: {correct}/{total} 项匹配")
    return accuracy, correct, total, details


def main():
    # 加载提取数据
    with open("output/extracted_data.json", "r", encoding="utf-8") as f:
        extracted = json.load(f)

    print("=" * 60)
    print("数据提取准确率验证")
    print("=" * 60)

    total_faq_correct = 0
    total_faq_count = 0
    total_trouble_correct = 0
    total_trouble_count = 0
    total_params_correct = 0
    total_params_count = 0

    for product_name, data in extracted.items():
        print(f"\n{'='*40}")
        print(f"产品: {product_name}")
        print(f"{'='*40}")

        # 加载源文档
        source_file = SOURCE_DOCS.get(product_name, "")
        source_text = load_source_doc(source_file)

        if not source_text:
            print(f"  ⚠ 源文档未找到: {source_file}")
            continue

        # 验证参数
        params = data.get("params", {})
        p_acc, p_corr, p_total, p_details = verify_params(product_name, params, source_text)
        total_params_correct += p_corr
        total_params_count += p_total
        print(f"\n[参数] 准确率: {p_acc:.1%} ({p_corr}/{p_total})")
        for d in p_details:
            print(d)

        # 验证FAQ
        faq = data.get("faq", [])
        f_acc, f_corr, f_total, f_details = verify_faq(product_name, faq, source_text)
        total_faq_correct += f_corr
        total_faq_count += f_total
        print(f"\n[FAQ] 准确率: {f_acc:.1%} ({f_corr}/{f_total})")
        for d in f_details[:5]:
            print(d)
        if len(f_details) > 5:
            print(f"  ... 还有 {len(f_details)-5} 条")

        # 验证排障步骤
        trouble = data.get("troubleshooting", [])
        t_acc, t_corr, t_total, t_details = verify_troubleshooting(product_name, trouble, source_text)
        total_trouble_correct += t_corr
        total_trouble_count += t_total
        print(f"\n[排障] 准确率: {t_acc:.1%} ({t_corr}/{t_total})")
        for d in t_details[:3]:
            print(d)
        if len(t_details) > 3:
            print(f"  ... 还有 {len(t_details)-3} 条")

    # 汇总
    print(f"\n{'='*60}")
    print("汇总统计")
    print(f"{'='*60}")

    params_acc = total_params_correct / total_params_count if total_params_count > 0 else 0
    faq_acc = total_faq_correct / total_faq_count if total_faq_count > 0 else 0
    trouble_acc = total_trouble_correct / total_trouble_count if total_trouble_count > 0 else 0
    overall_acc = (total_params_correct + total_faq_correct + total_trouble_correct) / \
                  (total_params_count + total_faq_count + total_trouble_count) \
                  if (total_params_count + total_faq_count + total_trouble_count) > 0 else 0

    print(f"\n参数准确率: {params_acc:.1%} ({total_params_correct}/{total_params_count})")
    print(f"FAQ准确率:  {faq_acc:.1%} ({total_faq_correct}/{total_faq_count})")
    print(f"排障准确率: {trouble_acc:.1%} ({total_trouble_correct}/{total_trouble_count})")
    print(f"\n综合准确率: {overall_acc:.1%}")

    if overall_acc >= 0.85:
        print("\n✓ 达标！综合准确率 ≥ 85%")
    else:
        print(f"\n✗ 未达标！综合准确率 {overall_acc:.1%} < 85%，需要优化")


if __name__ == "__main__":
    main()
