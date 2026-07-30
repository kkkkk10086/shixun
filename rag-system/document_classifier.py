"""
模块：文档分类标签（6.1节）
9种文档类型标签 + 自动分类逻辑
"""

import re
from typing import List, Dict

# 9种文档类型标签（6.1节）
DOCUMENT_TYPES = {
    "操作手册": "包含产品操作步骤、使用方法、按键说明等",
    "规格参数": "包含产品技术参数、尺寸、重量、配置等",
    "常见问题FAQ": "包含常见问题及解答",
    "故障排除": "包含故障现象、原因分析、解决方法",
    "培训资料": "包含产品培训、教学指导内容",
    "宣传推广": "包含产品介绍、卖点、促销信息",
    "售后政策": "包含保修、退换货、维修政策",
    "合规认证": "包含3C认证、入网许可证、合规文件",
    "开发文档": "包含SDK、API接口、开发指南",
}

# 每种类型的检测关键词
TYPE_KEYWORDS: Dict[str, List[str]] = {
    "操作手册": [
        "操作步骤", "使用方法", "按键说明", "功能介绍", "操作指南",
        "使用说明", "如何", "步骤", "按", "长按", "短按", "单击", "双击",
        "打开", "关闭", "设置", "连接", "安装", "启动", "操作",
    ],
    "规格参数": [
        "规格参数", "技术参数", "产品参数", "尺寸", "重量", "分辨率",
        "电池", "容量", "内存", "存储", "处理器", "屏幕", "接口",
        "产品规格", "参数", "型号", "版本",
    ],
    "常见问题FAQ": [
        "常见问题", "FAQ", "问:", "答:", "Q:", "A:", "问题解答",
        "疑问解答", "常见疑问",
    ],
    "故障排除": [
        "故障", "排除", "无法", "不能", "失败", "错误", "异常",
        "不工作", "没反应", "解决", "排查", "检查",
    ],
    "培训资料": [
        "培训", "教学", "课程", "学习目标", "知识点", "讲解",
        "教程", "指导", "课堂",
    ],
    "宣传推广": [
        "新品", "上市", "首发", "促销", "优惠", "购买", "抢购",
        "卖点", "亮点", "推荐", "选择", "为什么选",
    ],
    "售后政策": [
        "保修", "退换货", "维修", "售后", "三包", "质保",
        "退货", "换货", "客服", "服务政策",
    ],
    "合规认证": [
        "认证", "3C", "入网", "许可证", "合规", "标准",
        "GB/T", "CCC", "SRRC", "CMA",
    ],
    "开发文档": [
        "SDK", "API", "接口", "开发", "接入", "调用",
        "参数说明", "返回", "请求", "响应",
    ],
}

# 排除关键词（避免误判）
EXCLUDE_KEYWORDS: Dict[str, List[str]] = {
    "操作手册": ["保修", "退货", "认证"],
    "规格参数": ["故障", "保修"],
    "宣传推广": ["故障", "错误代码"],
}


def classify_document(text: str, title: str = "") -> str:
    """
    根据文档内容自动分类（6.1节规则）
    返回最匹配的文档类型标签
    
    策略：
    1. 先检查标题中的关键词
    2. 按内容中的关键词匹配得分排序
    3. 排除关键词降低误判
    """
    combined = (title + "\n" + text).lower()
    scores = {}

    for doc_type, keywords in TYPE_KEYWORDS.items():
        score = 0
        for kw in keywords:
            count = combined.count(kw.lower())
            if count > 0:
                score += count * 2 if kw in title.lower() else count

        # 排除关键词扣分
        exclude_list = EXCLUDE_KEYWORDS.get(doc_type, [])
        for ek in exclude_list:
            if ek.lower() in combined:
                score -= 3

        if score > 0:
            scores[doc_type] = score

    if not scores:
        return "操作手册"  # 默认类型

    # 按得分排序
    sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_types[0][0]


def classify_document_batch(chunks: List[dict]) -> List[str]:
    """
    批量分类文档块
    输入: chunks列表，每个元素是包含text和title的dict
    输出: 分类标签列表
    """
    results = []
    for chunk in chunks:
        text = chunk.get("text", "")
        title = chunk.get("title", chunk.get("heading", ""))
        doc_type = classify_document(text, title)
        results.append(doc_type)
    return results


def get_document_type_stats(chunks) -> dict:
    """统计文档类型分布"""
    type_counts = {t: 0 for t in DOCUMENT_TYPES}
    for chunk_type in chunks:
        if chunk_type in type_counts:
            type_counts[chunk_type] += 1
    return {
        "total": sum(type_counts.values()),
        "distribution": type_counts,
    }


if __name__ == "__main__":
    # 测试
    test_texts = [
        ("产品规格参数", "尺寸：150mm × 80mm × 12mm\n重量：200g\n电池：4000mAh"),
        ("操作步骤", "1. 长按电源键3秒开机\n2. 进入设置菜单\n3. 选择Wi-Fi网络"),
        ("常见问题", "Q: 无法开机怎么办？\nA: 请先充电30分钟再尝试开机"),
        ("新品上市", "新品首发！限时优惠，立即购买"),
    ]
    for title, text in test_texts:
        result = classify_document(text, title)
        print(f"标题: {title} → 分类: {result}")