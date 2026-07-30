"""
模块：SOP 生成器 + 场景模板定制
实现文档 8.1/8.2 节 SOP 结构和 10.4 节场景模板
"""

import json
import uuid
from typing import List, Optional
from datetime import datetime


# ========== SOP 数据模型（8.1节） ==========

class SOPStep:
    """SOP 单个步骤"""
    def __init__(self, step_no: int, action: str, operation_target: str = "",
                 expected_result: str = "", visual_requirement: str = "",
                 source_document: str = "", source_page: int = 0,
                 warning: str = ""):
        self.step_no = step_no
        self.action = action
        self.operation_target = operation_target
        self.expected_result = expected_result
        self.visual_requirement = visual_requirement
        self.source_document = source_document
        self.source_page = source_page
        self.warning = warning

    def to_dict(self):
        return {
            "step_no": self.step_no,
            "action": self.action,
            "operation_target": self.operation_target,
            "expected_result": self.expected_result,
            "visual_requirement": self.visual_requirement,
            "source_document": self.source_document,
            "source_page": self.source_page,
            "warning": self.warning,
        }


class SOP:
    """标准作业程序（8.1节）"""
    def __init__(self, sop_id: str = "", product_line: str = "",
                 product_model: str = "", title: str = "",
                 applicable_version: str = "", prerequisites: Optional[List[str]] = None,
                 warnings: Optional[List[str]] = None,
                 steps: Optional[List[SOPStep]] = None,
                 completion_check: Optional[List[str]] = None,
                 common_errors: Optional[List[dict]] = None,
                 created_at: str = "", updated_at: str = ""):
        self.sop_id = sop_id or f"SOP-{uuid.uuid4().hex[:8].upper()}"
        self.product_line = product_line
        self.product_model = product_model
        self.title = title
        self.applicable_version = applicable_version
        self.prerequisites = prerequisites or []
        self.warnings = warnings or []
        self.steps = steps or []
        self.completion_check = completion_check or []
        self.common_errors = common_errors or []
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.created_at = created_at or now
        self.updated_at = updated_at or now

    def to_dict(self):
        return {
            "sop_id": self.sop_id,
            "product_line": self.product_line,
            "product_model": self.product_model,
            "title": self.title,
            "applicable_version": self.applicable_version,
            "prerequisites": self.prerequisites,
            "warnings": self.warnings,
            "steps": [s.to_dict() for s in self.steps],
            "completion_check": self.completion_check,
            "common_errors": self.common_errors,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# ========== 场景模板定制（10.4节） ==========

SCENE_TEMPLATES = {
    "title": {
        "name": "ProductTitleScene",
        "description": "产品名称、型号和视频标题",
        "template": {"type": "title", "duration": 3, "text": "", "subtitle": ""},
    },
    "hardware_action": {
        "name": "HardwareActionScene",
        "description": "真实产品操作演示",
        "template": {"type": "hardware_video", "asset": "", "voice": "", "subtitle": ""},
    },
    "screen_operation": {
        "name": "ScreenOperationScene",
        "description": "学习机或 App 录屏操作",
        "template": {"type": "screen_record", "asset": "", "voice": "", "subtitle": ""},
    },
    "scan_demo": {
        "name": "ScanDemonstrationScene",
        "description": "词典笔扫描演示",
        "template": {"type": "hardware_video", "asset": "", "voice": "", "subtitle": ""},
    },
    "translation_conv": {
        "name": "TranslationConversationScene",
        "description": "翻译机场景对话",
        "template": {"type": "animation", "asset": "", "voice": "", "subtitle": ""},
    },
    "warning": {
        "name": "WarningScene",
        "description": "安全警告或错误操作",
        "template": {"type": "overlay", "text": "", "voice": "", "subtitle": ""},
    },
    "result_check": {
        "name": "ResultCheckScene",
        "description": "展示操作完成状态",
        "template": {"type": "image", "asset": "", "voice": "", "subtitle": ""},
    },
    "troubleshooting": {
        "name": "TroubleshootingScene",
        "description": "展示故障现象和处理方法",
        "template": {"type": "split_screen", "asset_left": "", "asset_right": "", "voice": "", "subtitle": ""},
    },
}


# ========== SOP 生成 ==========

def generate_sop_from_docs(query: str, product_desc: str, doc_context: str) -> SOP:
    """
    使用 LLM 从知识库文档中提取 SOP（8.2节规则）
    返回 SOP 对象
    """
    from openai import OpenAI
    from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
    llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    prompt = f"""你是一个 SOP（标准作业程序）生成专家。请根据以下产品文档内容，生成结构化的 SOP。

产品：{product_desc}
用户需求：{query}

文档参考：
{doc_context}

【SOP 生成规则（8.2节）】
1. 必须严格基于文档内容，不得自行补充文档中不存在的操作
2. 每个步骤必须包含：明确动作、操作对象、预期结果、文档来源
3. 涉及风险时必须添加警告
4. 涉及版本差异时需要有适用说明

请输出严格的 JSON 格式 SOP（不要包含其他文字）：
{{
    "title": "操作标题",
    "applicable_version": "适用版本",
    "prerequisites": ["前置条件1", "前置条件2"],
    "warnings": ["警告1"],
    "steps": [
        {{
            "step_no": 1,
            "action": "动作描述",
            "operation_target": "操作对象",
            "expected_result": "预期结果",
            "visual_requirement": "画面要求",
            "source_document": "文档来源",
            "source_page": 0,
            "warning": ""
        }}
    ],
    "completion_check": ["完成标志1"],
    "common_errors": [
        {{"symptom": "故障现象", "solution": "解决方法"}}
    ]
}}"""

    try:
        response = llm.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
            temperature=0.3
        )
        result_text = response.choices[0].message.content.strip()

        # 清理 markdown 标记
        if result_text.startswith("```"):
            result_text = result_text.split("\n", 1)[1]
            if result_text.endswith("```"):
                result_text = result_text[:-3].strip()
            result_text = result_text.strip()

        data = json.loads(result_text)

        steps = []
        for i, s in enumerate(data.get("steps", [])):
            steps.append(SOPStep(
                step_no=s.get("step_no", i + 1),
                action=s.get("action", ""),
                operation_target=s.get("operation_target", ""),
                expected_result=s.get("expected_result", ""),
                visual_requirement=s.get("visual_requirement", ""),
                source_document=s.get("source_document", ""),
                source_page=s.get("source_page", 0),
                warning=s.get("warning", ""),
            ))

        return SOP(
            product_line=product_desc.split(" ")[0] if " " in product_desc else product_desc,
            product_model=product_desc.split(" ")[-1] if " " in product_desc else "",
            title=data.get("title", ""),
            applicable_version=data.get("applicable_version", ""),
            prerequisites=data.get("prerequisites", []),
            warnings=data.get("warnings", []),
            steps=steps,
            completion_check=data.get("completion_check", []),
            common_errors=data.get("common_errors", []),
        )
    except Exception as e:
        print(f"  [SOP生成] 失败: {e}")
        return SOP(title=f"{product_desc} 操作指南")


# ========== SOP → OpenMAIC 场景转换 ==========

def sop_to_openmaic_scenes(sop: SOP, scene_type: str = "hardware_action") -> list:
    """
    将 SOP 步骤转换为 OpenMAIC 场景序列
    根据 10.4 节场景模板生成
    """
    scenes = []

    # 标题场景
    title_template = SCENE_TEMPLATES["title"]["template"].copy()
    title_template["text"] = sop.title
    title_template["subtitle"] = f"{sop.product_line} {sop.product_model}"
    scenes.append(title_template)

    # 前置条件场景（如果有）
    if sop.prerequisites:
        prereq_text = "前置条件：" + "；".join(sop.prerequisites)
        scenes.append({
            "type": "overlay",
            "text": prereq_text,
            "voice": prereq_text,
            "subtitle": prereq_text,
            "duration": max(3, len(sop.prerequisites) * 2),
        })

    # 警告场景（如果有）
    for warning in sop.warnings:
        scenes.append({
            "type": "overlay",
            "text": f"⚠️ {warning}",
            "voice": warning,
            "subtitle": warning,
            "duration": 4,
        })

    # 操作步骤场景
    base_template = SCENE_TEMPLATES.get(scene_type, SCENE_TEMPLATES["hardware_action"])
    for step in sop.steps:
        scene = base_template["template"].copy()
        scene["voice"] = f"第{step.step_no}步：{step.action}"
        scene["subtitle"] = f"步骤{step.step_no}：{step.action}"
        if "duration" not in scene:
            scene["duration"] = 5
        scenes.append(scene)

        # 预期结果检查
        if step.expected_result:
            result_scene = SCENE_TEMPLATES["result_check"]["template"].copy()
            result_scene["voice"] = f"确认：{step.expected_result}"
            result_scene["subtitle"] = f"确认：{step.expected_result}"
            scenes.append(result_scene)

    # 完成检查场景
    if sop.completion_check:
        check_text = "完成检查：" + "；".join(sop.completion_check)
        scenes.append({
            "type": "overlay",
            "text": check_text,
            "voice": check_text,
            "subtitle": check_text,
            "duration": 4,
        })

    # 常见错误场景
    for error in sop.common_errors:
        scenes.append({
            "type": "split_screen",
            "asset_left": "",
            "asset_right": "",
            "voice": f"常见问题：{error.get('symptom', '')}。解决方法：{error.get('solution', '')}",
            "subtitle": f"❓ {error.get('symptom', '')} → {error.get('solution', '')}",
            "duration": 6,
        })

    return scenes


def build_openmaic_request(sop: SOP, scene_type: str = "hardware_action") -> dict:
    """
    构建 OpenMAIC 视频生成请求
    包含 SOP 结构化输入和场景模板
    """
    scenes = sop_to_openmaic_scenes(sop, scene_type)

    return {
        "requirement": f"生成{sop.product_line}{sop.product_model}的{sop.title}教学视频",
        "scenes": scenes,
        "sop": sop.to_dict(),
        "enableVideoGeneration": False,
        "enableTTS": True,
        "enableImageGeneration": True,
        "enableWebSearch": False,
        "scene_template": scene_type,
    }