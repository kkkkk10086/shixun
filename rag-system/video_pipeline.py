"""
模块：视频内容生产流水线（2.3节）
文档 → RAG → 操作任务识别 → SOP → 脚本 → 素材匹配 → OpenMAIC → 审核 → 发布
"""

import json
import os
import uuid
from datetime import datetime
from typing import List, Optional, Dict
from enum import Enum

# 存储路径
PIPELINE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_data")
PIPELINE_FILE = os.path.join(PIPELINE_DATA_DIR, "pipeline.json")


class PipelineStage(str, Enum):
    """流水线阶段"""
    DOCUMENT_INGEST = "document_ingest"        # 文档导入
    RAG_RETRIEVAL = "rag_retrieval"            # RAG检索
    TASK_IDENTIFICATION = "task_identification"  # 操作任务识别
    SOP_GENERATION = "sop_generation"          # SOP生成
    SCRIPT_WRITING = "script_writing"          # 脚本撰写
    MATERIAL_MATCHING = "material_matching"    # 素材匹配
    OPENMAIC_GENERATION = "openmaic_generation"  # OpenMAIC生成
    REVIEW = "review"                          # 审核
    PUBLISH = "publish"                        # 发布


# 产品线视频模板（3.1~3.4节）
PRODUCT_LINE_TEMPLATES = {
    "学习机": {
        "scene_types": ["screen_operation", "hardware_action", "result_check", "troubleshooting"],
        "default_scene": "screen_operation",
        "voice_style": "亲切教学",
        "subtitle_style": "清晰分步",
        "duration_per_step": 8,
        "description": "学习机以录屏操作为主，配合真人演示，重点展示学习功能和交互效果",
    },
    "翻译机": {
        "scene_types": ["translation_conv", "hardware_action", "screen_operation", "result_check"],
        "default_scene": "translation_conv",
        "voice_style": "标准清晰",
        "subtitle_style": "双语对照",
        "duration_per_step": 6,
        "description": "翻译机以场景对话为主，展示真实翻译场景，突出多语言能力",
    },
    "词典笔": {
        "scene_types": ["scan_demo", "screen_operation", "hardware_action", "result_check"],
        "default_scene": "scan_demo",
        "voice_style": "简洁明快",
        "subtitle_style": "显示扫描结果",
        "duration_per_step": 5,
        "description": "词典笔以扫描演示为主，突出扫描速度和准确率，配合屏幕显示",
    },
    "办公本": {
        "scene_types": ["hardware_action", "screen_operation", "result_check", "troubleshooting"],
        "default_scene": "hardware_action",
        "voice_style": "专业沉稳",
        "subtitle_style": "功能要点",
        "duration_per_step": 7,
        "description": "办公本以手写和语音交互为主，展示办公效率提升场景",
    },
    "录音笔": {
        "scene_types": ["hardware_action", "result_check", "troubleshooting"],
        "default_scene": "hardware_action",
        "voice_style": "清晰专业",
        "subtitle_style": "录音参数",
        "duration_per_step": 5,
        "description": "录音笔以硬件操作为主，展示录音效果和智能转写功能",
    },
    "英语宝": {
        "scene_types": ["screen_operation", "hardware_action", "result_check"],
        "default_scene": "screen_operation",
        "voice_style": "亲切活泼",
        "subtitle_style": "学习内容",
        "duration_per_step": 6,
        "description": "英语宝以听力练习和口语评测为主，展示学习互动场景",
    },
}


class PipelineTask:
    """流水线任务"""
    def __init__(
        self,
        task_id: str = "",
        title: str = "",
        product_line: str = "",
        product_model: str = "",
        query: str = "",             # 用户原始问题
        topic_id: str = "",          # 关联选题ID
        doc_context: str = "",       # RAG检索到的文档上下文
        sop_data: Optional[dict] = None,  # SOP数据
        script: str = "",            # 生成的脚本
        materials: Optional[List[dict]] = None,  # 匹配的素材
        openmaic_job_id: str = "",   # OpenMAIC任务ID
        openmaic_result: Optional[dict] = None,  # OpenMAIC调用结果
        review_id: str = "",         # 审核记录ID
        current_stage: str = PipelineStage.DOCUMENT_INGEST,
        status: str = "pending",     # pending | running | completed | failed
        error: str = "",
        created_at: str = "",
        updated_at: str = "",
    ):
        self.task_id = task_id or f"PIPE-{uuid.uuid4().hex[:8].upper()}"
        self.title = title
        self.product_line = product_line
        self.product_model = product_model
        self.query = query
        self.topic_id = topic_id
        self.doc_context = doc_context
        self.sop_data = sop_data
        self.script = script
        self.materials = materials or []
        self.openmaic_job_id = openmaic_job_id
        self.review_id = review_id
        self.current_stage = current_stage
        self.status = status
        self.error = error
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.created_at = created_at or now
        self.updated_at = updated_at or now

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "product_line": self.product_line,
            "product_model": self.product_model,
            "query": self.query,
            "topic_id": self.topic_id,
            "doc_context": self.doc_context[:500] if self.doc_context else "",
            "sop_data": self.sop_data,
            "script": self.script[:500] if self.script else "",
            "materials": self.materials,
            "openmaic_job_id": self.openmaic_job_id,
            "openmaic_result": getattr(self, "openmaic_result", None),
            "review_id": self.review_id,
            "current_stage": self.current_stage,
            "status": self.status,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(d: dict):
        task = PipelineTask(
            task_id=d.get("task_id", ""),
            title=d.get("title", ""),
            product_line=d.get("product_line", ""),
            product_model=d.get("product_model", ""),
            query=d.get("query", ""),
            topic_id=d.get("topic_id", ""),
            doc_context=d.get("doc_context", ""),
            sop_data=d.get("sop_data"),
            script=d.get("script", ""),
            materials=d.get("materials", []),
            openmaic_job_id=d.get("openmaic_job_id", ""),
            review_id=d.get("review_id", ""),
            current_stage=d.get("current_stage", PipelineStage.DOCUMENT_INGEST),
            status=d.get("status", "pending"),
            error=d.get("error", ""),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )
        # 兼容旧数据：从 dict 中恢复新字段（如果存在）
        if "openmaic_result" in d:
            task.openmaic_result = d["openmaic_result"]
        return task


class VideoPipeline:
    """视频内容生产流水线管理"""

    def __init__(self):
        os.makedirs(PIPELINE_DATA_DIR, exist_ok=True)
        self._tasks: List[PipelineTask] = []
        self._load()

    def _load(self):
        if os.path.exists(PIPELINE_FILE):
            try:
                with open(PIPELINE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._tasks = [PipelineTask.from_dict(d) for d in data]
            except Exception:
                self._tasks = []
        else:
            self._tasks = []
            self._save()

    def _save(self):
        data = [t.to_dict() for t in self._tasks]
        with open(PIPELINE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def create_task(self, title: str, product_line: str, product_model: str,
                    query: str = "", topic_id: str = "") -> PipelineTask:
        """创建流水线任务"""
        task = PipelineTask(
            title=title,
            product_line=product_line,
            product_model=product_model,
            query=query,
            topic_id=topic_id,
            current_stage=PipelineStage.DOCUMENT_INGEST,
            status="pending",
        )
        self._tasks.append(task)
        self._save()
        return task

    def advance_stage(self, task_id: str, next_stage: str, updates: dict = None) -> Optional[PipelineTask]:
        """推进到下一阶段"""
        task = self._get_task(task_id)
        if not task:
            return None

        task.current_stage = next_stage
        task.status = "running"
        if updates:
            for key, value in updates.items():
                if hasattr(task, key):
                    setattr(task, key, value)
        task.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save()
        return task

    def complete_task(self, task_id: str, updates: dict = None) -> Optional[PipelineTask]:
        """完成任务"""
        task = self._get_task(task_id)
        if not task:
            return None

        task.status = "completed"
        if updates:
            for key, value in updates.items():
                if hasattr(task, key):
                    setattr(task, key, value)
        task.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save()
        return task

    def fail_task(self, task_id: str, error: str) -> Optional[PipelineTask]:
        """标记任务失败"""
        task = self._get_task(task_id)
        if not task:
            return None
        task.status = "failed"
        task.error = error
        task.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save()
        return task

    def list_tasks(self, status: str = "", product_line: str = "") -> List[PipelineTask]:
        """列出流水线任务"""
        results = []
        for t in self._tasks:
            if status and t.status != status:
                continue
            if product_line and t.product_line != product_line:
                continue
            results.append(t)
        results.sort(key=lambda x: x.updated_at, reverse=True)
        return results

    def get_task(self, task_id: str) -> Optional[PipelineTask]:
        return self._get_task(task_id)

    def _get_task(self, task_id: str) -> Optional[PipelineTask]:
        for t in self._tasks:
            if t.task_id == task_id:
                return t
        return None

    def get_stats(self) -> dict:
        """流水线统计"""
        by_status = {}
        by_stage = {}
        by_product = {}
        for t in self._tasks:
            by_status[t.status] = by_status.get(t.status, 0) + 1
            by_stage[t.current_stage] = by_stage.get(t.current_stage, 0) + 1
            by_product[t.product_line] = by_product.get(t.product_line, 0) + 1
        return {
            "total": len(self._tasks),
            "by_status": by_status,
            "by_stage": by_stage,
            "by_product": by_product,
        }


# 全局实例
pipeline = VideoPipeline()


def get_product_template(product_line: str) -> dict:
    """获取产品线视频模板（3.1~3.4节）"""
    return PRODUCT_LINE_TEMPLATES.get(product_line, PRODUCT_LINE_TEMPLATES.get("学习机", {}))


if __name__ == "__main__":
    print("视频生产流水线管理")
    print(f"产品线模板: {list(PRODUCT_LINE_TEMPLATES.keys())}")
    for pl, tmpl in PRODUCT_LINE_TEMPLATES.items():
        print(f"  {pl}: {tmpl['description']}")