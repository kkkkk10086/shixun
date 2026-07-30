"""
模块：审核与发布中心
多级审核（产品/技术/品牌/版本）+ 发布/下架管理
"""

import json
import os
import uuid
from datetime import datetime
from typing import List, Optional


# 存储路径
REVIEW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_data")
REVIEW_FILE = os.path.join(REVIEW_DATA_DIR, "reviews.json")


class ReviewRecord:
    """审核记录"""
    def __init__(
        self,
        review_id: str = "",
        target_type: str = "",       # video | sop | document
        target_id: str = "",          # 被审核对象的ID
        target_title: str = "",       # 被审核对象的标题
        product_line: str = "",
        product_model: str = "",
        reviews: Optional[List[dict]] = None,  # [{"reviewer": "", "type": "", "status": "", "comment": "", "date": ""}]
        overall_status: str = "pending",  # pending | in_review | approved | rejected | published | archived
        published_at: str = "",
        created_at: str = "",
        updated_at: str = "",
    ):
        self.review_id = review_id or f"REVIEW-{uuid.uuid4().hex[:8].upper()}"
        self.target_type = target_type
        self.target_id = target_id
        self.target_title = target_title
        self.product_line = product_line
        self.product_model = product_model
        self.reviews = reviews or []
        self.overall_status = overall_status
        self.published_at = published_at
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.created_at = created_at or now
        self.updated_at = updated_at or now

    def to_dict(self) -> dict:
        return {
            "review_id": self.review_id,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "target_title": self.target_title,
            "product_line": self.product_line,
            "product_model": self.product_model,
            "reviews": self.reviews,
            "overall_status": self.overall_status,
            "published_at": self.published_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(d: dict):
        return ReviewRecord(
            review_id=d.get("review_id", ""),
            target_type=d.get("target_type", ""),
            target_id=d.get("target_id", ""),
            target_title=d.get("target_title", ""),
            product_line=d.get("product_line", ""),
            product_model=d.get("product_model", ""),
            reviews=d.get("reviews", []),
            overall_status=d.get("overall_status", "pending"),
            published_at=d.get("published_at", ""),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )


# 审核类型定义
REVIEW_TYPES = [
    {"type": "product", "name": "产品审核", "description": "确认内容准确、符合产品实际情况"},
    {"type": "technical", "name": "技术审核", "description": "确认技术参数、操作步骤正确"},
    {"type": "brand", "name": "品牌审核", "description": "确认品牌形象、用语规范"},
    {"type": "version", "name": "版本审核", "description": "确认适用固件/软件版本"},
]


class ReviewCenter:
    """审核与发布中心管理"""

    def __init__(self):
        os.makedirs(REVIEW_DATA_DIR, exist_ok=True)
        self._records: List[ReviewRecord] = []
        self._load()

    def _load(self):
        if os.path.exists(REVIEW_FILE):
            try:
                with open(REVIEW_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._records = [ReviewRecord.from_dict(d) for d in data]
            except Exception:
                self._records = []
        else:
            self._records = []
            self._save()

    def _save(self):
        data = [r.to_dict() for r in self._records]
        with open(REVIEW_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ---- 审核流程管理 ----

    def create_review(self, target_type: str, target_id: str, target_title: str,
                      product_line: str = "", product_model: str = "") -> ReviewRecord:
        """创建审核任务，自动生成4道审核"""
        record = ReviewRecord(
            target_type=target_type,
            target_id=target_id,
            target_title=target_title,
            product_line=product_line,
            product_model=product_model,
            reviews=[
                {
                    "type": rt["type"],
                    "name": rt["name"],
                    "status": "pending",
                    "reviewer": "",
                    "comment": "",
                    "date": "",
                }
                for rt in REVIEW_TYPES
            ],
            overall_status="in_review",
        )
        self._records.append(record)
        self._save()
        return record

    def submit_review(self, review_id: str, review_type: str,
                      status: str, reviewer: str = "", comment: str = "") -> Optional[ReviewRecord]:
        """
        提交某项审核结果
        review_type: product | technical | brand | version
        status: approved | rejected
        """
        record = self._get_record(review_id)
        if not record:
            return None

        for r in record.reviews:
            if r["type"] == review_type:
                r["status"] = status
                r["reviewer"] = reviewer
                r["comment"] = comment
                r["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                break

        # 检查是否所有审核都已完成
        statuses = [r["status"] for r in record.reviews]
        all_approved = all(s == "approved" for s in statuses)
        any_rejected = any(s == "rejected" for s in statuses)

        if any_rejected:
            record.overall_status = "rejected"
        elif all_approved:
            record.overall_status = "approved"

        record.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save()
        return record

    def publish(self, review_id: str) -> Optional[ReviewRecord]:
        """发布（仅approved状态可发布）"""
        record = self._get_record(review_id)
        if not record or record.overall_status != "approved":
            return None

        record.overall_status = "published"
        record.published_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record.updated_at = record.published_at
        self._save()
        return record

    def unpublish(self, review_id: str) -> Optional[ReviewRecord]:
        """下架"""
        record = self._get_record(review_id)
        if not record or record.overall_status != "published":
            return None

        record.overall_status = "archived"
        record.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save()
        return record

    # ---- 查询 ----

    def list_records(self, status: str = "", target_type: str = "",
                     product_line: str = "") -> List[ReviewRecord]:
        """列出审核记录"""
        results = []
        for r in self._records:
            if status and r.overall_status != status:
                continue
            if target_type and r.target_type != target_type:
                continue
            if product_line and r.product_line != product_line:
                continue
            results.append(r)
        # 按更新时间倒序
        results.sort(key=lambda x: x.updated_at, reverse=True)
        return results

    def get_record(self, review_id: str) -> Optional[ReviewRecord]:
        return self._get_record(review_id)

    def get_record_by_target(self, target_type: str, target_id: str) -> Optional[ReviewRecord]:
        for r in self._records:
            if r.target_type == target_type and r.target_id == target_id:
                return r
        return None

    def _get_record(self, review_id: str) -> Optional[ReviewRecord]:
        for r in self._records:
            if r.review_id == review_id:
                return r
        return None

    # ---- 统计 ----

    def get_stats(self) -> dict:
        """审核中心统计"""
        by_status = {}
        by_type = {}
        for r in self._records:
            by_status[r.overall_status] = by_status.get(r.overall_status, 0) + 1
            by_type[r.target_type] = by_type.get(r.target_type, 0) + 1

        # 各审核环节统计
        review_progress = {rt["type"]: {"total": 0, "approved": 0, "rejected": 0, "pending": 0}
                          for rt in REVIEW_TYPES}
        for r in self._records:
            for rev in r.reviews:
                t = rev["type"]
                if t in review_progress:
                    review_progress[t]["total"] += 1
                    review_progress[t][rev["status"]] = review_progress[t].get(rev["status"], 0) + 1

        return {
            "total": len(self._records),
            "by_status": by_status,
            "by_type": by_type,
            "review_progress": review_progress,
        }


# 全局实例
review_center = ReviewCenter()


def init_demo_reviews():
    """初始化示例审核记录"""
    if review_center.list_records():
        return

    # 已发布示例
    r1 = review_center.create_review(
        target_type="video", target_id="VIDEO-DICT-001",
        target_title="X8 Pro 首次开机与Wi-Fi连接",
        product_line="词典笔", product_model="X8 Pro",
    )
    for rt in ["product", "technical", "brand", "version"]:
        review_center.submit_review(r1.review_id, rt, "approved", "审核员")
    review_center.publish(r1.review_id)

    # 待审核示例
    r2 = review_center.create_review(
        target_type="video", target_id="VIDEO-LEARN-001",
        target_title="S90 Pro AI精准学功能使用",
        product_line="学习机", product_model="S90 Pro",
    )
    review_center.submit_review(r2.review_id, "product", "approved", "产品经理")
    review_center.submit_review(r2.review_id, "technical", "approved", "技术专家")

    print(f"  [审核中心] 已初始化示例数据")


if __name__ == "__main__":
    init_demo_reviews()
    stats = review_center.get_stats()
    print(f"审核记录总数: {stats['total']}")
    print(f"状态分布: {stats['by_status']}")
    print(f"审核进度: {stats['review_progress']}")