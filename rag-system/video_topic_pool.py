"""
模块：视频选题库（7.1/7.2节）
选题来源管理 + 优先级评分算法
"""

import json
import os
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict

# 存储路径
TOPIC_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_data")
TOPIC_FILE = os.path.join(TOPIC_DATA_DIR, "topics.json")


class VideoTopic:
    """视频选题（7.1节）"""
    def __init__(
        self,
        topic_id: str = "",
        title: str = "",
        product_line: str = "",
        product_model: str = "",
        source: str = "",           # 选题来源：rag_query | manual_input | faq_analysis | user_feedback | operation_doc
        source_detail: str = "",    # 来源详情，如"用户高频查询：价格"
        query_count: int = 0,       # 查询频次（7.2节评分因子）
        difficulty: str = "中",     # 制作难度：低/中/高
        estimated_duration: int = 60,  # 预计视频时长（秒）
        priority_score: float = 0.0,   # 优先级评分（7.2节算法）
        status: str = "pending",    # pending | planned | in_production | completed | archived
        sop_id: str = "",           # 关联的SOP ID
        video_id: str = "",         # 关联的视频ID
        created_at: str = "",
        updated_at: str = "",
    ):
        self.topic_id = topic_id or f"TOPIC-{uuid.uuid4().hex[:8].upper()}"
        self.title = title
        self.product_line = product_line
        self.product_model = product_model
        self.source = source
        self.source_detail = source_detail
        self.query_count = query_count
        self.difficulty = difficulty
        self.estimated_duration = estimated_duration
        self.priority_score = priority_score
        self.status = status
        self.sop_id = sop_id
        self.video_id = video_id
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.created_at = created_at or now
        self.updated_at = updated_at or now

    def to_dict(self) -> dict:
        return {
            "topic_id": self.topic_id,
            "title": self.title,
            "product_line": self.product_line,
            "product_model": self.product_model,
            "source": self.source,
            "source_detail": self.source_detail,
            "query_count": self.query_count,
            "difficulty": self.difficulty,
            "estimated_duration": self.estimated_duration,
            "priority_score": self.priority_score,
            "status": self.status,
            "sop_id": self.sop_id,
            "video_id": self.video_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(d: dict):
        return VideoTopic(
            topic_id=d.get("topic_id", ""),
            title=d.get("title", ""),
            product_line=d.get("product_line", ""),
            product_model=d.get("product_model", ""),
            source=d.get("source", ""),
            source_detail=d.get("source_detail", ""),
            query_count=d.get("query_count", 0),
            difficulty=d.get("difficulty", "中"),
            estimated_duration=d.get("estimated_duration", 60),
            priority_score=d.get("priority_score", 0.0),
            status=d.get("status", "pending"),
            sop_id=d.get("sop_id", ""),
            video_id=d.get("video_id", ""),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )


# 选题来源权重（7.2节评分因子）
SOURCE_WEIGHTS = {
    "rag_query": 1.0,          # RAG高频查询
    "manual_input": 0.7,       # 手动录入
    "faq_analysis": 0.9,       # FAQ分析
    "user_feedback": 0.8,      # 用户反馈
    "operation_doc": 0.6,      # 操作文档
}

# 难度系数
DIFFICULTY_FACTORS = {
    "低": 0.8,
    "中": 1.0,
    "高": 1.3,
}


def calculate_priority_score(
    query_count: int,
    source: str,
    difficulty: str,
    days_since_creation: int = 0,
    product_importance: float = 1.0,
) -> float:
    """
    优先级评分算法（7.2节）

    score = query_count * source_weight * difficulty_factor * importance * recency_boost

    - query_count: 查询频次
    - source_weight: 来源权重
    - difficulty_factor: 难度系数（越低越优先）
    - product_importance: 产品重要性（1.0基准）
    - recency_boost: 新选题加成（7天内+20%）
    """
    source_weight = SOURCE_WEIGHTS.get(source, 0.5)
    difficulty_factor = 1.0 / DIFFICULTY_FACTORS.get(difficulty, 1.0)

    # 新选题加成
    recency_boost = 1.2 if days_since_creation <= 7 else 1.0

    score = query_count * source_weight * difficulty_factor * product_importance * recency_boost
    return round(score, 2)


class TopicPool:
    """视频选题库管理"""

    def __init__(self):
        os.makedirs(TOPIC_DATA_DIR, exist_ok=True)
        self._topics: List[VideoTopic] = []
        self._load()

    def _load(self):
        if os.path.exists(TOPIC_FILE):
            try:
                with open(TOPIC_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._topics = [VideoTopic.from_dict(d) for d in data]
            except Exception:
                self._topics = []
        else:
            self._topics = []
            self._save()

    def _save(self):
        data = [t.to_dict() for t in self._topics]
        with open(TOPIC_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def list_topics(self, product_line: str = "", status: str = "", sort_by: str = "priority") -> List[VideoTopic]:
        """列出选题，支持过滤和排序"""
        results = []
        for t in self._topics:
            if product_line and t.product_line != product_line:
                continue
            if status and t.status != status:
                continue
            results.append(t)

        if sort_by == "priority":
            results.sort(key=lambda x: x.priority_score, reverse=True)
        elif sort_by == "created":
            results.sort(key=lambda x: x.created_at, reverse=True)
        elif sort_by == "query_count":
            results.sort(key=lambda x: x.query_count, reverse=True)

        return results

    def get_topic(self, topic_id: str) -> Optional[VideoTopic]:
        for t in self._topics:
            if t.topic_id == topic_id:
                return t
        return None

    def add_topic(self, topic: VideoTopic) -> VideoTopic:
        """添加选题并自动计算优先级评分"""
        # 计算优先级评分
        days_since = 0
        if topic.created_at:
            try:
                created = datetime.strptime(topic.created_at, "%Y-%m-%d %H:%M:%S")
                days_since = (datetime.now() - created).days
            except Exception:
                pass

        topic.priority_score = calculate_priority_score(
            query_count=topic.query_count,
            source=topic.source,
            difficulty=topic.difficulty,
            days_since_creation=days_since,
        )
        self._topics.append(topic)
        self._save()
        return topic

    def update_topic(self, topic_id: str, updates: dict) -> Optional[VideoTopic]:
        topic = self.get_topic(topic_id)
        if not topic:
            return None

        for key, value in updates.items():
            if hasattr(topic, key) and key not in ("topic_id", "created_at"):
                setattr(topic, key, value)

        # 重新计算优先级
        topic.priority_score = calculate_priority_score(
            query_count=topic.query_count,
            source=topic.source,
            difficulty=topic.difficulty,
        )
        topic.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save()
        return topic

    def delete_topic(self, topic_id: str) -> bool:
        for i, t in enumerate(self._topics):
            if t.topic_id == topic_id:
                self._topics.pop(i)
                self._save()
                return True
        return False

    def get_stats(self) -> dict:
        """选题库统计"""
        by_status = {}
        by_product = {}
        by_source = {}
        for t in self._topics:
            by_status[t.status] = by_status.get(t.status, 0) + 1
            by_product[t.product_line] = by_product.get(t.product_line, 0) + 1
            by_source[t.source] = by_source.get(t.source, 0) + 1
        return {
            "total": len(self._topics),
            "by_status": by_status,
            "by_product": by_product,
            "by_source": by_source,
        }


# 全局实例
topic_pool = TopicPool()


def init_demo_topics():
    """初始化示例选题"""
    if topic_pool.list_topics():
        return

    demos = [
        VideoTopic(
            title="X8 Pro 首次开机与Wi-Fi连接",
            product_line="词典笔",
            product_model="X8 Pro",
            source="rag_query",
            source_detail="用户高频查询：如何连接Wi-Fi",
            query_count=45,
            difficulty="低",
            estimated_duration=65,
            status="planned",
        ),
        VideoTopic(
            title="S90 Pro AI精准学功能使用",
            product_line="学习机",
            product_model="S90 Pro",
            source="faq_analysis",
            source_detail="FAQ分析：精准学使用方法",
            query_count=38,
            difficulty="中",
            estimated_duration=120,
            status="planned",
        ),
        VideoTopic(
            title="双屏翻译机 离线翻译设置",
            product_line="翻译机",
            product_model="双屏翻译机2.0",
            source="user_feedback",
            source_detail="用户反馈：离线翻译怎么用",
            query_count=22,
            difficulty="中",
            estimated_duration=85,
            status="pending",
        ),
        VideoTopic(
            title="办公本X2 手写笔记功能详解",
            product_line="办公本",
            product_model="X2",
            source="operation_doc",
            source_detail="操作文档：手写笔记功能",
            query_count=15,
            difficulty="高",
            estimated_duration=150,
            status="pending",
        ),
        VideoTopic(
            title="录音笔H1 Pro 录音模式切换",
            product_line="录音笔",
            product_model="H1 Pro",
            source="rag_query",
            source_detail="用户高频查询：录音模式",
            query_count=30,
            difficulty="低",
            estimated_duration=55,
            status="pending",
        ),
    ]

    for t in demos:
        topic_pool.add_topic(t)
    print(f"  [选题库] 已添加 {len(demos)} 个示例选题")


if __name__ == "__main__":
    init_demo_topics()
    stats = topic_pool.get_stats()
    print(f"选题总数: {stats['total']}")
    print(f"状态分布: {stats['by_status']}")
    print(f"来源分布: {stats['by_source']}")