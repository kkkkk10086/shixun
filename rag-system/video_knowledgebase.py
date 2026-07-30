"""
模块：视频知识库
视频元数据管理 + 章节管理 + 文件存储
遵循文档 12.1/12.2 节定义的视频元数据与章节结构
"""

import os
import json
import uuid
from datetime import datetime
from typing import List, Optional

# 视频存储路径
VIDEO_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_data")
VIDEO_META_FILE = os.path.join(VIDEO_DATA_DIR, "videos.json")
VIDEO_FILES_DIR = os.path.join(VIDEO_DATA_DIR, "files")


# ========== 数据模型 ==========

class VideoChapter:
    """视频章节（12.2节）"""
    def __init__(self, start_time: int, end_time: int, title: str):
        self.start_time = start_time
        self.end_time = end_time
        self.title = title

    def to_dict(self):
        return {"start_time": self.start_time, "end_time": self.end_time, "title": self.title}

    @staticmethod
    def from_dict(d):
        return VideoChapter(d["start_time"], d["end_time"], d["title"])


class Video:
    """视频元数据（12.1节）"""
    def __init__(
        self,
        video_id: str = "",
        product_line: str = "",
        product_model: str = "",
        firmware_version: str = "",
        title: str = "",
        duration: int = 0,
        video_url: str = "",
        thumbnail_url: str = "",
        applicable_questions: Optional[List[str]] = None,
        chapters: Optional[List[VideoChapter]] = None,
        review_status: str = "draft",
        created_at: str = "",
        updated_at: str = "",
    ):
        self.video_id = video_id or f"VIDEO-{uuid.uuid4().hex[:8].upper()}"
        self.product_line = product_line
        self.product_model = product_model
        self.firmware_version = firmware_version
        self.title = title
        self.duration = duration
        self.video_url = video_url
        self.thumbnail_url = thumbnail_url
        self.applicable_questions = applicable_questions or []
        self.chapters = chapters or []
        self.review_status = review_status  # draft | approved | published | archived
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.created_at = created_at or now
        self.updated_at = updated_at or now

    def to_dict(self):
        return {
            "video_id": self.video_id,
            "product_line": self.product_line,
            "product_model": self.product_model,
            "firmware_version": self.firmware_version,
            "title": self.title,
            "duration": self.duration,
            "video_url": self.video_url,
            "thumbnail_url": self.thumbnail_url,
            "applicable_questions": self.applicable_questions,
            "chapters": [c.to_dict() for c in self.chapters],
            "review_status": self.review_status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(d):
        chapters = [VideoChapter.from_dict(c) for c in d.get("chapters", [])]
        return Video(
            video_id=d.get("video_id", ""),
            product_line=d.get("product_line", ""),
            product_model=d.get("product_model", ""),
            firmware_version=d.get("firmware_version", ""),
            title=d.get("title", ""),
            duration=d.get("duration", 0),
            video_url=d.get("video_url", ""),
            thumbnail_url=d.get("thumbnail_url", ""),
            applicable_questions=d.get("applicable_questions", []),
            chapters=chapters,
            review_status=d.get("review_status", "draft"),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )


# ========== 存储管理 ==========

class VideoStore:
    """视频知识库存储（JSON文件持久化）"""

    def __init__(self):
        os.makedirs(VIDEO_DATA_DIR, exist_ok=True)
        os.makedirs(VIDEO_FILES_DIR, exist_ok=True)
        self._videos: List[Video] = []
        self._load()

    def _load(self):
        """从 JSON 文件加载视频数据"""
        if os.path.exists(VIDEO_META_FILE):
            try:
                with open(VIDEO_META_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._videos = [Video.from_dict(d) for d in data]
                print(f"  [视频库] 加载 {len(self._videos)} 个视频")
            except Exception as e:
                print(f"  [视频库] 加载失败: {e}")
                self._videos = []
        else:
            self._videos = []
            self._save()

    def _save(self):
        """保存视频数据到 JSON 文件"""
        data = [v.to_dict() for v in self._videos]
        with open(VIDEO_META_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ---- CRUD ----

    def list_videos(self, product_line: str = "", product_model: str = "", status: str = "") -> List[Video]:
        """列出视频，支持按产品线/型号/状态过滤"""
        results = []
        for v in self._videos:
            if product_line and v.product_line != product_line:
                continue
            if product_model and v.product_model != product_model:
                continue
            if status and v.review_status != status:
                continue
            results.append(v)
        return results

    def get_video(self, video_id: str) -> Optional[Video]:
        """按 ID 获取视频"""
        for v in self._videos:
            if v.video_id == video_id:
                return v
        return None

    def search_videos(self, query: str, product_line: str = "") -> List[Video]:
        """按关键词搜索视频（标题 + 适用问题 + 章节）"""
        query_lower = query.lower()
        results = []
        for v in self._videos:
            if product_line and v.product_line != product_line:
                continue
            # 搜索标题
            if query_lower in v.title.lower():
                results.append(v)
                continue
            # 搜索适用问题
            for q in v.applicable_questions:
                if query_lower in q.lower():
                    results.append(v)
                    break
            else:
                # 搜索章节标题
                for c in v.chapters:
                    if query_lower in c.title.lower():
                        results.append(v)
                        break
        return results

    def add_video(self, video: Video) -> Video:
        """添加视频"""
        self._videos.append(video)
        self._save()
        return video

    def update_video(self, video_id: str, updates: dict) -> Optional[Video]:
        """更新视频信息"""
        video = self.get_video(video_id)
        if not video:
            return None

        for key, value in updates.items():
            if hasattr(video, key) and key not in ("video_id", "created_at"):
                setattr(video, key, value)

        video.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save()
        return video

    def delete_video(self, video_id: str) -> bool:
        """删除视频"""
        for i, v in enumerate(self._videos):
            if v.video_id == video_id:
                self._videos.pop(i)
                self._save()
                return True
        return False

    # ---- 章节管理 ----

    def add_chapter(self, video_id: str, chapter: VideoChapter) -> Optional[Video]:
        """添加章节"""
        video = self.get_video(video_id)
        if not video:
            return None
        video.chapters.append(chapter)
        video.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save()
        return video

    def update_chapter(self, video_id: str, chapter_index: int, chapter: VideoChapter) -> Optional[Video]:
        """更新章节"""
        video = self.get_video(video_id)
        if not video or chapter_index < 0 or chapter_index >= len(video.chapters):
            return None
        video.chapters[chapter_index] = chapter
        video.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save()
        return video

    def delete_chapter(self, video_id: str, chapter_index: int) -> Optional[Video]:
        """删除章节"""
        video = self.get_video(video_id)
        if not video or chapter_index < 0 or chapter_index >= len(video.chapters):
            return None
        video.chapters.pop(chapter_index)
        video.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save()
        return video

    # ---- 视频推荐（12.3节召回规则） ----

    def recommend_videos(self, query: str, product_line: str = "", product_model: str = "", top_k: int = 3) -> List[Video]:
        """
        根据用户问题和产品上下文推荐相关视频（12.3节召回规则）

        硬过滤条件：
        - 产品线必须匹配
        - 型号必须匹配（如果指定）
        - 视频必须已发布 (published)

        排序依据：
        - 适用问题匹配（最高权重）
        - 视频标题匹配
        - 章节标题匹配
        """
        import jieba
        query_lower = query.lower().strip()

        # 步骤1：硬过滤 — 只保留已发布、产品线匹配的视频
        candidates = []
        for v in self._videos:
            if v.review_status != "published":
                continue
            if product_line and v.product_line != product_line:
                continue
            if product_model and v.product_model != product_model:
                continue
            candidates.append(v)

        if not candidates:
            return []

        # 步骤2：关键词提取
        try:
            keywords = list(jieba.analyse.extract_tags(query, topK=10))
        except Exception:
            keywords = [query_lower]

        # 步骤3：评分排序
        scored = []
        for v in candidates:
            score = 0.0

            # 适用问题匹配（最高权重，每个匹配 +5）
            for q in v.applicable_questions:
                q_lower = q.lower()
                # 完全匹配
                if query_lower == q_lower:
                    score += 10
                # 包含关系
                elif query_lower in q_lower or q_lower in query_lower:
                    score += 5
                # 关键词匹配
                else:
                    for kw in keywords:
                        if kw in q_lower:
                            score += 3
                            break

            # 视频标题匹配（权重中等，每个匹配 +3）
            title_lower = v.title.lower()
            if query_lower == title_lower:
                score += 6
            elif query_lower in title_lower or title_lower in query_lower:
                score += 4
            else:
                for kw in keywords:
                    if kw in title_lower:
                        score += 2
                        break

            # 章节标题匹配（权重较低，每个匹配 +1）
            for c in v.chapters:
                ch_title_lower = c.title.lower()
                if query_lower == ch_title_lower:
                    score += 3
                elif query_lower in ch_title_lower or ch_title_lower in query_lower:
                    score += 2
                else:
                    for kw in keywords:
                        if kw in ch_title_lower:
                            score += 1
                            break

            if score > 0:
                scored.append((score, v))

        # 按分数降序排序
        scored.sort(key=lambda x: x[0], reverse=True)

        # 返回 top_k 个结果
        return [v for _, v in scored[:top_k]]

    # ---- 统计 ----

    def get_stats(self) -> dict:
        """获取视频库统计"""
        total = len(self._videos)
        by_product = {}
        by_status = {}
        for v in self._videos:
            by_product[v.product_line] = by_product.get(v.product_line, 0) + 1
            by_status[v.review_status] = by_status.get(v.review_status, 0) + 1
        return {
            "total": total,
            "by_product": by_product,
            "by_status": by_status,
        }


# 全局实例
video_store = VideoStore()


# ========== 示例视频数据 ==========

def init_demo_videos():
    """初始化示例视频数据（首次使用时添加）"""
    if video_store.list_videos():
        return  # 已有数据，跳过

    demos = [
        Video(
            video_id="VIDEO-DICT-001",
            product_line="词典笔",
            product_model="X8 Pro",
            firmware_version="V2.0",
            title="X8 Pro 首次开机与Wi-Fi连接",
            duration=65,
            video_url="/video_data/files/demo_wifi.mp4",
            applicable_questions=[
                "怎么开机",
                "如何连接Wi-Fi",
                "首次使用怎么设置",
                "怎么联网",
            ],
            chapters=[
                VideoChapter(0, 15, "开机与初始化"),
                VideoChapter(15, 40, "进入Wi-Fi设置"),
                VideoChapter(40, 65, "输入密码并连接"),
            ],
            review_status="published",
        ),
        Video(
            video_id="VIDEO-DICT-002",
            product_line="词典笔",
            product_model="X8 Pro",
            firmware_version="V2.0",
            title="X8 Pro 扫描查词操作指南",
            duration=80,
            video_url="/video_data/files/demo_scan.mp4",
            applicable_questions=[
                "怎么扫描单词",
                "扫描不准确怎么办",
                "如何查词",
                "扫描角度怎么掌握",
            ],
            chapters=[
                VideoChapter(0, 20, "握笔姿势与扫描角度"),
                VideoChapter(20, 50, "单次扫描查词"),
                VideoChapter(50, 80, "整句扫描与翻译"),
            ],
            review_status="published",
        ),
        Video(
            video_id="VIDEO-TRANS-001",
            product_line="翻译机",
            product_model="双屏翻译机2.0",
            firmware_version="V1.5",
            title="双屏翻译机 离线翻译设置",
            duration=85,
            video_url="/video_data/files/demo_offline.mp4",
            applicable_questions=[
                "怎么使用离线翻译",
                "没有网络能不能翻译",
                "怎么下载离线语言包",
                "离线翻译怎么设置",
            ],
            chapters=[
                VideoChapter(0, 12, "进入离线语言管理"),
                VideoChapter(12, 38, "下载语言包"),
                VideoChapter(38, 60, "启用离线翻译"),
                VideoChapter(60, 85, "测试离线翻译效果"),
            ],
            review_status="published",
        ),
        Video(
            video_id="VIDEO-LEARN-001",
            product_line="学习机",
            product_model="S90 Pro",
            firmware_version="V3.0",
            title="S90 Pro AI精准学功能使用",
            duration=120,
            video_url="/video_data/files/demo_ai_study.mp4",
            applicable_questions=[
                "AI精准学怎么用",
                "怎么进行学情诊断",
                "如何生成学习报告",
                "精准学适合什么年级",
            ],
            chapters=[
                VideoChapter(0, 20, "进入AI精准学"),
                VideoChapter(20, 50, "完成诊断测试"),
                VideoChapter(50, 80, "查看知识图谱"),
                VideoChapter(80, 120, "针对薄弱点练习"),
            ],
            review_status="published",
        ),
    ]

    for v in demos:
        video_store.add_video(v)
    print(f"  [视频库] 已添加 {len(demos)} 个示例视频")


if __name__ == "__main__":
    print("视频知识库管理工具")
    init_demo_videos()
    stats = video_store.get_stats()
    print(f"视频总数: {stats['total']}")
    print(f"产品分布: {stats['by_product']}")
    print(f"状态分布: {stats['by_status']}")