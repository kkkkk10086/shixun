"""
工具模块：装饰器
实现函数耗时统计与日志记录
"""

import time
import functools
import logging
from datetime import datetime

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("rag-system")


def timer(func):
    """函数耗时统计装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time

        # 格式化耗时
        if elapsed < 1:
            time_str = f"{elapsed*1000:.1f}ms"
        elif elapsed < 60:
            time_str = f"{elapsed:.2f}s"
        else:
            time_str = f"{elapsed/60:.1f}min"

        logger.info(f"[耗时] {func.__name__} 执行完成: {time_str}")
        return result
    return wrapper


def log_call(func):
    """函数调用日志装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 记录输入参数（截断过长内容）
        args_str = str(args)[:100] if args else ""
        kwargs_str = str(kwargs)[:100] if kwargs else ""
        logger.info(f"[调用] {func.__name__}({args_str} {kwargs_str})")

        try:
            result = func(*args, **kwargs)
            logger.info(f"[完成] {func.__name__} → 成功")
            return result
        except Exception as e:
            logger.error(f"[错误] {func.__name__} → {type(e).__name__}: {e}")
            raise
    return wrapper


def timer_and_log(func):
    """组合装饰器：耗时统计 + 调用日志"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        args_str = str(args)[:80] if args else ""
        logger.info(f"[调用] {func.__name__}({args_str}...)")

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            time_str = f"{elapsed*1000:.1f}ms" if elapsed < 1 else f"{elapsed:.2f}s"
            logger.info(f"[完成] {func.__name__} → 成功 ({time_str})")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[错误] {func.__name__} → {type(e).__name__}: {e} ({elapsed:.2f}s)")
            raise
    return wrapper
