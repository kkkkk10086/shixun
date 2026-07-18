"""
模块1：文档处理管线
任意格式文档 → MarkItDown转换 → Markdown文本 → 智能分块
支持 asyncio 并发处理多个文件
"""

import os
import asyncio
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from markitdown import MarkItDown
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP, OUTPUT_DIR


# 线程池
_executor = ThreadPoolExecutor(max_workers=4)


def convert_to_markdown(file_path: str) -> str:
    """
    将文档转换为 Markdown
    全部使用 MinerU API 解析（支持 PDF、Word、图片等）
    """
    return _convert_with_mineru(file_path)


def _convert_with_mineru(file_path: str) -> str:
    """使用 MinerU API 解析文档"""
    try:
        from mineru_parser import parse_document
        print(f"    使用 MinerU 解析...")
        return parse_document(file_path)
    except Exception as e:
        print(f"    MinerU 失败: {e}，降级使用 MarkItDown")
        return _convert_with_markitdown(file_path)


def _convert_with_markitdown(file_path: str) -> str:
    """使用 MarkItDown 解析文档（降级方案）"""
    md_converter = MarkItDown()
    result = md_converter.convert(file_path)
    return result.text_content


def split_text(markdown_text: str) -> list:
    """将 Markdown 文本进行智能分块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", ".", " "],
        length_function=len
    )
    return splitter.split_text(markdown_text)


def _process_single_file(file_path: str) -> tuple:
    """处理单个文件，返回 (文件名, 分块列表)"""
    try:
        file_name = Path(file_path).name
        print(f"  → 处理: {file_name}")

        start_time = time.time()

        # 转换为 Markdown
        markdown_text = convert_to_markdown(file_path)
        convert_time = time.time() - start_time
        print(f"    转换完成 ({len(markdown_text)} 字符, {convert_time:.1f}秒)")

        # 智能分块
        chunks = split_text(markdown_text)
        print(f"    分块完成 ({len(chunks)} 块)")

        return file_name, chunks
    except Exception as e:
        print(f"    处理失败: {e}")
        return Path(file_path).name, []


def _collect_files(doc_paths: list) -> list:
    """收集所有需要处理的文件路径"""
    all_files = []
    for doc_path in doc_paths:
        path = Path(doc_path)
        if path.is_dir():
            for file in path.rglob("*"):
                if file.suffix.lower() in [".pdf", ".docx", ".doc", ".txt", ".md"]:
                    all_files.append(str(file))
        elif path.is_file():
            all_files.append(str(path))
    return all_files


def process_documents(doc_paths: list) -> dict:
    """同步处理文档列表（兼容旧接口）"""
    all_files = _collect_files(doc_paths)
    print(f"共发现 {len(all_files)} 个文件需要处理")

    all_chunks = {}
    for i, file_path in enumerate(all_files):
        print(f"\n[{i+1}/{len(all_files)}]")
        name, chunks = _process_single_file(file_path)
        if chunks:
            all_chunks[name] = chunks

    return all_chunks


async def process_documents_async(doc_paths: list) -> dict:
    """
    异步并发处理多个文件
    使用 asyncio.gather 同时处理多个文档
    """
    all_files = _collect_files(doc_paths)
    print(f"共发现 {len(all_files)} 个文件，启动异步并发处理...")

    start_time = time.time()

    # 使用线程池执行并发处理
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(_executor, _process_single_file, file_path)
        for file_path in all_files
    ]

    results = await asyncio.gather(*tasks)

    all_chunks = {}
    for name, chunks in results:
        if chunks:
            all_chunks[name] = chunks

    total_time = time.time() - start_time
    total_chunks = sum(len(c) for c in all_chunks.values())
    print(f"\n并发处理完成: {len(all_chunks)} 个文档, {total_chunks} 个块, 耗时 {total_time:.1f}秒")

    return all_chunks


def save_chunks(all_chunks: dict, output_dir: str = OUTPUT_DIR):
    """保存分块结果"""
    os.makedirs(output_dir, exist_ok=True)
    for doc_name, chunks in all_chunks.items():
        output_path = os.path.join(output_dir, f"{doc_name}_chunks.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                f.write(f"--- Chunk {i+1} ---\n{chunk}\n\n")
        print(f"  已保存分块: {output_path}")


def save_markdown(all_chunks: dict, output_dir: str = OUTPUT_DIR):
    """保存 Markdown 转换结果"""
    os.makedirs(output_dir, exist_ok=True)
    for doc_name, chunks in all_chunks.items():
        md_path = os.path.join(output_dir, f"{doc_name}_markdown.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {doc_name}\n\n")
            for chunk in chunks:
                f.write(chunk + "\n\n---\n\n")
        print(f"  已保存Markdown: {md_path}")
