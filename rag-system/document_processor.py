"""
模块1：文档处理管线（语义分块版）
任意格式文档 → MarkItDown转换 → Markdown文本 → 语义分块 + 元数据标注
"""
import os, re, asyncio, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from markitdown import MarkItDown
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SECTION_SIZE, CHUNK_TABLE_MAX_ROWS, CHUNK_MIN_SIZE, OUTPUT_DIR
from decorators import timer_and_log

_executor = ThreadPoolExecutor(max_workers=4)

# 产品关键词映射表（用于元数据标注）
PRODUCT_KEYWORDS = {
    "录音笔": ["录音笔", "SR702", "SR502", "SR302", "S6 Plus", "S6系列", "S8离线版", "H1 Pro", "Magic", "Pokee"],
    "翻译机": ["翻译机", "双屏翻译机", "翻译设备"],
    "办公本": ["办公本", "X2", "X2LAMY", "墨水屏", "智能办公本"],
    "录音卡": ["录音卡", "AIR2611"],
    "词典笔": ["词典笔", "X8 Pro", "X9 Pro", "翻译笔"],
    "学习机": ["学习机", "S90 Pro", "S90"],
    "英语宝": ["英语宝", "EBOX", "听力宝"],
    "键盘": ["键盘", "T8星火版", "T8"],
    "鼠标": ["鼠标", "AM50", "AM50Pro"],
}

# ——— 内容质量过滤 ———

# 噪声关键词：包含这些关键词的块视为无效内容
NOISE_PATTERNS = [
    # 隐私政策 / 用户协议
    r"隐私政策", r"个人信息", r"用户协议", r"注销账户", r"未成年人",
    r"Cookie", r"数据安全", r"免责声明", r"法律",
    # 物流/快递
    r"快递", r"物流", r"顺丰", r"发货", r"退换货",
    # 代码片段
    r"private String", r"public class", r"JWT", r"import java",
    r"@Autowired", r"@Service", r"@RestController",
    # 非产品内容标识
    r"JavaWeb", r"比赛安排", r"项目立项", r"软件创新大赛",
    r"张家界学院", r"开题报告",
]

# 纯分隔符/标题行模式（这些块只有标题，没有实际内容）
TITLE_ONLY_PATTERN = re.compile(
    r'^(#+\s[^\n]+)\s*$',  # 只有标题行
    re.MULTILINE
)

# 纯分隔符/空行模式
SEPARATOR_ONLY = re.compile(r'^[\s\-=|#*]+$')


def _filter_noise_content(text: str) -> str:
    """
    过滤文本中的噪声内容
    返回过滤后的文本，如果整段都是噪声则返回空字符串
    """
    if not text or len(text.strip()) < CHUNK_MIN_SIZE:
        return ""

    # 检查是否为纯分隔符/空行/标题
    stripped = text.strip()
    if SEPARATOR_ONLY.match(stripped):
        return ""

    # 检查是否为纯标题（只有一行标题，没有正文）
    lines = [l.strip() for l in stripped.split("\n") if l.strip()]
    if len(lines) == 1 and lines[0].startswith("#"):
        # 单行标题，没有实际内容
        return ""

    # 检查噪声关键词
    for pattern in NOISE_PATTERNS:
        if re.search(pattern, text):
            # 如果整个块都是噪声内容，丢弃
            noise_lines = [l for l in lines if re.search(pattern, l)]
            useful_lines = [l for l in lines if not re.search(pattern, l)]
            if len(useful_lines) < max(2, len(lines) * 0.3):
                # 噪声行超过70%，整块丢弃
                return ""
            # 否则只移除噪声行
            return "\n".join(useful_lines)

    return stripped


def _is_valid_chunk(text: str) -> bool:
    """
    判断一个文本块是否有效
    无效块：噪声内容、纯标题、纯分隔符、过短内容
    """
    cleaned = _filter_noise_content(text)
    if not cleaned:
        return False

    # 去除标题后的实际内容长度
    content_without_heading = re.sub(r'^#+\s[^\n]+\n?', '', cleaned).strip()
    if len(content_without_heading) < 20:
        # 去掉标题后几乎没有内容，视为无效
        return False

    return True


def _clean_chunk_text(text: str) -> str:
    """
    清理块文本，去除噪声行
    """
    return _filter_noise_content(text)

# ——— 数据结构 ———

@dataclass
class Chunk:
    """带元数据的文档块"""
    text: str
    source: str = ""
    heading: str = ""           # 所属标题（如 "## 1. 讯飞AI录音笔系列"）
    subheading: str = ""        # 子标题（如 "### SR702 星火版"）
    product: str = ""           # 检测到的产品类别
    chunk_type: str = "text"    # "text" | "table" | "product_spec"

    # 扩展元数据字段
    brand: str = ""             # 品牌（如 "科大讯飞"）
    product_model: str = ""     # 产品型号（如 "SR702", "X2"）
    firmware_version: str = ""  # 固件版本
    document_name: str = ""     # 文档名称（如 "讯飞AI录音笔用户手册"）
    document_version: str = ""  # 文档版本
    chapter: str = ""           # 章节
    page_number: str = ""       # 页码
    content_type: str = ""      # 内容类型（如 "操作指南"、"规格参数"、"常见问题"）
    risk_level: str = ""        # 风险等级（"低"、"中"、"高"）
    effective_status: str = ""  # 生效状态（"生效"、"已废弃"）

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def metadata(self) -> dict:
        return {
            "source": self.source,
            "heading": self.heading,
            "subheading": self.subheading,
            "product": self.product,
            "chunk_type": self.chunk_type,
            "brand": self.brand,
            "product_model": self.product_model,
            "firmware_version": self.firmware_version,
            "document_name": self.document_name or self.source,
            "document_version": self.document_version,
            "chapter": self.chapter,
            "page_number": self.page_number,
            "content_type": self.content_type or self.chunk_type,
            "risk_level": self.risk_level,
            "effective_status": self.effective_status,
        }


# ——— 文档转换 ———

def convert_to_markdown(file_path: str) -> str:
    """将文档转换为 Markdown（MinerU → MarkItDown 降级）"""
    try:
        from mineru_parser import parse_document
        print(f"    使用 MinerU 解析...")
        return parse_document(file_path)
    except Exception as e:
        print(f"    MinerU 失败: {e}，降级使用 MarkItDown")
        md_converter = MarkItDown()
        result = md_converter.convert(file_path)
        return result.text_content


# ——— 产品类型检测 ———

def detect_product(text: str) -> str:
    """从文本中检测产品类型"""
    text_lower = text  # 中文无需 lower
    for product, keywords in PRODUCT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return product
    return ""


# ——— 语义分块 ———

def split_text_semantic(markdown_text: str, source_name: str = "") -> List[Chunk]:
    """
    语义分块：按标题层级拆分为语义完整的块，并标注元数据

    策略：
    1. 提取表格 → 超大表格按行组分拆 → 保留表头
    2. 按 ## 标题分割 → 每个标题一个语义块
    3. 超长块的按 ### 子标题再拆
    4. 剩余文本按段落切分
    5. 每块标注 source / heading / product
    """
    chunks: List[Chunk] = []
    remaining = markdown_text

    # ——— 第1步：提取表格 ———
    table_pattern = r'(\|.+\|[\n\r]+\|[-| :]+\|[\n\r]+(?:\|.+\|[\n\r]*)+)'
    tables_found = []

    def extract_tables(text: str) -> str:
        nonlocal tables_found
        tables_found = []
        for m in re.finditer(table_pattern, text):
            tables_found.append(m.group(1).strip())
        for t in tables_found:
            text = text.replace(t, "", 1)
        return text

    remaining = extract_tables(remaining)

    # 处理表格：整表保留，超大表按行组拆分
    for table_text in tables_found:
        if not _is_valid_chunk(table_text):
            continue
        lines = table_text.strip().split("\n")
        if len(lines) < 3:  # 至少需要表头+分隔线+数据行
            continue
        header = lines[0] + "\n" + lines[1]  # 表头 + 分隔线
        data_lines = lines[2:]
        product = detect_product(table_text) or detect_product(source_name)

        if len(data_lines) <= CHUNK_TABLE_MAX_ROWS:
            c = Chunk(text=table_text.strip(), source=source_name,
                      heading="(表格)", chunk_type="table", product=product,
                      document_name=source_name, brand="科大讯飞" if product else "")
            chunks.append(c)
        else:
            # 超大表格：按行组分拆，每份带表头
            for i in range(0, len(data_lines), CHUNK_TABLE_MAX_ROWS):
                group = data_lines[i:i + CHUNK_TABLE_MAX_ROWS]
                subtable = header + "\n" + "\n".join(group)
                if not _is_valid_chunk(subtable):
                    continue
                c = Chunk(text=subtable.strip(), source=source_name,
                          heading="(表格)", chunk_type="table", product=product,
                          document_name=source_name, brand="科大讯飞" if product else "")
                chunks.append(c)

    # ——— 第2步：按 ## 标题分割 ———
    sections = re.split(r'(?=^##\s)', remaining, flags=re.MULTILINE)
    for section in sections:
        section = section.strip()
        if len(section) < CHUNK_MIN_SIZE:
            continue

        # 提取标题
        heading_match = re.match(r'(##\s[^\n]+)', section)
        heading = heading_match.group(1).strip() if heading_match else ""
        heading_text = heading.lstrip("#").strip() if heading else ""
        product = detect_product(section) or detect_product(source_name)

        # ——— 第3步：检查是否需要按 ### 子标题拆分 ———
        if len(section) > CHUNK_SECTION_SIZE:
            # 如果有 ### 子标题，按子标题拆
            subsections = re.split(r'(?=^###\s)', section, flags=re.MULTILINE)
            if len(subsections) > 1:
                for sub in subsections:
                    sub = sub.strip()
                    if len(sub) < CHUNK_MIN_SIZE:
                        continue
                    if not _is_valid_chunk(sub):
                        continue
                    sub_match = re.match(r'(###\s[^\n]+)', sub)
                    subheading = sub_match.group(1).strip() if sub_match else ""
                    sub_product = detect_product(sub) or product
                    # 在块文本前加上标题和产品上下文
                    chunk_text = sub
                    if heading and heading not in sub[:50]:
                        chunk_text = heading + "\n" + sub
                    if sub_product and sub_product not in chunk_text[:100]:
                        chunk_text = f"[产品: {sub_product}]\n" + chunk_text
                    c = Chunk(text=chunk_text, source=source_name,
                              heading=heading, subheading=subheading,
                              chunk_type="product_spec" if heading_text else "text",
                              product=sub_product,
                              document_name=source_name, brand="科大讯飞" if sub_product else "")
                    chunks.append(c)
                continue  # 已处理，跳过后续
            else:
                # 无子标题但超长 → 按段落拆分
                paragraphs = [p.strip() for p in re.split(r'\n\n+', section) if p.strip()]
                if len(paragraphs) > 2:
                    # 合并小段落，保持语义完整
                    mini_chunks = []
                    buf = ""
                    for p in paragraphs:
                        if len(p) < 50:
                            buf += p + "\n"
                            continue
                        if buf:
                            p = buf + p
                            buf = ""
                        if len(mini_chunks) == 0 and heading:
                            p = heading + "\n" + p
                        if _is_valid_chunk(p):
                            mini_chunks.append(p)
                    for mc in mini_chunks:
                        if len(mc) >= CHUNK_MIN_SIZE:
                            chunk_text = mc
                            if heading and heading not in mc[:50]:
                                chunk_text = heading + "\n" + mc
                            if product and product not in chunk_text[:100]:
                                chunk_text = f"[产品: {product}]\n" + chunk_text
                            c = Chunk(text=chunk_text, source=source_name,
                                      heading=heading, chunk_type="text", product=product,
                                      document_name=source_name, brand="科大讯飞" if product else "")
                            chunks.append(c)
                    continue

        # 正常大小的 section，直接作为一个块
        chunk_type = "product_spec" if heading_text else "text"
        if _is_valid_chunk(section):
            # 在块文本前加上标题和产品上下文，提高检索召回
            chunk_text = section
            if heading and heading not in section[:50]:
                chunk_text = heading + "\n" + section
            if product and product not in chunk_text[:100]:
                chunk_text = f"[产品: {product}]\n" + chunk_text
            c = Chunk(text=chunk_text, source=source_name,
                      heading=heading, chunk_type=chunk_type, product=product,
                      document_name=source_name, brand="科大讯飞" if product else "")
            chunks.append(c)

    # ——— 最终过滤：清理所有块中的噪声内容 ———
    final_chunks = []
    for c in chunks:
        cleaned_text = _clean_chunk_text(c.text)
        if cleaned_text and len(cleaned_text) >= CHUNK_MIN_SIZE:
            c.text = cleaned_text
            final_chunks.append(c)

    # 打印过滤统计
    removed = len(chunks) - len(final_chunks)
    if removed > 0:
        print(f"    质量过滤: 移除 {removed} 个无效块，保留 {len(final_chunks)} 个")

    return final_chunks


# ——— 单文件处理 ———

@timer_and_log
def _process_single_file(file_path: str) -> tuple:
    """处理单个文件，返回 (文件名, Chunk列表)"""
    try:
        file_name = Path(file_path).name
        print(f"  → 处理: {file_name}")

        start_time = time.time()
        markdown_text = convert_to_markdown(file_path)
        convert_time = time.time() - start_time
        print(f"    转换完成 ({len(markdown_text)} 字符, {convert_time:.1f}秒)")

        # 语义分块
        chunks = split_text_semantic(markdown_text, source_name=file_name)
        print(f"    分块完成 ({len(chunks)} 块)")

        # 打印元数据概况
        products = set(c.product for c in chunks if c.product)
        types = set(c.chunk_type for c in chunks)
        print(f"    产品: {products or '未识别'} | 类型: {types}")

        return file_name, chunks
    except Exception as e:
        print(f"    处理失败: {e}")
        return Path(file_path).name, []


# ——— 文件收集 ———

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


# ——— 批量处理（兼容旧接口） ———

def process_documents(doc_paths: list) -> dict:
    """
    同步处理文档列表
    返回: {filename: [Chunk, Chunk, ...], ...}
    每个 Chunk 有 .text 和 .metadata 属性
    """
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
    """异步并发处理多个文件"""
    all_files = _collect_files(doc_paths)
    print(f"共发现 {len(all_files)} 个文件，启动异步并发处理...")

    start_time = time.time()
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(_executor, _process_single_file, f) for f in all_files]
    results = await asyncio.gather(*tasks)

    all_chunks = {}
    for name, chunks in results:
        if chunks:
            all_chunks[name] = chunks

    total = sum(len(c) for c in all_chunks.values())
    print(f"\n并发处理完成: {len(all_chunks)} 个文档, {total} 块, 耗时 {time.time()-start_time:.1f}秒")
    return all_chunks


# ——— 保存 ———

def save_chunks(all_chunks: dict, output_dir: str = OUTPUT_DIR):
    """保存分块结果（含完整元数据）"""
    os.makedirs(output_dir, exist_ok=True)
    for doc_name, chunks in all_chunks.items():
        output_path = os.path.join(output_dir, f"{doc_name}_chunks.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks):
                meta = chunk.metadata if hasattr(chunk, 'metadata') else {}
                meta_str = " | ".join(f"{k}={v}" for k, v in meta.items() if v)
                text = chunk.text if hasattr(chunk, 'text') else (chunk.get('text') if isinstance(chunk, dict) else chunk)
                f.write(f"--- Chunk {i+1} [{meta_str}] ---\n{text}\n\n")
        print(f"  已保存分块: {output_path}")


def save_markdown(all_chunks: dict, output_dir: str = OUTPUT_DIR):
    """保存 Markdown 转换结果（含扩展元数据）"""
    os.makedirs(output_dir, exist_ok=True)
    for doc_name, chunks in all_chunks.items():
        md_path = os.path.join(output_dir, f"{doc_name}_markdown.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {doc_name}\n\n")
            for chunk in chunks:
                text = chunk.text if hasattr(chunk, 'text') else (chunk.get('text') if isinstance(chunk, dict) else chunk)
                meta = chunk.metadata if hasattr(chunk, 'metadata') else {}
                meta_parts = [f"{k}={v}" for k, v in meta.items() if v and k not in ("source",)]
                if meta_parts:
                    f.write(f"> {' | '.join(meta_parts)}\n\n")
                f.write(text + "\n\n---\n\n")
        print(f"  已保存Markdown: {md_path}")
