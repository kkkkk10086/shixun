"""
模块4：MySQL 存储
将分块结果和元数据存入 MySQL 数据库
"""

import pymysql
from config import MYSQL_CONFIG


def get_connection():
    """获取 MySQL 连接"""
    return pymysql.connect(**MYSQL_CONFIG)


def init_database():
    """初始化数据库表"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INT AUTO_INCREMENT PRIMARY KEY,
            doc_name VARCHAR(255) NOT NULL,
            doc_path TEXT,
            chunk_count INT DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INT AUTO_INCREMENT PRIMARY KEY,
            doc_id INT,
            chunk_index INT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (doc_id) REFERENCES documents(id)
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("[MySQL] 数据库表初始化完成")


def save_chunks_to_mysql(all_chunks: dict):
    """将分块结果存入 MySQL"""
    conn = get_connection()
    cursor = conn.cursor()

    for doc_name, chunks in all_chunks.items():
        cursor.execute(
            "INSERT INTO documents (doc_name, chunk_count) VALUES (%s, %s)",
            (doc_name, len(chunks))
        )
        doc_id = cursor.lastrowid

        for i, chunk in enumerate(chunks):
            cursor.execute(
                "INSERT INTO chunks (doc_id, chunk_index, content) VALUES (%s, %s, %s)",
                (doc_id, i, chunk)
            )

    conn.commit()
    cursor.close()
    conn.close()
    print(f"[MySQL] 已存入 {len(all_chunks)} 个文档的分块数据")


def search_from_mysql(query: str, top_k: int = 5) -> list:
    """从 MySQL 中按关键词搜索"""
    conn = get_connection()
    cursor = conn.cursor()

    sql = """
        SELECT c.content, d.doc_name
        FROM chunks c
        JOIN documents d ON c.doc_id = d.id
        WHERE c.content LIKE %s
        LIMIT %s
    """
    cursor.execute(sql, (f"%{query}%", top_k))
    results = cursor.fetchall()

    cursor.close()
    conn.close()

    return [{"content": row[0], "source": row[1]} for row in results]


def get_stats():
    """获取数据库统计信息"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM documents")
    doc_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM chunks")
    chunk_count = cursor.fetchone()[0]

    cursor.close()
    conn.close()

    return {"doc_count": doc_count, "chunk_count": chunk_count}
