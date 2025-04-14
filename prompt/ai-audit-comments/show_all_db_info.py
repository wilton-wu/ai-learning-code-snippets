#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SQLite数据库完整信息显示工具
自动显示所有表的结构和数据
"""

import sqlite3
import sys
from pathlib import Path
from typing import Union, List, Tuple

BASE_PATH = Path(__file__).parent


def show_table_structure(cursor: sqlite3.Cursor, table_name: str) -> List[Tuple]:
    """显示表结构信息"""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns_info = cursor.fetchall()

    print("\n表结构:")
    print("-" * 50)
    for col in columns_info:
        _, col_name, col_type, not_null, default_val, pk = col
        print(f"列名: {col_name}")
        print(f"  类型: {col_type}")
        print(f"  非空: {'是' if not_null else '否'}")
        print(f"  默认值: {default_val if default_val else '无'}")
        print(f"  主键: {'是' if pk else '否'}")
        print()

    return columns_info


def show_table_data(
    cursor: sqlite3.Cursor, table_name: str, columns_info: List[Tuple]
) -> None:
    """显示表数据"""
    # 获取总行数
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    total_rows = cursor.fetchone()[0]

    # 获取前10行数据
    cursor.execute(f"SELECT * FROM {table_name} LIMIT 10")
    rows = cursor.fetchall()

    print(f"表数据 (总行数: {total_rows}):")
    print("-" * 50)

    if not rows:
        print("表中没有数据")
        return

    # 获取列名并打印
    column_names = [col[1] for col in columns_info]
    print(" | ".join(column_names))
    print("-" * 50)

    # 打印数据行
    for row in rows:
        row_str = [
            cell[:30] + "..." if isinstance(cell, str) and len(cell) > 30 else str(cell)
            for cell in row
        ]
        print(" | ".join(row_str))

    if total_rows > 10:
        print(f"... (还有 {total_rows - 10} 行未显示)")


def show_all_db_info(db_path: Union[str, Path]) -> None:
    """显示数据库中所有表的结构和数据"""
    db_path = Path(db_path)

    # 检查数据库文件是否存在
    if not db_path.exists():
        print(f"数据库文件 '{db_path}' 不存在!")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 获取所有表名
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        if not tables:
            print(f"数据库 '{db_path}' 中没有表!")
            return

        # 显示表概览
        print(f"数据库 '{db_path}' 中共有 {len(tables)} 个表:")
        for i, (table_name,) in enumerate(tables, 1):
            print(f"{i}. {table_name}")

        # 遍历每个表显示详细信息
        for (table_name,) in tables:
            print("\n" + "=" * 50)
            print(f"表名: {table_name}")
            print("=" * 50)

            # 显示表结构
            columns_info = show_table_structure(cursor, table_name)

            # 显示表数据
            show_table_data(cursor, table_name, columns_info)

        print("\n数据库分析完成")

    except sqlite3.Error as e:
        print(f"SQLite错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        if "conn" in locals():
            conn.close()


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else BASE_PATH / "comments.db"
    show_all_db_info(db_path)
