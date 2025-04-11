import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 创建logger实例
logger = logging.getLogger(__name__)

# 中文显示配置 - 使用更好的中文字体支持
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Arial"]
plt.rcParams["font.size"] = 12


def get_next_file_number(base_name, output_dir, extension=".png"):
    """
    获取下一个可用的文件编号
    Args:
        base_name: 基础文件名
        output_dir: 输出目录
        extension: 文件扩展名
    Returns:
        下一个可用的文件编号
    """
    # 获取目录中所有文件
    files = list(Path(output_dir).glob(f"{base_name}_*{extension}"))

    if not files:
        return 1

    # 提取现有文件的编号
    numbers = []
    for file in files:
        try:
            # 从文件名中提取编号
            num_str = file.stem.split("_")[-1]
            if num_str.isdigit():
                numbers.append(int(num_str))
        except (ValueError, IndexError):
            continue

    # 返回最大编号+1，如果没有编号则返回1
    return max(numbers, default=0) + 1


def plot_top_ngrams(
    ngrams: List[Tuple[str, int]],
    figsize=(12, 10),
    output_dir=".",
    color_palette="viridis",
    dpi=100,
):
    """
    可视化TopN高频n-gram词汇
    Args:
        ngrams: 包含(词汇, 频次)的元组列表
        figsize: 图表尺寸
        output_dir: 输出目录路径
        color_palette: 颜色主题
        dpi: 图像分辨率
    """
    if not ngrams:
        logger.warning("没有n-gram数据可供可视化")
        return

    try:
        # 创建图表
        plt.figure(figsize=figsize, dpi=dpi)

        # 准备数据
        df = pd.DataFrame(ngrams, columns=["ngram", "count"])

        # 按频次降序排序
        df = df.sort_values("count", ascending=False)

        # 创建柱状图 - 按照新的API要求设置参数
        # 将y变量同时用作hue，并关闭图例
        ax = sns.barplot(
            x="count",
            y="ngram",
            hue="ngram",  # 使用y变量作为hue
            data=df,
            palette=color_palette,
            legend=False,  # 关闭图例
        )

        # 在柱状图上添加数值标签
        for i, v in enumerate(df["count"]):
            ax.text(v + 0.5, i, str(v), va="center")

        # 设置标题和标签
        plt.title("高频n-gram词汇分布", fontsize=16, pad=20)
        plt.xlabel("出现频次", fontsize=12)
        plt.ylabel("n-gram组合", fontsize=12)

        # 调整布局
        plt.tight_layout()

        # 获取下一个可用的文件编号
        file_number = get_next_file_number("top_ngrams", output_dir)
        output_path = Path(output_dir) / f"top_ngrams_{file_number}.png"

        # 保存图表
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"高频n-gram词汇分布图已保存至: {output_path}")

        plt.close()
    except Exception as e:
        logger.error(f"可视化生成失败: {str(e)}")


def plot_similarity_matrix(
    matrix: np.ndarray,
    hotel_names: List[str],
    figsize=(14, 12),
    output_dir=".",
    max_items=15,
    cmap="YlGnBu",
    dpi=100,
):
    """
    可视化酒店相似度矩阵
    Args:
        matrix: 相似度矩阵
        hotel_names: 酒店名称列表
        figsize: 图表尺寸
        output_dir: 输出目录路径
        max_items: 最大显示项目数
        cmap: 热图颜色主题
        dpi: 图像分辨率
    """
    if matrix.size == 0 or len(hotel_names) == 0:
        logger.warning("没有相似度数据可供可视化")
        return

    try:
        # 清理酒店名称中的特殊字符
        clean_names = [name.replace("\xa0", " ").strip() for name in hotel_names]

        # 限制显示的酒店数量，避免图表过于拥挤
        n_items = min(max_items, len(clean_names), matrix.shape[0])

        # 创建图表
        plt.figure(figsize=figsize, dpi=dpi)

        # 截取名称和矩阵
        display_names = clean_names[:n_items]
        display_matrix = matrix[:n_items, :n_items]

        # 创建热图
        mask = np.triu(np.ones_like(display_matrix, dtype=bool))  # 只显示下三角
        sns.heatmap(
            display_matrix,
            xticklabels=display_names,
            yticklabels=display_names,
            cmap=cmap,
            annot=True,
            fmt=".2f",
            mask=mask,
            cbar_kws={"label": "相似度分数"},
        )

        # 设置标题和标签
        plt.title("酒店特征相似度矩阵", fontsize=16, pad=20)

        # 调整标签角度，提高可读性
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(rotation=0, fontsize=10)

        # 调整布局
        plt.tight_layout()

        # 获取下一个可用的文件编号
        file_number = get_next_file_number("similarity_matrix", output_dir)
        output_path = Path(output_dir) / f"similarity_matrix_{file_number}.png"

        # 保存图表
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"相似度矩阵图已保存至: {output_path}")

        plt.close()
    except Exception as e:
        logger.error(f"相似度矩阵可视化失败: {str(e)}")
