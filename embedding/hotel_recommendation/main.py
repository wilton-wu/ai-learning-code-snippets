import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from data_processing import get_top_ngrams, load_data
from model import get_recommendations, train_model
import visualization as viz

# 配置日志和参数
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],  # 明确指定stdout，避免编码问题
)

# 获取项目根目录
BASE_DIR = Path(__file__).parent

# 配置参数
CONFIG = {
    "data_path": BASE_DIR / "Seattle_Hotels.csv",
    "ngram_params": {"ngram_range": (3, 3), "top_k": 20},
    "tfidf_params": {"ngram_range": (1, 3), "min_df": 0.01},
    "output_dir": BASE_DIR / "dist",
    "model_method": "tfidf",
    "top_ngrams": 15,
    "top_recommendations": 10,
}


def load_hotel_data() -> pd.DataFrame:
    """加载酒店数据并进行预处理"""
    if not CONFIG["data_path"].exists():
        logger.error(f"数据文件不存在: {CONFIG['data_path']}")
        sys.exit(1)

    logger.info(f"开始加载数据: {CONFIG['data_path']}")
    df = load_data(CONFIG["data_path"])

    if df.empty:
        logger.warning("加载的数据集为空")
        sys.exit(1)

    df["name"] = df["name"].str.replace("\xa0", " ").str.strip()
    logger.info(f"数据加载完成，共 {len(df)} 条记录")
    return df


def analyze_features(df: pd.DataFrame, output_dir: Path) -> None:
    """分析酒店特征，提取高频n-gram"""
    logger.info("分析酒店描述文本特征...")
    common_words = get_top_ngrams(df["desc_clean"], **CONFIG["ngram_params"])
    viz.plot_top_ngrams(common_words, output_dir=output_dir, dpi=150)


def train_recommendation_models(df: pd.DataFrame) -> Tuple[pd.Series, dict]:
    """训练推荐模型"""
    logger.info("训练推荐模型...")
    _, indices, tfidf_sim = train_model(df, text_column="desc_clean", method="tfidf")
    _, _, count_sim = train_model(df, text_column="desc_clean", method="count")
    return indices, {"TF-IDF": tfidf_sim, "Count": count_sim}


def generate_recommendations(
    hotel_name: str, indices: pd.Series, cosine_sim: np.ndarray, output_dir: Path
) -> List[str]:
    """生成示例推荐"""
    try:
        logger.info(f'为酒店 "{hotel_name}" 生成推荐...')
        recommendations = get_recommendations(
            hotel_name, indices, cosine_sim, top_n=CONFIG["top_recommendations"]
        )

        # 可视化相似度矩阵
        logger.info("生成相似度矩阵可视化...")
        all_hotels = list(indices.values)
        viz.plot_similarity_matrix(
            cosine_sim, all_hotels, output_dir=output_dir, dpi=150
        )

        logger.info(f"推荐生成完成，共 {len(recommendations)} 条推荐")
        return recommendations
    except Exception as e:
        logger.error(f"推荐生成失败: {str(e)}")
        raise


def main() -> None:
    """主函数"""
    start_time = time.time()

    try:
        # 确保输出目录存在
        output_dir = CONFIG["output_dir"]
        output_dir.mkdir(exist_ok=True)

        # 加载数据
        df = load_hotel_data()

        # 特征分析
        analyze_features(df, output_dir)

        # 训练模型
        indices, method_comparison = train_recommendation_models(df)

        # 为示例酒店生成推荐
        example_hotel = df["name"].iloc[0]
        recommendations = generate_recommendations(
            example_hotel, indices, method_comparison["TF-IDF"], output_dir
        )

        # 打印推荐结果
        logger.info(f'为酒店 "{example_hotel}" 的推荐结果:')
        for i, hotel in enumerate(recommendations, 1):
            # 确保酒店名称中没有特殊字符
            clean_hotel = hotel.replace("\xa0", " ").strip()
            logger.info(f"{i}. {clean_hotel}")

        end_time = time.time()
        logger.info(f"程序执行完成，总耗时: {end_time - start_time:.2f}秒")

    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        logger.info("程序异常退出")


if __name__ == "__main__":
    main()
