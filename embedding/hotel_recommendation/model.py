import logging
import time
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import Normalizer

# 创建logger实例
logger = logging.getLogger(__name__)

# 向量化器参数配置
VECTORIZER_PARAMS = {
    "tfidf": {
        "ngram_range": (1, 3),
        "min_df": 0.01,
        "stop_words": "english",
        "max_features": 5000,  # 限制特征数量，提高性能
    },
    "count": {
        "ngram_range": (1, 3),
        "stop_words": "english",
        "binary": False,
        "max_features": 5000,  # 限制特征数量，提高性能
    },
}


def train_model(
    df: pd.DataFrame, text_column: str = "desc_clean", method: str = "tfidf"
) -> Tuple[Any, pd.Series, np.ndarray]:
    """
    训练TF-IDF或词频计数模型并计算相似度矩阵
    Args:
        df: 包含清洗后文本的数据框
        text_column: 文本列名称
        method: 向量化方法，可选 'tfidf' 或 'count'
    Returns:
        (向量化器, 索引序列, 余弦相似度矩阵)
    Raises:
        ValueError: 当method参数不是'tfidf'或'count'时
    """
    if method not in VECTORIZER_PARAMS:
        raise ValueError(f"不支持的向量化方法: {method}，请使用 'tfidf' 或 'count'")

    if text_column not in df.columns:
        raise ValueError(f"数据框中不存在列: {text_column}")

    # 记录处理时间，用于性能分析
    start_time = time.time()
    logger.info(f"开始训练 {method} 向量化模型...")

    # 根据指定的方法选择合适的向量化器
    vectorizer_class = TfidfVectorizer if method == "tfidf" else CountVectorizer
    vectorizer = vectorizer_class(**VECTORIZER_PARAMS[method])

    # 对数据框中的文本列进行向量化处理
    try:
        feature_matrix = vectorizer.fit_transform(df[text_column])
        logger.info(f"特征矩阵大小: {feature_matrix.shape}")

        # 应用L2归一化处理CountVectorizer的输出
        if method == "count":
            feature_matrix = Normalizer(norm="l2").fit_transform(feature_matrix)

        # 计算特征矩阵的余弦相似度矩阵
        # 使用矩阵乘法计算余弦相似度，对于归一化的向量，这等同于余弦相似度
        cosine_sim = (feature_matrix * feature_matrix.T).toarray()

        # 确保对角线上的值为1（自身与自身的相似度）
        np.fill_diagonal(cosine_sim, 1.0)

        # 创建索引序列，用于后续推荐
        indices = pd.Series(df["name"].values, index=df.index)

        end_time = time.time()
        logger.info(f"{method} 模型训练完成，耗时: {end_time - start_time:.2f}秒")

        return vectorizer, indices, cosine_sim
    except Exception as e:
        logger.error(f"{method} 模型训练失败: {str(e)}")
        raise


def get_recommendations(
    name: str, indices: pd.Series, cosine_similarities: np.ndarray, top_n: int = 10
) -> List[str]:
    """
    生成酒店推荐列表
    Args:
        name: 目标酒店名称
        indices: 酒店名称索引序列
        cosine_similarities: 余弦相似度矩阵
        top_n: 推荐数量
    Returns:
        推荐酒店名称列表
    Raises:
        ValueError: 当找不到指定酒店时
    """
    if not isinstance(name, str) or not name:
        raise ValueError("酒店名称必须是非空字符串")

    if top_n < 1:
        raise ValueError("推荐数量必须大于0")

    try:
        # 查找目标酒店的索引
        idx_matches = indices[indices == name].index
        if len(idx_matches) == 0:
            raise ValueError(f"未找到酒店: {name}")

        idx = idx_matches[0]

        # 获取相似度分数
        sim_scores = list(enumerate(cosine_similarities[idx]))

        # 按相似度降序排序
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # 排除自身，获取top_n个推荐
        sim_scores = sim_scores[1 : top_n + 1]

        # 提取推荐酒店名称
        top_indices = [i for i, _ in sim_scores]
        recommendations = [indices[i] for i in top_indices]

        return recommendations
    except Exception as e:
        logger.error(f"生成推荐时出错: {str(e)}")
        raise
