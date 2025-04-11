import logging
import re
import time
from typing import List, Tuple

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# 创建logger实例
logger = logging.getLogger(__name__)

# 确保nltk数据已下载
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

# 文本预处理配置
REPLACE_BY_SPACE_RE = re.compile(r"[/(){}\[\]\|@,;]")
BAD_SYMBOLS_RE = re.compile(r"[^0-9a-z #+_]")
STOPWORDS = set(stopwords.words("english"))


def load_data(file_path: str) -> pd.DataFrame:
    """
    加载并初始化酒店数据集
    Args:
        file_path: 数据文件完整路径
    Returns:
        pd.DataFrame: 包含酒店数据的DataFrame
    Raises:
        FileNotFoundError: 当文件不存在时
        pd.errors.EmptyDataError: 当文件为空时
        pd.errors.ParserError: 当文件解析错误时
    """
    try:
        start_time = time.time()
        df = pd.read_csv(file_path, encoding="latin-1")

        total_rows = len(df)
        logger.info(f"开始处理 {total_rows} 条酒店描述文本...")

        # 添加处理进度跟踪
        logger.info("文本清洗开始 | 开始时间: %s", time.strftime("%Y-%m-%d %H:%M:%S"))
        df["desc_clean"] = df["desc"].apply(clean_text)
        logger.info("文本清洗完成 | 总耗时: %.2f秒", time.time() - start_time)

        # 记录处理结果样例
        sample_data = df["desc_clean"].sample(3).tolist()
        logger.info(
            "清洗后文本样例:\n%s",
            "\n".join(f"{i + 1}. {text[:50]}..." for i, text in enumerate(sample_data)),
        )

        return df
    except pd.errors.EmptyDataError:
        logger.error(f"数据文件 {file_path} 为空")
        raise
    except pd.errors.ParserError:
        logger.error(f"数据文件 {file_path} 解析错误")
        raise


def clean_text(text: str) -> str:
    """
    清洗文本数据
    Args:
        text: 原始文本
    Returns:
        清洗后的文本
    """
    if not isinstance(text, str) or not text:
        return ""

    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(" ", text)
    text = BAD_SYMBOLS_RE.sub("", text)
    # 使用列表推导式而不是生成器表达式，性能更好
    words = [word for word in text.split() if word not in STOPWORDS]
    return " ".join(words)


def get_top_ngrams(
    corpus: pd.Series, ngram_range: Tuple[int, int] = (3, 3), top_k: int = 20
) -> List[Tuple[str, int]]:
    """
    获取指定n-gram的TopK高频词
    Args:
        corpus: 文本语料
        ngram_range: n-gram范围
        top_k: 返回的高频词数量
    Returns:
        包含(词汇, 频次)的元组列表
    """
    if corpus.empty:
        logger.warning("输入的文本语料为空")
        return []

    # 添加进度日志
    logger.info(f"开始提取 {ngram_range} 范围的n-gram...")

    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words="english")
    try:
        bag_of_words = vectorizer.fit_transform(corpus)
        sum_words = bag_of_words.sum(axis=0)

        # word_freq = [
        #     (word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()
        # ]
        # top_words = sorted(word_freq, key=lambda x: x[1], reverse=True)[:top_k]
        # 使用numpy的argsort获取top-k，性能更好
        word_idx = np.argsort(sum_words.A1)[-top_k:][::-1]  # 使用.A1获取展平后的密集数组
        top_words = [(list(vectorizer.vocabulary_.keys())[list(vectorizer.vocabulary_.values()).index(i)], sum_words[0, i]) for i in word_idx]

        logger.info(f"成功提取 {len(top_words)} 个高频n-gram")
        return top_words
    except Exception as e:
        logger.error(f"提取n-gram时出错: {str(e)}")
        return []
