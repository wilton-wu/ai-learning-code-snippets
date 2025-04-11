"""
模型工具模块，包含Word2Vec模型训练和相似度计算功能
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from gensim.models import word2vec
from gensim.models.word2vec import LineSentence

logger = logging.getLogger(__name__)


def train_word2vec(corpus_path: Path, output_dir: Path, model_name: str, 
                  vector_size: int = 100, window: int = 5, 
                  min_count: int = 1, workers: int = 4, 
                  force: bool = False) -> word2vec.Word2Vec:
    """
    训练word2vec模型

    Args:
        corpus_path: 语料库路径
        output_dir: 输出目录
        model_name: 模型名称
        vector_size: 向量维度
        window: 窗口大小
        min_count: 最小词频
        workers: 工作线程数
        force: 是否强制重新训练，即使模型文件已存在

    Returns:
        训练好的模型
    """
    try:
        model_path = output_dir / model_name

        # 如果模型文件已存在且不强制重新训练，则直接加载
        if model_path.exists() and not force:
            logger.info(f"模型文件已存在，直接加载: {model_path}")
            model = word2vec.Word2Vec.load(str(model_path))
            return model

        logger.info(f"开始训练word2vec模型，使用语料库: {corpus_path}")

        # 使用gensim的LineSentence类处理语料库
        # LineSentence是一个内存友好的迭代器，可以多次遍历语料库
        sentences = LineSentence(corpus_path)

        # 训练模型
        model = word2vec.Word2Vec(
            sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
        )

        # 保存模型
        model.save(str(model_path))
        logger.info(f"模型训练完成，已保存至: {model_path}")

        return model
    except Exception as e:
        logger.error(f"模型训练过程出错: {str(e)}")
        raise


def calculate_similarity(
    model: word2vec.Word2Vec,
    word_pairs: Optional[List[Tuple[str, str]]] = None,
    positive: Optional[List[str]] = None,
    negative: Optional[List[str]] = None,
    topn: int = 10
) -> Dict[str, Union[float, List[Tuple[str, float]]]]:
    """
    计算词向量相似度
    
    支持三种计算模式：
    1. 词对相似度：计算两个词之间的余弦相似度
    2. 相似词查找：查找与给定词最相似的词
    3. 词语类比：计算词语间的类比关系（如：king - man + woman ≈ queen）

    Args:
        model: 训练好的word2vec模型
        word_pairs: 需要计算相似度的词对列表，格式为[(word1, word2), ...]
        positive: 正向词列表，用于相似词查找或类比计算
        negative: 负向词列表，用于类比计算
        topn: 返回的相似词或类比结果数量，默认为10

    Returns:
        相似度计算结果字典，根据计算类型包含不同的键值对
    """
    results = {}

    try:
        # 检查模型是否有效
        if not hasattr(model, 'wv'):
            raise ValueError("提供的模型不包含词向量属性(wv)")
            
        # 计算词对相似度
        if word_pairs:
            _calculate_word_pair_similarity(model, word_pairs, results)
            
        # 查找最相似的词
        if positive:
            if not negative:
                _find_similar_words(model, positive, results, topn)
            else:
                _calculate_word_analogy(model, positive, negative, results, topn)
                
        return results
    except Exception as e:
        logger.error(f"计算相似度过程出错: {str(e)}")
        raise


def _calculate_word_pair_similarity(
    model: word2vec.Word2Vec, 
    word_pairs: List[Tuple[str, str]], 
    results: Dict
) -> None:
    """计算词对之间的相似度"""
    for word1, word2 in word_pairs:
        # 检查词是否在词表中
        if word1 not in model.wv:
            logger.warning(f"词汇 '{word1}' 不在词表中")
            continue
        if word2 not in model.wv:
            logger.warning(f"词汇 '{word2}' 不在词表中")
            continue
            
        # 计算相似度
        similarity = model.wv.similarity(word1, word2)
        key = f"{word1}-{word2}相似度"
        results[key] = similarity
        logger.info(f"{word1}与{word2}的相似度: {similarity:.4f}")


def _find_similar_words(
    model: word2vec.Word2Vec, 
    words: List[str], 
    results: Dict,
    topn: int = 10
) -> None:
    """查找与给定词最相似的词"""
    for word in words:
        if word not in model.wv:
            logger.warning(f"词汇 '{word}' 不在词表中")
            continue
            
        similar_words = model.wv.most_similar(word, topn=topn)
        results[f"{word}的相似词"] = similar_words
        
        # 格式化输出，提高可读性
        similar_words_formatted = [f"{w}({s:.4f})" for w, s in similar_words]
        logger.info(f"{word}的相似词: {', '.join(similar_words_formatted)}")


def _calculate_word_analogy(
    model: word2vec.Word2Vec, 
    positive: List[str], 
    negative: List[str], 
    results: Dict,
    topn: int = 10
) -> None:
    """计算词语类比关系"""
    # 检查所有词是否在词表中
    missing_words = [word for word in positive + negative if word not in model.wv]
    if missing_words:
        logger.warning(f"以下词汇不在词表中: {', '.join(missing_words)}")
        return
        
    # 计算类比结果
    analogy_results = model.wv.most_similar(positive=positive, negative=negative, topn=topn)
    results["词类比结果"] = analogy_results
    
    # 格式化输出，提高可读性
    analogy_formatted = [f"{w}({s:.4f})" for w, s in analogy_results]
    logger.info(
        f"词类比结果 ({'+'.join(positive)}-{'+'.join(negative)}): {', '.join(analogy_formatted)}"
    )
