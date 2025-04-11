"""
文本处理模块，包含分词和停用词处理功能
"""

import logging
import re
from pathlib import Path

import jieba

logger = logging.getLogger(__name__)


def ensure_dir_exists(directory: Path) -> None:
    """统一目录创建函数

    Args:
        directory: 需要确保存在的目录路径
    """
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"创建目录: {directory}")


def get_chinese_stopwords() -> set:
    """
    获取中文停用词集合

    Returns:
        中文停用词集合
    """
    # fmt: off
    # 常用中文停用词
    common_stopwords = {
        # 代词和连接词
        "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个", 
        "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", 
        "自己", "这", "那", "这个", "那个", "这些", "那些", "来", "他", "她", "它", "们",
        # 介词和连词
        "为", "可以", "个", "然后", "没", "什么", "与", "而", "使", "如果", "因为", 
        "所以", "但是", "但", "却", "对", "能", "被",
    }

    # 中文标点符号和特殊字符
    chinese_punctuation = {
        # 基本标点
        " ", "，", "。", "、", "：", "；", "！", "？", "（", "）", 
        # 引号和书名号
        "「", "」", """, """, "'", "'", "《", "》", 
        # 其他标点
        "…", "—", "【", "】", "～", "·", "〈", "〉", "『", "』",
        # 特殊空白字符
        "\n", "\t", "\r", "\u3000",
    }
    # fmt: on

    # 合并所有停用词
    return common_stopwords.union(chinese_punctuation)


def segment(file_path: Path, output_dir: Path, force: bool = False) -> Path:
    """增强版分词处理函数

    Args:
        file_path: 输入文件路径
        output_dir: 输出目录
        force: 是否强制重新分词

    Returns:
        分词后的文件路径

    Raises:
        UnicodeDecodeError: 当无法解码文件内容时
    """
    try:
        # 获取中文停用词
        stopwords = get_chinese_stopwords()

        segment_out_path = output_dir / f"segmented_{file_path.name}"

        # 如果分词文件已存在且不强制重新分词，则直接返回
        if segment_out_path.exists() and not force:
            logger.info(f"分词文件已存在，跳过分词操作: {segment_out_path}")
            return segment_out_path

        logger.info(f"开始对 {file_path} 进行分词")

        # 使用二进制模式读取文件，手动处理编码
        # 自动检测文件编码
        for encoding in ["utf-8", "gb18030", "latin1"]:
            try:
                with open(file_path, "rb") as f:
                    content = f.read().decode(encoding)
                    break
            except UnicodeDecodeError:
                continue
        else:
            raise UnicodeDecodeError("", b"", 0, 1, f"无法解码文件: {file_path}")

        # 按句子分割文本（以。！？等标点为分隔符）
        # 使用正则表达式将文本分割成句子，保留标点符号
        sentences = []
        for paragraph in content.split("\n"):
            # 使用正则表达式分割句子，保留分隔符
            parts = re.split(r"([。！？!?.])", paragraph)

            # 将分割后的部分重新组合成完整句子
            i = 0
            while i < len(parts) - 1:
                if parts[i].strip() or (i + 1 < len(parts) and parts[i + 1].strip()):
                    sentence = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
                    if sentence.strip():
                        sentences.append(sentence.strip())
                i += 2

        # 使用二进制模式写入文件
        with open(segment_out_path, "wb") as out_f:
            # 逐句处理
            for sentence in sentences:
                if not sentence.strip():
                    continue  # 跳过空句子

                # 使用jieba进行分词
                # 中文文本中词与词之间没有空格分隔，因此需要分词技术来识别词语边界
                text_cut = jieba.cut(sentence.strip())

                # 过滤停用词
                sentence_segment = [word for word in text_cut if word not in stopwords]

                if sentence_segment:  # 确保分词后的句子不为空
                    # 将分词结果写入文件，每个句子占一行
                    result = " ".join(sentence_segment) + "\n"
                    out_f.write(result.encode("utf-8"))

        logger.info(f"分词完成，结果保存至: {segment_out_path}")
        return segment_out_path
    except Exception as e:
        logger.error(f"分词过程出错: {str(e)}")
        raise
