"""
Word2Vec词向量相似度计算主程序
"""
import logging
from pathlib import Path

# 导入自定义模块
from text_processor import ensure_dir_exists, segment
from model_utils import train_word2vec, calculate_similarity

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 获取项目根目录
BASE_DIR = Path(__file__).parent

# 项目配置
CONFIG = {
    "data_path": BASE_DIR / "journey_to_the_west.txt",
    "output_dir": BASE_DIR / "dist",
    "model_name": "word2vec.model",
    "vector_size": 100,
    "window": 5,
    "min_count": 1,
    "workers": 4,
}


def main() -> None:
    """主函数"""
    # 程序初始化时统一创建输出目录
    ensure_dir_exists(CONFIG["output_dir"])
    
    try:
        # 分词处理，如果分词文件已存在则不重新分词
        segmented_file = segment(
            file_path=CONFIG["data_path"], 
            output_dir=CONFIG["output_dir"], 
            force=False
        )

        # 训练模型，如果模型文件已存在则直接加载
        model = train_word2vec(
            corpus_path=segmented_file,
            output_dir=CONFIG["output_dir"],
            model_name=CONFIG["model_name"],
            vector_size=CONFIG["vector_size"],
            window=CONFIG["window"],
            min_count=CONFIG["min_count"],
            workers=CONFIG["workers"],
            force=False
        )

        # 1. 计算主要角色之间的相似度
        logger.info("===== 角色相似度分析 =====")
        calculate_similarity(
            model,
            word_pairs=[
                ("孙悟空", "猪八戒"),  # 师兄弟关系
                ("孙悟空", "唐僧"),    # 师徒关系
                ("唐僧", "如来"),      # 佛教关系
                ("孙悟空", "妖怪"),    # 对立关系
                ("唐僧", "取经"),      # 人物与使命
            ],
        )
        
        # 2. 查找与主要角色最相似的词
        logger.info("===== 角色相似词分析 =====")
        calculate_similarity(
            model,
            positive=["孙悟空", "唐僧", "猪八戒", "沙僧"],
            topn=8
        )
        
        # 3. 词语类比关系测试
        logger.info("===== 词语类比关系分析 =====")
        # 测试关系模式：如果A对于B相当于C对于D，那么A-B+C≈D
        calculate_similarity(
            model, 
            positive=["孙悟空", "师父"], 
            negative=["徒弟"],
            topn=5
        )
        
        # 测试另一种关系
        calculate_similarity(
            model, 
            positive=["唐僧", "西天"], 
            negative=["东土"],
            topn=5
        )

        logger.info("程序执行完成")
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")


if __name__ == "__main__":
    main()
