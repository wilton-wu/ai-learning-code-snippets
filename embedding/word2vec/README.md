# Word2Vec 工具

- 数据集：西游记
- 功能：计算小说中的人物相似度，比如孙悟空与猪八戒，孙悟空与孙行者

## 方案步骤

1. 使用分析工具进行分词，比如 NLTK、JIEBA
2. 将训练语料转化成一个 sentence 的迭代器
3. 使用 gensim 训练 Word2Vec 模型
4. 使用训练好的模型计算人物相似度
