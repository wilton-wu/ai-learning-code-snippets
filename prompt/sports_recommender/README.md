# 运动员素质分析与项目推荐系统

基于思维树(Tree of Thoughts)方法实现的运动员身体素质分析与运动项目推荐系统。通过分析运动员的各项身体素质指标，结合思维树搜索算法，为运动员推荐最适合的运动项目。

## 项目背景

在体育训练领域，根据运动员的身体素质特点选择适合的运动项目对于提高训练效果和竞技水平至关重要。本系统通过分析运动员的速度、耐力、力量等关键指标，使用思维树方法进行多层次推理，找出最适合的运动项目。

## 思维树算法说明

思维树(Tree of Thoughts)是一种结合大语言模型能力的推理方法，通过多步骤、多层次的思考过程来解决复杂问题。在本项目中，思维树算法的实现包括：

1. **第一层**：分析运动员身体素质，评估速度、耐力、力量等关键指标
2. **第二层**：基于强项素质筛选可能适合的运动项目（剪枝策略：只考虑强项）
3. **第三层**：评估每个候选运动项目对其他素质的要求，确定最终适合的项目

思维树方法的优势在于能够模拟人类专家的思考过程，通过逐层推理和剪枝策略提高推荐的准确性和效率。

## 功能特点

- 多维度身体素质评估（速度、耐力、力量）
- 基于强项的运动项目初筛
- 详细的素质要求匹配分析
- 个性化的运动项目推荐报告
- 完整的分析过程展示

## 使用方法

1. 运行主程序：

```bash
python main.py
```

2. 程序会自动分析示例数据并输出结果。如需分析其他运动员数据，可修改`main.py`中的测试用例：

```python
def test_excellent_athlete():
    name = "运动员姓名"
    performance = "100米跑成绩：XX秒，1500米跑成绩：XX分XX秒，铅球成绩：XX米。"
    category = "运动类别"

    analyze_athlete(name, performance, category)
```

## 输出说明

程序输出包括：

1. 身体素质评估结果
2. 基于强项筛选的运动项目列表
3. 每个运动项目对各项素质的要求评估
4. 最终推荐的运动项目及详细分析报告

## 技术栈

- Python 3.8+
- OpenAI API (GPT-4o-mini)
- 思维树(Tree of Thoughts)算法

## 注意事项

- 需要有效的OpenAI API密钥
- API调用会产生费用，请注意控制使用量
- 分析结果仅供参考，实际训练方案应结合专业教练建议
