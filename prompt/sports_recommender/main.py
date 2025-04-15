"""
思维树(Tree of Thoughts)方法实现的运动员素质分析与项目推荐系统
"""

import json
from typing import Dict, List, Set
from openai import OpenAI
from dotenv import load_dotenv

# 初始化环境和客户端
load_dotenv()
client = OpenAI()


def get_completion(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0,
    response_format: str = "text",
) -> str:
    """调用OpenAI API获取模型回复

    Args:
        prompt: 输入提示词
        model: 使用的模型名称
        temperature: 输出随机性，0-1之间，值越小随机性越低
        response_format: 返回格式，可选"text"或"json_object"

    Returns:
        模型生成的回复内容
    """
    print("\n" + "-" * 50)
    print(f"【调用API】模型：{model}，温度：{temperature}，格式：{response_format}")
    print(f"【提示词】\n{prompt}")

    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format={"type": response_format},
    )

    content = response.choices[0].message.content
    print(f"【返回结果】\n{content}")
    print("-" * 50 + "\n")

    return content


def analyze_physical_attributes(text: str) -> Dict[str, int]:
    """分析候选人的身体素质并评分

    Args:
        text: 包含候选人姓名和运动成绩的文本

    Returns:
        包含各项素质评分的字典，如{'速度': 3, '耐力': 2, '力量': 3}
    """
    prompt = (
        f"{text}\n请根据以上成绩，分析候选人在速度、耐力、力量三方面素质的分档。"
        f"分档包括：强（3），中（2），弱（1）三档。\n"
        f"以JSON格式输出，其中key为素质名，value为以数值表示的分档。"
    )
    response = get_completion(prompt, response_format="json_object")
    return json.loads(response)


def get_sports_by_attribute(attribute: str, category: str) -> List[str]:
    """获取需要特定素质的特定类别运动列表

    Args:
        attribute: 素质名称，如"速度"、"耐力"、"力量"
        category: 运动类别，如"搏击"

    Returns:
        符合条件的运动列表
    """
    prompt = f"需要{attribute}强的{category}运动有哪些。给出3个例子，以JSON数组形式输出，不要包含任何键名。"
    response = get_completion(prompt, temperature=0.8, response_format="json_object")
    result = json.loads(response)

    # 处理可能的字典情况，提取列表
    if isinstance(result, dict):
        # 尝试从字典中获取第一个值，通常是列表
        for value in result.values():
            if isinstance(value, list):
                return value
        # 如果没有找到列表，则将字典的键作为列表返回
        return list(result.keys())
    elif isinstance(result, list):
        # 如果已经是列表，直接返回
        return result
    else:
        # 如果是其他类型，转换为字符串并返回单元素列表
        return [str(result)]


def check_attribute_requirement(sport: str, attribute: str, value: int) -> bool:
    """评估候选人在特定素质上是否满足特定运动的要求

    Args:
        sport: 运动名称
        attribute: 素质名称
        value: 候选人该素质的评分

    Returns:
        True表示满足要求，False表示不满足
    """
    prompt = f"分析{sport}运动对{attribute}方面素质的要求: 强（3），中（2），弱（1）。\n直接输出挡位数字。"
    response = get_completion(prompt)
    requirement = int(response)
    print(
        f"  {sport}对{attribute}的要求为{requirement}级，候选人{attribute}为{value}级"
    )
    return value >= requirement


def generate_report(
    name: str, performance: str, attributes: Dict[str, int], sport: str
) -> str:
    """生成候选人适合特定运动的分析报告

    Args:
        name: 候选人姓名
        performance: 候选人的运动成绩
        attributes: 候选人各项素质的评分
        sport: 推荐的运动项目

    Returns:
        分析报告文本
    """
    level = ["弱", "中", "强"]
    attributes_desc = {k: level[v - 1] for k, v in attributes.items()}
    prompt = f"已知{name}{performance}\n身体素质：{attributes_desc}。\n生成一篇{name}适合{sport}训练的分析报告。"
    return get_completion(prompt, model="gpt-4o-mini")


def analyze_athlete(name: str, performance: str, category: str) -> None:
    """使用思维树方法分析运动员适合的运动项目

    Args:
        name: 运动员姓名
        performance: 运动成绩描述
        category: 运动类别
    """
    print(f"姓名：{name}")
    print(f"成绩：{performance}")
    print(f"运动类别：{category}")

    # 第一层：分析身体素质
    attributes = analyze_physical_attributes(name + performance)
    print("=" * 30)
    print("身体素质评估")
    print("=" * 30)
    print(json.dumps(attributes, ensure_ascii=False, indent=2))

    # 使用集合去重
    evaluated_sports: Set[str] = set()
    found_strong_attribute = False
    suitable_sports = []

    # 第二层：基于强项素质筛选运动
    for attribute, score in attributes.items():
        if score < 3:  # 剪枝：只考虑强项
            print(f"【剪枝】{attribute}素质评分为{score}，不满足强项要求，跳过")
            continue

        found_strong_attribute = True
        # 获取需要该强项的运动列表
        sports_list = get_sports_by_attribute(attribute, category)
        print("=" * 30)
        print(f"需要{attribute}强的{category}运动")
        print("=" * 30)

        # 第三层：评估每项运动的适合度
        for sport in sports_list:
            if sport in evaluated_sports:
                continue

            evaluated_sports.add(sport)
            print(f"\n【评估】{sport}是否适合{name}")
            is_suitable = True

            # 检查其他素质是否满足要求
            for other_attribute, other_score in attributes.items():
                if other_attribute == attribute:
                    continue

                # 评估其他素质是否满足要求
                if not check_attribute_requirement(sport, other_attribute, other_score):
                    is_suitable = False
                    break

            # 生成适合的运动项目报告
            if is_suitable:
                suitable_sports.append(sport)
                print(f"  【结论】{sport}适合{name}，生成推荐报告")
                report = generate_report(name, performance, attributes, sport)
                print("=" * 50)
                print(f"推荐运动项目报告：{sport}")
                print("=" * 50)
                print(report)
                print("=" * 50)

    # 总结分析结果
    if found_strong_attribute and not suitable_sports:
        print("=" * 50)
        print(f"【总结】{name}虽有强项素质，但没有找到完全适合的{category}运动项目")
        print(f"【建议】可以选择性地提高其他素质，以适应特定的{category}运动")
        print("=" * 50)
    elif not found_strong_attribute:
        print("=" * 50)
        print(
            f"【总结】由于{name}没有评分为3（强）的素质，根据剪枝策略，不会进一步搜索适合的运动项目"
        )
        print(
            f"【建议】{name}需要先提高基础身体素质，才能更好地适应{category}类运动训练"
        )
        print("=" * 50)


def test_excellent_athlete():
    """测试用例1：优秀运动员数据"""
    name = "张三"
    performance = "100米跑成绩：9.8秒，1500米跑成绩：3分01秒，铅球成绩：18米。"
    category = "搏击"

    analyze_athlete(name, performance, category)


def test_average_person():
    """测试用例2：普通人数据"""
    name = "李四"
    performance = "100米跑成绩：13.5秒，1500米跑成绩：5分40秒，铅球成绩：7米。"
    category = "搏击"

    analyze_athlete(name, performance, category)


def main():
    """主函数，使用思维树方法分析候选人适合的运动项目"""
    # 使用两个不同的测试数据
    print("\n" + "=" * 40)
    print("测试数据1：优秀运动员")
    print("=" * 40 + "\n")
    test_excellent_athlete()

    print("\n" + "=" * 40)
    print("测试数据2：普通人数据")
    print("=" * 40 + "\n")
    test_average_person()


if __name__ == "__main__":
    main()
