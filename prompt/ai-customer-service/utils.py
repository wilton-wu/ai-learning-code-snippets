"""
公共工具模块，包含共享的常量、函数和配置
"""

from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 创建OpenAI客户端，默认使用环境变量中的 OPENAI_API_KEY 和 OPENAI_BASE_URL
client = OpenAI()


def get_completion(prompt, response_format="text", model="gpt-4o-mini", temperature=0):
    """调用OpenAI API获取模型回复

    Args:
        prompt: 提示词
        response_format: 返回格式，text或json_object
        model: 使用的模型
        temperature: 温度参数，控制输出的随机性

    Returns:
        模型回复内容
    """
    try:
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format=dict(type=response_format),
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API调用出错: {e}")
        return None


# 任务描述
INSTRUCTION = """
你的任务是识别用户对手机流量套餐产品的选择条件。
每种流量套餐产品包含三个属性：名称（name），月费价格（price），月流量（data）。
根据用户输入，识别用户在上述三种属性上的需求是什么。
"""

# 输出格式
OUTPUT_FORMAT = """
以JSON格式输出
1. name字段的取值为string类型，取值必须为以下之一：经济套餐、畅游套餐、无限套餐、校园套餐或null；
2. price字段的取值为一个结构体或null，包含两个字段：
  (1) operator，string类型，取值范围：'<=' (小于等于), '>=' (大于等于), '==' (等于)
  (2) value, int类型
3. data字段的取值为一个结构体或null，包含两个字段：
  (1) operator，string类型，取值范围：'<=' (小于等于), '>=' (大于等于), '==' (等于)
  (2) value，int类型或string类型，string类型只能是"无上限"
4. 用户的意图可以包含按price或data排序，以sort字段标识，取值为一个结构体：
  (1) 结构体中以"ordering"="descend"表示按降序排序，以"value"字段存储待排序的字段
  (2) 结构体中以"ordering"="ascend"表示按升序排序，以"value"字段存储待排序的字段

输出中只包含用户提及的字段，不要猜测任何用户未直接提及的字段，不输出值为null的字段。
DO NOT OUTPUT NULL-VALUED FIELD! 确保输出能被json.loads加载。
"""

# 单次对话示例
SINGLE_EXAMPLES = """
便宜的套餐：{"sort": {"ordering": "ascend", "value": "price"}}
有没有不限流量的：{"data": {"operator": "==", "value": "无上限"}}
流量大的：{"sort": {"ordering": "descend", "value": "data"}}
100G以上流量的套餐最便宜的是哪个：{"sort": {"ordering": "ascend", "value": "price"}, "data": {"operator": ">=", "value": 100}}
月费不超过200的：{"price": {"operator": "<=", "value": 200}}
就要月费180那个套餐：{"price": {"operator": "==", "value": 180}}
经济套餐：{"name": "经济套餐"}
土豪套餐：{"name": "无限套餐"}
"""

# 多轮对话示例
MULTI_EXAMPLES = """
客服：有什么可以帮您
用户：100G套餐有什么

{"data": {"operator": ">=", "value": 100}}

客服：有什么可以帮您
用户：100G套餐有什么
客服：我们现在有无限套餐，不限流量，月费300元
用户：太贵了，有200元以内的不

{"data": {"operator": ">=", "value": 100}, "price": {"operator": "<=", "value": 200}}

客服：有什么可以帮您
用户：便宜的套餐有什么
客服：我们现在有经济套餐，每月50元，10G流量
用户：100G以上的有什么

{"data": {"operator": ">=", "value": 100}, "sort": {"ordering": "ascend", "value": "price"}}

客服：有什么可以帮您
用户：100G以上的套餐有什么
客服：我们现在有畅游套餐，流量100G，月费180元
用户：流量最多的呢

{"sort": {"ordering": "descend", "value": "data"}, "data": {"operator": ">=", "value": 100}}
"""
