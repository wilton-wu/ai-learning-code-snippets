from utils import (
    get_completion,
    INSTRUCTION,
    OUTPUT_FORMAT,
    SINGLE_EXAMPLES,
    MULTI_EXAMPLES,
)


def process_single_dialog(input_text):
    """处理单次对话

    Args:
        input_text: 用户输入文本

    Returns:
        模型回复
    """
    prompt = f"""
# 目标
{INSTRUCTION}

# 输出格式
{OUTPUT_FORMAT}

# 举例
{SINGLE_EXAMPLES}

# 用户输入
{input_text}
"""
    print("=" * 4 + "单次对话" + "=" * 4)
    return get_completion(prompt, response_format="json_object")


def process_multi_dialog(input_text):
    """处理多轮对话

    Args:
        input_text: 用户输入文本

    Returns:
        模型回复
    """
    # 多轮对话上下文
    context = f"""
客服：有什么可以帮您
用户：有什么100G以上的套餐推荐
客服：我们有畅游套餐和无限套餐，您有什么价格倾向吗
用户：{input_text}
"""

    prompt = f"""
# 目标
{INSTRUCTION}

# 输出格式
{OUTPUT_FORMAT}

# 举例
{MULTI_EXAMPLES}

# 对话上下文
{context}
"""
    print("=" * 4 + "多轮对话" + "=" * 4)
    return get_completion(prompt, response_format="json_object")


def main():
    """主函数，程序入口"""
    # 测试用例 - 更具特征性的输入
    single_input = "我想要一个200元以内的套餐，但流量必须在100G以上，最好是经济型的"
    multi_input = "我需要最便宜的无限流量套餐，价格不能超过250元"

    # 处理单次对话
    single_response = process_single_dialog(single_input)
    print(f"用户输入: {single_input}")
    print(f"模型输出: {single_response}\n")

    # 处理多轮对话
    multi_response = process_multi_dialog(multi_input)
    print(f"用户输入: {multi_input}")
    print(f"模型输出: {multi_response}")


if __name__ == "__main__":
    main()
