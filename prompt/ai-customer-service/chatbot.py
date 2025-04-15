"""
简化版OpenAI聊天机器人，使用utils.py中的公共组件
"""

import json
from utils import client


def print_json(data):
    """打印格式化的JSON数据

    Args:
        data: 要打印的数据，可以是字典、列表或其他类型
    """
    if hasattr(data, "model_dump_json"):
        data = json.loads(data.model_dump_json())

    if isinstance(data, (list, dict)):
        print(json.dumps(data, indent=4, ensure_ascii=False))
    else:
        print(data)


class ChatBot:
    """简单的聊天机器人，维护对话上下文"""

    def __init__(self, system_prompt=None):
        """初始化聊天机器人

        Args:
            system_prompt: 系统提示词，用于设置机器人的行为和背景知识
        """
        # 默认系统提示词
        default_prompt = """
你是一个手机流量套餐的客服代表，你叫小瓜。可以帮助用户选择最合适的流量套餐产品。可以选择的套餐包括：
经济套餐，月费50元，10G流量；
畅游套餐，月费180元，100G流量；
无限套餐，月费300元，1000G流量；
校园套餐，月费150元，200G流量，仅限在校生。
"""
        # 初始化消息历史
        self.messages = [
            {
                "role": "system",
                "content": system_prompt if system_prompt else default_prompt,
            }
        ]

    def chat(self, user_input, model="gpt-4o-mini", temperature=0.7):
        """与用户进行对话

        Args:
            user_input: 用户输入的文本
            model: 使用的模型名称
            temperature: 温度参数，控制输出的随机性

        Returns:
            模型的回复
        """
        # 添加用户消息
        self.messages.append({"role": "user", "content": user_input})

        # 调用API获取回复
        response = client.chat.completions.create(
            model=model,
            messages=self.messages,
            temperature=temperature,
        )
        reply = response.choices[0].message.content

        # 添加助手回复到历史
        self.messages.append({"role": "assistant", "content": reply})

        # 打印回复
        print(f"小瓜: {reply}")
        return reply

    def get_chat_history(self):
        """获取完整的对话历史"""
        return self.messages


def main():
    """主函数"""
    print("=== 流量套餐客服小瓜 ===")
    print("输入 'exit' 结束对话\n")

    # 创建聊天机器人实例
    bot = ChatBot()

    # 演示多轮对话
    bot.chat("流量最大的套餐是什么？")
    bot.chat("多少钱？")
    bot.chat("给我办一个")

    # 交互式对话
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() in ["exit", "quit", "退出", "再见"]:
            print("小瓜: 感谢您的咨询，再见！")
            break
        bot.chat(user_input)


if __name__ == "__main__":
    main()
