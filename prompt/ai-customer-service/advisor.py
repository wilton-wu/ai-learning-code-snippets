import json
import copy
from utils import (
    client,
    INSTRUCTION as instruction,
    OUTPUT_FORMAT as output_format,
    SINGLE_EXAMPLES as examples,
)


class NLU:
    """自然语言理解模块：将用户输入转为结构化查询条件"""

    def __init__(self):
        self.prompt_template = (
            f"{instruction}\n\n{output_format}\n\n{examples}\n\n用户输入：\n__INPUT__"
        )

    def _get_completion(self, prompt, model="gpt-4o-mini"):
        """调用大模型API获取结构化语义"""
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        semantics = json.loads(response.choices[0].message.content)
        return {k: v for k, v in semantics.items() if v}

    def parse(self, user_input):
        """解析用户输入为结构化语义"""
        prompt = self.prompt_template.replace("__INPUT__", user_input)
        return self._get_completion(prompt)


class DST:
    """对话状态跟踪：维护多轮对话状态"""

    def update(self, state, semantics):
        """更新对话状态

        Args:
            state: 当前对话状态
            semantics: NLU解析出的语义

        Returns:
            更新后的对话状态
        """
        # 名称指定时清空状态
        if "name" in semantics:
            state.clear()

        # 处理排序与等值冲突
        if "sort" in semantics:
            slot = semantics["sort"]["value"]
            if slot in state and state[slot].get("operator") == "==":
                del state[slot]

        # 合并新语义
        state.update(semantics)
        return state


class MockedDB:
    """套餐信息模拟数据库"""

    def __init__(self):
        self.data = [
            {"name": "经济套餐", "price": 50, "data": 10, "requirement": None},
            {"name": "畅游套餐", "price": 180, "data": 100, "requirement": None},
            {"name": "无限套餐", "price": 300, "data": 1000, "requirement": None},
            {"name": "校园套餐", "price": 150, "data": 200, "requirement": "在校生"},
        ]

    def retrieve(self, **kwargs):
        """按条件检索套餐

        Args:
            **kwargs: 检索条件，如name、price、data、sort等

        Returns:
            符合条件的套餐列表
        """
        records = []

        # 筛选记录
        for record in self.data:
            # 检查特殊要求
            if record["requirement"] and (
                "status" not in kwargs or kwargs["status"] != record["requirement"]
            ):
                continue

            if not self._match_record(record, kwargs):
                continue

            records.append(record)

        # 排序处理
        if len(records) <= 1:
            return records

        return self._sort_records(records, kwargs.get("sort"))

    def _match_record(self, record, conditions):
        """检查记录是否匹配条件"""
        for key, value in conditions.items():
            if key == "sort":
                continue

            # 处理"无上限"特殊情况
            if (
                key == "data"
                and isinstance(value, dict)
                and value.get("value") == "无上限"
            ):
                if record[key] != 1000:
                    return False
                continue

            # 处理比较操作符
            if isinstance(value, dict) and "operator" in value:
                if not eval(f"{record[key]}{value['operator']}{value['value']}"):
                    return False
            # 精确匹配
            elif str(record[key]) != str(value):
                return False

        return True

    def _sort_records(self, records, sort_config):
        """按指定条件排序记录"""
        if not sort_config:
            # 默认按价格升序
            return sorted(records, key=lambda x: x["price"])

        key = sort_config["value"]
        reverse = sort_config["ordering"] == "descend"
        return sorted(records, key=lambda x: x[key], reverse=reverse)


class DialogManager:
    """对话管理器：协调NLU、DST、DB和回复生成"""

    SYSTEM_PROMPT = "你是一个手机流量套餐的客服代表，你叫小瓜。可以帮助用户选择最合适的流量套餐产品。"

    def __init__(self, prompt_templates):
        # 初始化状态和会话
        self.state = {}
        self.session = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        # 初始化组件
        self.nlu = NLU()
        self.dst = DST()
        self.db = MockedDB()
        self.prompt_templates = prompt_templates

    def _create_prompt(self, user_input, records):
        """生成回复提示词

        Args:
            user_input: 用户输入
            records: 检索结果
        """
        if not records:
            return self._create_not_found_prompt(user_input)
        return self._create_recommend_prompt(user_input, records[0])

    def _create_recommend_prompt(self, user_input, record):
        """生成推荐套餐的提示词"""
        prompt = self.prompt_templates["recommand"].replace("__INPUT__", user_input)
        for key, value in record.items():
            prompt = prompt.replace(f"__{key.upper()}__", str(value))
        return prompt

    def _create_not_found_prompt(self, user_input):
        """生成未找到套餐的提示词"""
        prompt = self.prompt_templates["not_found"].replace("__INPUT__", user_input)
        for key, value in self.state.items():
            if isinstance(value, dict) and "operator" in value:
                prompt = prompt.replace(
                    f"__{key.upper()}__", f"{value['operator']}{value['value']}"
                )
            else:
                prompt = prompt.replace(f"__{key.upper()}__", str(value))
        return prompt

    def _generate_response(self, prompt, model="gpt-4o-mini"):
        """生成自然语言回复"""
        session = copy.deepcopy(self.session)
        session.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=model, messages=session, temperature=0
        )
        return response.choices[0].message.content

    def run(self, user_input):
        """处理用户输入并生成回复

        Args:
            user_input: 用户输入文本

        Returns:
            系统回复
        """
        # 1. 理解用户意图
        semantics = self.nlu.parse(user_input)
        print("===语义解析===")
        print(semantics)

        # 2. 更新对话状态
        self.state = self.dst.update(self.state, semantics)
        print("===对话状态===")
        print(self.state)

        # 3. 检索套餐
        records = self.db.retrieve(**self.state)

        # 4. 生成回复
        prompt = self._create_prompt(user_input, records)
        print("===生成提示词===")
        print(prompt)

        response = self._generate_response(prompt)

        # 5. 更新对话历史
        self.session.append({"role": "user", "content": user_input})
        self.session.append({"role": "assistant", "content": response})

        return response


# 配置和主程序
def main():
    """主函数"""
    # 基础模板
    templates = {
        "recommand": "用户说：__INPUT__ \n\n向用户介绍如下产品：__NAME__，月费__PRICE__元，每月流量__DATA__G。",
        "not_found": "用户说：__INPUT__ \n\n没有找到满足__PRICE__元价位__DATA__G流量的产品，询问用户是否有其他选择倾向。",
    }

    # 添加通用回复指导
    guidance = "\n\n遇到类似问题，请参照以下回答：\n问：流量包太贵了\n答：亲，我们都是全省统一价哦。"
    templates = {k: v + guidance for k, v in templates.items()}

    # 创建对话管理器
    dm = DialogManager(templates)

    # 测试多轮对话
    print("===第一轮对话===")
    response = dm.run("300太贵了，200元以内有吗")
    print("===系统回复===")
    print(response)

    print("\n===第二轮对话===")
    response = dm.run("流量大的")
    print("===系统回复===")
    print(response)


if __name__ == "__main__":
    main()
