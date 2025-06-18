from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from IPython.display import display, Image

load_dotenv()
llm = init_chat_model("openai:gpt-4o")

# 工作流状态定义
class State(TypedDict):
    joke: str  # 生成的笑话内容
    topic: str  # 笑话主题（用户输入）
    feedback: str  # 评估反馈意见
    funny_or_not: str  # 评估结果（funny/not funny）


# 评估结果结构化输出定义
class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(
        description="判断笑话是否有趣，只能选择'funny'或'not funny'",
    )
    feedback: str = Field(
        description="如果笑话无趣，提供具体的改进建议",
    )


# 为LLM添加结构化输出能力（绑定Feedback模型）
evaluator = llm.with_structured_output(Feedback)


# 核心节点定义
# 1. 笑话生成节点
def llm_call_generator(state: State):
    """根据主题和反馈生成笑话
    如果存在历史反馈，则基于反馈优化笑话；否则直接生成新笑话
    返回包含生成笑话的状态字典
    """

    if state.get("feedback"):
        msg = llm.invoke(
            f"围绕主题'{state['topic']}'创作笑话，同时参考以下反馈进行改进：{state['feedback']}"
        )
    else:
        msg = llm.invoke(f"围绕主题'{state['topic']}'创作一个有趣的笑话")
    return {"joke": msg.content}


# 2. 笑话评估节点
def llm_call_evaluator(state: State):
    """评估笑话的趣味性
    调用结构化LLM对笑话进行评分，并生成改进建议
    返回包含评估结果和反馈的状态字典
    """

    grade = evaluator.invoke(f"请评估以下笑话是否有趣：{state['joke']}")
    return {"funny_or_not": grade.grade, "feedback": grade.feedback}


# 条件路由函数：根据评估结果决定流程走向
def route_joke(state: State):
    """根据评估结果路由工作流
    - 如果笑话有趣（funny），则流程结束
    - 如果笑话无趣（not funny），则返回生成节点重新创作
    """

    if state["funny_or_not"] == "funny":
        return "Accepted"
    elif state["funny_or_not"] == "not funny":
        return "Rejected + Feedback"


# 工作流构建
# 1. 创建状态图（指定全局状态类型）
optimizer_builder = StateGraph(State)

# 2. 注册节点到状态图
optimizer_builder.add_node("llm_call_generator", llm_call_generator)  # 笑话生成节点
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)  # 笑话评估节点

# 3. 定义节点间的连接关系
optimizer_builder.add_edge(START, "llm_call_generator")  # 起始点→生成节点
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")  # 生成节点→评估节点
# 评估节点→条件路由（根据评估结果决定下一步）
optimizer_builder.add_conditional_edges(
    "llm_call_evaluator",
    route_joke,
    {  # 路由返回值: 目标节点
        "Accepted": END,  # 接受→结束
        "Rejected + Feedback": "llm_call_generator"  # 拒绝→重新生成
    },
)

# 4. 编译状态图为可执行工作流
optimizer_workflow = optimizer_builder.compile()

# 可视化工作流图
display(Image(optimizer_workflow.get_graph().draw_mermaid_png()))

# 执行工作流：生成关于"猫"的笑话
state = optimizer_workflow.invoke({"topic": "Cats"})
print("最终生成的笑话：", state["joke"])
