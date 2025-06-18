import operator
from typing import Annotated, List, TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from IPython.display import display, Image, Markdown


# 报告章节结构定义（用于结构化输出）
class Section(BaseModel):
    name: str = Field(
        description="报告章节的名称",
    )
    description: str = Field(
        description="该章节涵盖的主要主题和概念的简要概述",
    )


# 报告章节列表容器
class Sections(BaseModel):
    sections: List[Section] = Field(
        description="报告的章节列表",
    )


# 加载环境变量并初始化LLM模型
load_dotenv()
llm = init_chat_model("openai:gpt-4o")
# 为LLM添加结构化输出能力（绑定Sections模型）
planner = llm.with_structured_output(Sections)


# 全局流程状态定义
class State(TypedDict):
    topic: str  # 报告主题（用户输入）
    sections: list[Section]  # 规划阶段生成的章节列表
    completed_sections: Annotated[
        list, operator.add
    ]  # 并行任务结果存储区（自动合并多节点输出）
    final_report: str  # 最终生成的完整报告


# 工作节点专用状态（仅包含单任务所需数据）
class WorkerState(TypedDict):
    section: Section  # 当前处理的章节信息
    completed_sections: Annotated[list, operator.add]  # 单章节生成结果


# 核心节点定义
# 1. 编排节点：负责生成报告大纲
def orchestrator(state: State):
    """根据报告主题生成详细章节规划
    调用LLM生成结构化的章节列表，包含每个章节的名称和内容概述
    """

    # 调用规划LLM生成章节结构
    report_sections = planner.invoke(
        [
            SystemMessage(content="生成报告的章节规划，确保逻辑连贯且全面覆盖主题"),
            HumanMessage(content=f"报告主题：{state['topic']}"),
        ]
    )

    return {"sections": report_sections.sections}


# 2. 工作节点：负责生成单个章节内容
def llm_call(state: WorkerState):
    """根据章节名称和概述生成具体内容
    接收编排节点分配的章节任务，调用LLM生成带Markdown格式的章节文本
    """

    # 生成章节具体内容
    section = llm.invoke(
        [
            SystemMessage(
                content="根据提供的章节名称和概述撰写报告内容。无需添加前言，直接使用Markdown格式"
            ),
            HumanMessage(
                content=f"章节名称：{state['section'].name}\n概述：{state['section'].description}"
            ),
        ]
    )

    # 将生成的章节内容写入结果列表
    return {"completed_sections": [section.content]}


# 3. 汇总节点：合并章节生成最终报告
def synthesizer(state: State):
    """将所有章节内容合并为完整报告
    接收所有并行工作节点的输出，按顺序拼接成最终报告
    """

    # 获取所有已完成的章节内容
    completed_sections = state["completed_sections"]

    # 用分隔线连接各章节内容
    completed_report_sections = "\n\n---\n\n".join(completed_sections)

    return {"final_report": completed_report_sections}


# 条件边函数：实现并行任务分配
def assign_workers(state: State):
    """为每个章节分配独立的工作节点
    根据规划阶段生成的章节列表，创建并行的llm_call任务
    """

    # 通过Send() API创建并行任务：为每个章节启动一个llm_call节点
    return [Send("llm_call", {"section": s}) for s in state["sections"]]


# 工作流构建
# 1. 创建状态图（指定全局状态类型）
orchestrator_worker_builder = StateGraph(State)

# 2. 注册节点到状态图
orchestrator_worker_builder.add_node("orchestrator", orchestrator)  # 规划节点
orchestrator_worker_builder.add_node("llm_call", llm_call)          # 工作节点
orchestrator_worker_builder.add_node("synthesizer", synthesizer)    # 汇总节点

# 3. 定义节点间的连接关系
orchestrator_worker_builder.add_edge(START, "orchestrator")  # 起始点→规划节点
# 规划节点→工作节点（条件边：动态创建并行任务）
orchestrator_worker_builder.add_conditional_edges(
    "orchestrator", assign_workers, ["llm_call"]
)
orchestrator_worker_builder.add_edge("llm_call", "synthesizer")  # 工作节点→汇总节点
orchestrator_worker_builder.add_edge("synthesizer", END)          # 汇总节点→结束点

# 4. 编译状态图为可执行工作流
orchestrator_worker = orchestrator_worker_builder.compile()

# 可视化工作流图
display(Image(orchestrator_worker.get_graph().draw_mermaid_png()))

# 执行工作流：生成"LLM缩放定律"报告
state = orchestrator_worker.invoke({"topic": "创建关于LLM缩放定律的报告"})

# 显示最终报告（Markdown格式）
Markdown(state["final_report"])
