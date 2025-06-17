from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.schema import AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from document_retrieval import retrieve_documents

# LangGraph 多智能体工作流示例
# 核心功能：实现带检索增强(RAG)的对话系统，支持自动判断信息充分性并请求人工干预
# 1. 状态管理：使用Annotated[list, add_messages]自动处理消息追加
# 2. 流程控制：通过条件边实现检索后决策分支
# 3. 中断机制：支持工作流暂停并请求人工输入

load_dotenv()
llm = init_chat_model("gpt-4o", model_provider="openai")

memory = MemorySaver()
thread_config = {
    "configurable": {
        "thread_id": "thread_id_001"
    }
}


class State(TypedDict):
    """工作流共享状态

    Attributes:
        messages: 对话消息列表，使用add_messages注解自动追加新消息
    """
    messages: Annotated[list, add_messages]


rag_template = """请根据对话历史和下面提供的信息回答上面用户提出的问题
{context}
"""
rag_prompt = ChatPromptTemplate.from_messages(
    [
        HumanMessagePromptTemplate.from_template(rag_template),
    ]
)

def retrieval(state: State):
    """检索节点：根据用户问题获取相关文档

    Args:
        state: 当前工作流状态，包含对话历史

    Returns:
        检索到的上下文信息
    """

    messages = []
    if len(state["messages"]) > 0:
        user_query = state["messages"][-1].content
        docs = retrieve_documents(user_query)
        context = "\n".join([doc.page_content for doc in docs])
        messages = rag_prompt.invoke(context).messages
    return dict(messages=messages)

def verify(state: State) -> Literal["chatbot", "ask_human"]:
    """验证节点：判断现有信息是否足够回答用户问题

    Args:
        state: 当前工作流状态，包含检索上下文

    Returns:
        下一节点名称："chatbot"（信息足够）或"ask_human"（需要追问）
    """

    message = HumanMessage("请根据对话历史和上面提供的信息判断，已知的信息是否能够回答用户的问题，直接输出你的判断'Y'或'N'")
    ret = llm.invoke(state["messages"] + [message])
    if 'Y' in ret.content:
        return "chatbot"
    else:
        return "ask_human"

def chatbot(state: State):
    """对话节点：使用检索上下文生成回答

    Args:
        state: 当前工作流状态，包含检索上下文

    Returns:
        AI生成的回答
    """

    return dict(messages=[llm.invoke(state["messages"])])

def ask_human(state: State):
    """人工交互节点：当信息不足时中断流程请求人工输入

    Args:
        state: 当前工作流状态，包含用户原始问题

    Returns:
        人工输入的补充信息
    """

    user_query = state["messages"][-2].content
    human_response = interrupt(dict(question=user_query))
    return dict(messages=[AIMessage(content=human_response)])

graph_builder = StateGraph(State)
graph_builder.add_node("retrieval", retrieval)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("ask_human", ask_human)
graph_builder.add_edge(START, "retrieval")
graph_builder.add_conditional_edges("retrieval", verify)
graph_builder.add_edge("chatbot", END)
graph_builder.add_edge("ask_human", END)
graph = graph_builder.compile(checkpointer=memory)

def stream_graph_updates(user_input):
    """启动工作流并处理流式输出

    Args:
        user_input: 用户输入的问题

    Returns:
        当需要人工输入时返回问题字符串，否则返回None
    """
    msg = HumanMessage(user_input)
    for event in graph.stream(dict(messages=[msg]), thread_config):
        print("节点：", " ".join([key for key in event.keys()]))
        for value in event.values():

            if isinstance(value, tuple):
                # 处理人工中断请求，提取需要追问的问题
                return value[0].value['question']
            elif "messages" in value and isinstance(value["messages"], list) and len(value["messages"]) > 0 and isinstance(value["messages"][-1], AIMessage):
                # 处理AI回答，打印并返回
                print("Assistant:", value["messages"][-1].content)
                return None
    return None

def resume_graph_updates(human_input):
    for event in graph.stream(Command(resume=human_input), thread_config, stream_mode="updates"):
        print("节点（resume）：", " ".join([key for key in event.keys()]))
        for value in event.values():
            if "messages" in value and isinstance(value["messages"][-1], AIMessage):
                print("Assistant:", value["messages"][-1].content)

def main():
    """主函数：启动对话交互循环"""
    while True:
        user_input = input("You: ")
        if user_input.strip() in ("exit", "quit", ""):
            break
        question = stream_graph_updates(user_input)
        if question:
            human_answer = input("Ask Human: " + question + "\nHuman: ")
            resume_graph_updates(human_answer)

if __name__ == "__main__":
    main()
