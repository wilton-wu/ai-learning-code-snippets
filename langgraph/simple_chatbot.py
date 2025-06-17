from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

load_dotenv()
llm = init_chat_model(model="openai:gpt-4.1")
memory = MemorySaver()
config = {"configurable": {"thread_id": "1234"}}


class State(TypedDict):
    # 状态变量 messages 类型是 list，更新方式是 add_messages
    # add_messages 是内置的一个方法，将新的消息列表追加在原列表后面
    messages: Annotated[list, add_messages]


@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response


tools = [get_weather, human_assistance]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools=tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph = graph_builder.compile(checkpointer=memory)


def stream_graph_updates(user_input):
    state = {"messages": [{"role": "user", "content": user_input}]}
    for event in graph.stream(state, config, stream_mode="values"):
        if "messages" in event:
            print(f"对话数量：{len(event['messages'])}")
            event["messages"][-1].pretty_print()
        elif "__interrupt__" in event:
            return event["__interrupt__"][0].value["query"]
    return None


def resume_graph_updates(human_input):
    human_command = Command(resume=human_input)
    for event in graph.stream(human_command, config, stream_mode="values"):
        if "messages" in event:
            event["messages"][-1].pretty_print()


def main():
    while True:
        user_input = input("You: ")
        if user_input.strip() in ("exit", "quit", ""):
            break
        query = stream_graph_updates(user_input)
        if query:
            human_response = input("Ask Human: " + query + "\nHuman: ")
            resume_graph_updates(human_response)


if __name__ == "__main__":
    main()
