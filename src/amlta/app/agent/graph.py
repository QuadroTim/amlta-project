from typing import Literal

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from amlta.app.agent.state import AgentState
from amlta.app.agent.toolkit import Toolkit
from amlta.app.llm import get_ollama
from amlta.app.rag.client import get_qdrant_client
from amlta.app.rag.collections import get_collections

qdrant_client = get_qdrant_client()
collections = get_collections(qdrant_client)

toolkit = Toolkit(collections=collections)
tools = toolkit.get_tools()

system_prompt = """
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

You are assisting a life cycle inventory (LCI) expert in browsing and querying the PROBAS
life cycle inventory database.

Your task is to use your given tools to help users query and analyze processes of the database.

## Tools
1. search_process: This tool allows you to search for processes that match a given user query.
Use a keyword-style query to search for the process, e.g., "energy from wind turbines, germany 2015".
2. select_process: This tool allows you to select a process from a list of candidates.
3. search_flows: This tool allows you to search for process flows that match a given quantifiable query.

## Instructions
Follow every of these steps in sequence.
1. You are given a user question that you must interpret and use to search for a process in the
PROBAS database using the search_process tool.
2. With the select_process tool, you must select the best fitting process from the list of candidates.
3. With the selected process, you must use the search_flows tool to search for process flows that
match a given quantifiable query.
""".strip()


def edge_condition(
    state: AgentState,
) -> Literal["search_process", "select_process", "search_flows", "tools", "__end__"]:
    ai_message = state["messages"][-1]

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:  # type: ignore
        return "tools"

    if not state.get("candidate_processes"):
        return "search_process"

    if not state.get("selected_process"):
        return "select_process"

    if not state.get("flows_result"):
        return "search_flows"

    return "__end__"


def search_process(state: AgentState, config: RunnableConfig):
    return {
        "messages": [
            get_ollama(**config.get("configurable", {"ollama": {}})["ollama"])
            .bind_tools([toolkit.search_process])
            .invoke(state["messages"], config=config)
        ]
    }


def select_process(state: AgentState, config: RunnableConfig):
    return {
        "messages": [
            get_ollama(**config.get("configurable", {"ollama": {}})["ollama"])
            .bind_tools([toolkit.select_process])
            .invoke(state["messages"], config=config)
        ]
    }


def search_flows(state: AgentState, config: RunnableConfig):
    return {
        "messages": [
            get_ollama(**config.get("configurable", {"ollama": {}})["ollama"])
            .bind_tools([toolkit.search_flows])
            .invoke(state["messages"], config=config)
        ]
    }


flow = StateGraph(AgentState)

flow.add_node("search_process", search_process)
flow.add_node("select_process", select_process)
flow.add_node("search_flows", search_flows)

flow.add_node("tools", ToolNode(tools))

flow.add_conditional_edges("search_process", tools_condition)
flow.add_conditional_edges("select_process", tools_condition)
flow.add_conditional_edges("search_flows", tools_condition)

flow.add_conditional_edges("tools", edge_condition)

flow.set_entry_point("search_process")

graph = flow.compile()
