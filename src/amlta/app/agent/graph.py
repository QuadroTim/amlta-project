from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from amlta.app.agent.state import AgentState
from amlta.app.agent.toolkit import Toolkit
from amlta.app.llm import ollama
from amlta.app.rag.client import get_qdrant_client
from amlta.app.rag.collections import Collections, get_collections, iter_collection
from amlta.app.rag.loaders import MarkdownGlossaryLoader, MarkdownProcessLoader


def load_documents(collections: Collections):
    glossary_loader = MarkdownGlossaryLoader()
    store_uuids = {record.id for record in iter_collection(collections.glossary)}

    for doc in glossary_loader.load():
        if doc.id in store_uuids:
            continue

        collections.glossary.add_documents([doc])

    process_loader = MarkdownProcessLoader()
    store_uuids = {record.id for record in iter_collection(collections.processes)}

    for doc in process_loader.load():
        if doc.id in store_uuids:
            continue

        collections.processes.add_documents([doc])


qdrant_client = get_qdrant_client()
collections = get_collections(qdrant_client)

toolkit = Toolkit()
tools = toolkit.get_tools()

ollama_with_tools = ollama.bind_tools(tools)


def chatbot(state: AgentState):
    return {"messages": [ollama_with_tools.invoke(state["messages"])]}


flow = StateGraph(AgentState)

flow.add_node("chatbot", chatbot)
flow.add_node("tools", ToolNode(tools))

flow.add_conditional_edges("chatbot", tools_condition)
flow.add_edge("tools", "chatbot")
flow.add_edge("chatbot", END)

flow.set_entry_point("chatbot")

graph = flow.compile()
