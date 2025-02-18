import streamlit as st
from langchain_core.documents import Document
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from amlta.app.llm import get_model
from amlta.app.rag.client import get_qdrant_client
from amlta.app.rag.collections import Collections, get_collections, iter_collection
from amlta.app.rag.loaders import MarkdownGlossaryLoader, MarkdownProcessLoader
from amlta.app.tools import Toolkit
from amlta.probas.processes import ProcessData


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


class State(MessagesState):
    initial_question: str
    candidate_processes: list[Document]
    selected_process: ProcessData


@st.cache_resource
def get_graph(ollama_model: str | None = None, ollama_base_url: str | None = None):
    qdrant_client = get_qdrant_client()
    collections = get_collections(qdrant_client)
    llm = get_model(model=ollama_model, base_url=ollama_base_url)

    toolkit = Toolkit()
    tools = toolkit.get_tools()
    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    flow = StateGraph(State)

    flow.add_node("chatbot", chatbot)
    flow.add_node("tools", ToolNode(tools))

    flow.add_conditional_edges("chatbot", tools_condition)
    flow.add_edge("tools", "chatbot")
    flow.add_edge("chatbot", END)

    flow.set_entry_point("chatbot")

    return flow.compile()
