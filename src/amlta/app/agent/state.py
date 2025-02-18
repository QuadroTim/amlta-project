from typing import NamedTuple

from langchain_core.documents import Document
from langgraph.graph import MessagesState

from amlta.probas.processes import ProcessData


class FlowsResult(NamedTuple):
    flow_uuids: list[str]
    aggregation: str


class AgentState(MessagesState):
    initial_question: str
    candidate_processes: list[Document]
    selected_process: ProcessData
    flows_result: FlowsResult
