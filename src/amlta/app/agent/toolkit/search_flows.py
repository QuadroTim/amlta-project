from typing import Annotated

import streamlit as st
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel
from transformers import TapasForQuestionAnswering, TapasTokenizer

from amlta.app.agent.state import AgentState, FlowsResult
from amlta.app.agent.toolkit.base import BaseToolInput
from amlta.probas.flows import extract_process_flows
from amlta.tapas.model import (
    load_tapas_model,
    load_tapas_tokenizer,
)
from amlta.tapas.retrieve import retrieve_rows


@st.cache_resource
def load_tokenizer() -> TapasTokenizer:
    return load_tapas_tokenizer()


@st.cache_resource
def load_model() -> TapasForQuestionAnswering:
    return load_tapas_model().model


TOOL_DESCRIPTION = """
This tool allows you to search for process flows that match a given quantifiable query.

Prior to using the tool you must interpret the user question and rewrite it to a specific query.

The query must be in the format 'What <is/are> the [input/output] [<aggregation>] <query> of the process?'.

Examples:
- What are the total output emissions to air of the process?
- What are the emissions of the process? (agnostic of input/output, no aggregation -> list the flows)
- What are the output values for carbon dioxide, methane, and nitrous oxide of the process?
""".strip()


class SearchFlowsToolInput(BaseToolInput):
    query: str
    state: Annotated[dict, InjectedState]


class SearchFlowsTool(BaseTool):
    args_schema: type[BaseModel] = SearchFlowsToolInput

    name: str = "search_flows"
    description: str = TOOL_DESCRIPTION

    def _run(self, *, query: str, state: AgentState, tool_call_id: str) -> Command:
        model = load_model()
        tokenizer = load_tokenizer()
        process = state["selected_process"]
        flows_df = extract_process_flows(process)

        flows, aggregation = retrieve_rows(
            flows_df, query, model=model, tokenizer=tokenizer
        )

        return Command(
            update={
                "messages": ToolMessage(
                    content=f"Found {len(flows)} flows.", tool_call_id=tool_call_id
                ),
                "flows_result": FlowsResult(
                    flow_uuids=[flow.flow_uuid for flow in flows],
                    aggregation=aggregation,
                ),
            }
        )
