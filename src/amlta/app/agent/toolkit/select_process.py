from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from amlta.app.agent.state import AgentState
from amlta.app.agent.toolkit.base import BaseToolInput
from amlta.probas.processes import ProcessData


class SelectProcessToolInput(BaseToolInput):
    process_uuid: str
    state: Annotated[dict, InjectedState]


class SelectProcessTool(BaseTool):
    args_schema: ArgsSchema | None = SelectProcessToolInput

    name: str = "select_process"
    description: str = (
        "This tool allows you to select a process from a list of candidates."
    )

    def _run(
        self, *, process_uuid: str, state: AgentState, tool_call_id: str
    ) -> Command:
        process = ProcessData.from_uuid(process_uuid)

        return Command(
            update={
                "selected_process": process,
                "messages": [
                    ToolMessage(
                        content=f"Selected process: {process}",
                        tool_call_id=tool_call_id,
                    ),
                ],
            },
        )
