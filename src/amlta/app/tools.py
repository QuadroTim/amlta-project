from typing import Annotated

from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, BaseToolkit
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel, Field


class BaseToolInput(BaseModel):
    tool_call_id: Annotated[str, InjectedToolCallId]


class SearchProcessToolInput(BaseToolInput):
    query: str


class SearchProcessTool(BaseTool):
    args_schema: type[BaseModel] = SearchProcessToolInput

    name: str = "search_process"
    description: str = "Search for a process."

    def _run(self, *, query: str, tool_call_id: str) -> Command:
        # TODO: Implement search
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="Process 1",
                        tool_call_id=tool_call_id,
                    )
                ],
                "candidate_processes": [
                    Document(page_content="Process 1", metadata={"id": "1"})
                ],
            }
        )


class Toolkit(BaseToolkit):
    search_process: SearchProcessTool = Field(default_factory=SearchProcessTool)

    def get_tools(self) -> list[BaseTool]:
        return (super().get_tools() or []) + [self.search_process]
