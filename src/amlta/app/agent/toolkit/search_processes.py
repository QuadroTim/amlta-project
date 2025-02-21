from typing import ClassVar

from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from langgraph.types import Command
from pydantic import ConfigDict

from amlta.app.agent.toolkit.base import BaseToolInput
from amlta.app.rag.collections import Collections


class SearchProcessToolInput(BaseToolInput):
    query: str


class SearchProcessTool(BaseTool):
    model_config: ClassVar[ConfigDict] = {
        "arbitrary_types_allowed": True,
    }

    collections: Collections

    args_schema: ArgsSchema | None = SearchProcessToolInput

    name: str = "search_process"
    description: str = "Search for a process."

    def _format_process_doc(self, process: Document) -> Document:
        content = process.page_content
        uuid = process.metadata["uuid"]
        new_content = f"# Process {uuid}\n{content}"

        return process.model_copy(update={"page_content": new_content})

    def _run(self, *, query: str, tool_call_id: str) -> Command:
        docs = self.collections.processes.similarity_search(query, k=5)
        updated_docs = [self._format_process_doc(doc) for doc in docs]

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="\n\n".join(doc.page_content for doc in updated_docs),
                        tool_call_id=tool_call_id,
                    )
                ],
                "candidate_processes": docs,
            }
        )
