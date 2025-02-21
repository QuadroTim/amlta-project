from functools import cached_property
from typing import ClassVar

from langchain_core.tools import BaseToolkit
from langchain_core.tools.base import BaseTool
from pydantic import ConfigDict

from amlta.app.rag.collections import Collections

from .search_flows import SearchFlowsTool
from .search_processes import SearchProcessTool
from .select_process import SelectProcessTool


class Toolkit(BaseToolkit):
    model_config: ClassVar[ConfigDict] = {
        "arbitrary_types_allowed": True,
    }

    collections: Collections

    @cached_property
    def search_process(self) -> SearchProcessTool:
        return SearchProcessTool(collections=self.collections)

    @cached_property
    def select_process(self) -> SelectProcessTool:
        return SelectProcessTool()

    @cached_property
    def search_flows(self) -> SearchFlowsTool:
        return SearchFlowsTool()

    def get_tools(self) -> list[BaseTool]:
        return [
            self.search_process,
            self.select_process,
            self.search_flows,
        ]
