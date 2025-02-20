from langchain_core.tools import BaseToolkit
from langchain_core.tools.base import BaseTool
from pydantic import Field

from .search_flows import SearchFlowsTool
from .search_processes import SearchProcessTool


class Toolkit(BaseToolkit):
    search_process: SearchProcessTool = Field(default_factory=SearchProcessTool)
    search_flows: SearchFlowsTool = Field(default_factory=SearchFlowsTool)

    def get_tools(self) -> list[BaseTool]:
        return (super().get_tools() or []) + [self.search_process, self.search_flows]
