from typing import Annotated

from langchain_core.tools.base import InjectedToolCallId
from pydantic import BaseModel


class BaseToolInput(BaseModel):
    tool_call_id: Annotated[str, InjectedToolCallId]
