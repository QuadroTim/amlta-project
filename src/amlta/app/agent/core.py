from typing import ClassVar, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator

from amlta.app.rag.client import get_qdrant_client
from amlta.app.rag.collections import get_collections
from amlta.probas.processes import ProcessData

qdrant_client = get_qdrant_client("data/qdrant-yaml")
collections = get_collections(qdrant_client)


class RewrittenProcessQuery(BaseModel):
    justification: str = Field(
        description="Why this query is suitable for searching processes"
    )
    query: str = Field(description="Rewritten process query")


flow_query_field_description = """
The query must be in the format 'What <is/are> the [input/output] [<aggregation>] <query> of the process?'.

Examples:
- "What are the total output emissions to air of the process?"
- "What are the emissions of the process?"
- "What are the output values for carbon dioxide, methane, and nitrous oxide of the process?"
""".strip()


class FlowsQuery(BaseModel):
    justification: str = Field(
        description="Why this query is suitable for searching flows"
    )
    query: str = Field(description=flow_query_field_description)

    @field_validator("query", mode="after")
    @classmethod
    def validate_query(cls, query: str):
        if not query.endswith("the process?"):
            raise ValueError("query must end with 'the process?'")

        return query


flow_query_join_type_description = """
If there are multiple subqueries, you can use 'intersection' or 'union'. Think about
if the user wants to know all the flows of all subqueries separately (join_type='union') or the users
wants a single results where both subqueries are applied as filters (join_type='intersection').
""".strip()


class FlowQueries(BaseModel):
    queries: list[FlowsQuery] = Field(description="List of flow queries")
    join_type: Literal["intersection", "union"] = Field(
        "union", description=flow_query_join_type_description
    )


class SelectedProcess(BaseModel):
    justification: str = Field(
        description="Why this process is suitable for the user question"
    )
    index: int = Field(description="Index of the selected process")


class AgentOutput(TypedDict):
    initial_question: str
    rewritten_process_query: RewrittenProcessQuery
    selected_process: SelectedProcess
    selected_process_uuid: str
    rewritten_flows_queries: FlowQueries
    flows_indices: list[int]
    aggregation: str


# Agent Events

EventCategory = Literal["process_selection", "flows_selection", "agent_finished"]


class RewritingProcessQueryEvent(BaseModel):
    category: ClassVar[EventCategory] = "process_selection"
    type: Literal["rewriting_process_query"] = "rewriting_process_query"


class RewrittenProcessQueryEvent(BaseModel):
    category: ClassVar[EventCategory] = "process_selection"
    type: Literal["rewritten_process_query"] = "rewritten_process_query"
    query: RewrittenProcessQuery


class ProcessCandidatesFetchedEvent(BaseModel):
    category: ClassVar[EventCategory] = "process_selection"
    type: Literal["process_candidates_fetched"] = "process_candidates_fetched"
    candidates: list[ProcessData]


class SelectingProcessEvent(BaseModel):
    category: ClassVar[EventCategory] = "process_selection"
    type: Literal["selecting_process"] = "selecting_process"


class SelectedProcessEvent(BaseModel):
    category: ClassVar[EventCategory] = "process_selection"
    type: Literal["selected_process"] = "selected_process"
    process: SelectedProcess
    process_uuid: str


class RewritingFlowsQueriesEvent(BaseModel):
    category: ClassVar[EventCategory] = "flows_selection"
    type: Literal["rewriting_flows_queries"] = "rewriting_flows_queries"


class RewrittenFlowsQueriesEvent(BaseModel):
    category: ClassVar[EventCategory] = "flows_selection"
    type: Literal["rewritten_flows_queries"] = "rewritten_flows_queries"
    rewritten_flows_queries: FlowQueries


class AgentFinishedEvent(BaseModel):
    category: ClassVar[EventCategory] = "agent_finished"
    type: Literal["agent_finished"] = "agent_finished"
    result: AgentOutput


class AgentEvent(BaseModel):
    event: (
        RewritingProcessQueryEvent
        | RewrittenProcessQueryEvent
        | ProcessCandidatesFetchedEvent
        | SelectingProcessEvent
        | SelectedProcessEvent
        | RewritingFlowsQueriesEvent
        | RewrittenFlowsQueriesEvent
        | AgentFinishedEvent
    ) = Field(discriminator="type")

    @classmethod
    def make(cls, data: dict) -> "AgentEvent":
        return cls(event=data)  # type: ignore
