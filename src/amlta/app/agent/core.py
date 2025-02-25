from typing import Any, ClassVar, Literal, TypedDict

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

The only aggregations/operators allowed to be conveyed in the query are total, average, sum, or count.
If the aggregation is not specified, leave it out.
If the query is about a different aggregation or operator (lowest/biggest/...), treat it as no
aggregation and leave it out as well.

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


class RemoveFlowAction(BaseModel):
    justification: str = Field(
        description="Why this flow should be removed from the result"
    )
    index: int = Field(description="Index of the flow to be removed")

    @field_validator("index", mode="before")
    @classmethod
    def validate_index(cls, index: Any):
        if index is None:
            return -1

        return int(index)


class FlowValidation(BaseModel):
    justification: str = Field(
        description="Why the flows should or should not be removed from the result"
    )
    removals: list[RemoveFlowAction] = Field(
        description="List of flows to be removed from the result. If none need to be removed, return an empty list."
    )

    @field_validator("removals", mode="before")
    @classmethod
    def validate_removals(cls, removals: Any):
        return [removal for removal in removals if removal["index"] != -1]


class FilteredFlows(BaseModel):
    flow_indices: list[int]
    aggregation: str


class FinalFlows(BaseModel):
    query: FlowsQuery
    filtered: list[dict]


class FinalFlowsList(BaseModel):
    join_type: Literal["intersection", "union"]
    flows: list[FinalFlows]


class PandasCodeOutput(BaseModel):
    justification: str = Field(description="Why this code is answers the user question")
    code: str = Field(description="The body of the `analyze_results` function")


class ProcessFlowAnalysisResult(BaseModel):
    code: PandasCodeOutput
    result: dict | list[dict] | None
    exception: str | None


class AgentOutput(TypedDict):
    initial_question: str
    rewritten_process_query: RewrittenProcessQuery
    selected_process: SelectedProcess
    selected_process_uuid: str
    rewritten_flows_queries: FlowQueries
    final_flows: FinalFlowsList
    analysis_result: ProcessFlowAnalysisResult
    final_answer: str


# Agent Events

EventCategory = Literal[
    "process_selection",
    "flows_selection",
    "flows_analysis",
    "agent_finished",
]


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


class FetchingFlowsEvent(BaseModel):
    category: ClassVar[EventCategory] = "flows_selection"
    type: Literal["fetching_flows"] = "fetching_flows"


class FetchedFlowsEvent(BaseModel):
    category: ClassVar[EventCategory] = "flows_selection"
    type: Literal["fetched_flows"] = "fetched_flows"
    flows: FinalFlowsList


class AnalyzingFlowsEvent(BaseModel):
    category: ClassVar[EventCategory] = "flows_analysis"
    type: Literal["analyzing_flows"] = "analyzing_flows"


class AnalyzedFlowsEvent(BaseModel):
    category: ClassVar[EventCategory] = "flows_analysis"
    type: Literal["analyzed_flows"] = "analyzed_flows"
    result: ProcessFlowAnalysisResult


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
        | FetchingFlowsEvent
        | FetchedFlowsEvent
        | AnalyzingFlowsEvent
        | AnalyzedFlowsEvent
        | AgentFinishedEvent
    ) = Field(discriminator="type")
