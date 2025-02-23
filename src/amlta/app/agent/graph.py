import asyncio
import json
from collections import Counter
from collections.abc import Awaitable
from typing import cast

import streamlit as st
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import StreamWriter
from transformers import (
    TapasForQuestionAnswering,
    TapasTokenizer,
)

from amlta.app.agent.core import (
    AgentEvent,
    AgentFinishedEvent,
    AgentOutput,
    FlowQueries,
    FlowsQuery,
    ProcessCandidatesFetchedEvent,
    RewritingFlowsQueriesEvent,
    RewritingProcessQueryEvent,
    RewrittenFlowsQueriesEvent,
    RewrittenProcessQuery,
    RewrittenProcessQueryEvent,
    SelectedProcess,
    SelectedProcessEvent,
    SelectingProcessEvent,
    collections,
)
from amlta.app.llm import get_ollama
from amlta.formatting.data import create_process_section
from amlta.probas.flows import extract_process_flows
from amlta.probas.processes import ProcessData
from amlta.tapas.model import (
    load_tapas_model as _load_tapas_model,
)
from amlta.tapas.model import (
    load_tapas_tokenizer as _load_tapas_tokenizer,
)
from amlta.tapas.retrieve import retrieve_rows


def inspect_prompt(input: dict):
    print(input)
    return input


@st.cache_resource
def load_tapas_tokenizer() -> TapasTokenizer:
    return _load_tapas_tokenizer()


@st.cache_resource
def load_tapas_model() -> TapasForQuestionAnswering:
    return _load_tapas_model()


base_prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n{system_prompt}",
        ),
        ("placeholder", "{history}"),
        ("human", "{human_input}"),
    ]
)


rewrite_process_query_system_prompt = f"""
You are assisting a life cycle inventory (LCI) expert in browsing and querying the PROBAS
life cycle inventory database.

## Instructions ##
Your task is to rewrite the user question to make it more suitable for searching processes in the
PROBAS database.

## Output ##
The query should completely ignore specifics about flows (i.e., inputs/outputs) and focus only on
the process itself.

## Output format ##
`query`: {RewrittenProcessQuery.model_fields["query"].description}
`justification`: {RewrittenProcessQuery.model_fields["justification"].description}

Return only the rewritten query in `query` and a include your reasoning and thought process in
`justification`.
""".strip()


noop_writer = lambda x: None


@task
async def rewrite_process_query(
    user_question: str, writer: StreamWriter = noop_writer
) -> RewrittenProcessQuery:
    writer(AgentEvent(event=RewritingProcessQueryEvent()))

    llm = get_ollama().with_structured_output(RewrittenProcessQuery)

    retriever = collections.glossary.as_retriever(
        search_type="mmr", search_kwargs={"k": 5}
    )

    # def retrieve(input: dict):
    #     return retriever.get_relevant_documents(input["human_input"])

    def human_template(input: dict):
        return {
            **input,
            "human_input": "<glossary>\n{context}\n</glossary>\n<question>{question}</question>".format(
                context="\n\n".join(doc.page_content for doc in input["context"]),
                question=input["human_input"],
            ),
        }

    chain = (
        {
            "context": retriever,
            "human_input": RunnablePassthrough(),
        }
        | RunnableLambda(human_template)
        | base_prompt.partial(system_prompt=rewrite_process_query_system_prompt)
        # | inspect_prompt
        | llm
    )

    res = cast(RewrittenProcessQuery, await chain.ainvoke(user_question))
    writer(AgentEvent(event=RewrittenProcessQueryEvent(query=res)))

    return res


rewrite_flows_query_system_prompt = f"""
You are assisting a life cycle inventory (LCI) expert in browsing and querying the PROBAS
life cycle inventory database.

## Instructions ##
1. Analyze the user question and categorize it into one of the following categories:
    a) the user asks for one or more specific flows. Multiple flows ARE ALLOWED and MUST NOT be
       splitted
    b) the user asks for one specific class of flows
    c) the user asks for one specific type of flow (elementary flow or wast flow)
2. If and only if the question contains multiple categories, decompose the question into multiple
    queries, one for each category.
3. Given the (decomposed) question, formulate a neutral natural question query -- that is, a query that
    excludes the specific process itself. Furthermore, the query must match the syntax explained in the
    schema field description:

## Output format ##
`queries`: {FlowQueries.model_fields["queries"].description}
`join_type`: {FlowQueries.model_fields["join_type"].description}

Per query:
`justification`: {FlowsQuery.model_fields["justification"].description}
`query`: {FlowsQuery.model_fields["query"].description}
""".strip()


@task
async def rewrite_flows_query(
    user_question: str, writer: StreamWriter = noop_writer
) -> FlowQueries:
    writer(AgentEvent(event=RewritingFlowsQueriesEvent()))

    llm = get_ollama().with_structured_output(
        FlowQueries, method="json_schema", include_raw=True
    )
    chain = base_prompt | llm
    history = []

    tries = 0

    while tries < 3:
        resp = cast(
            dict,
            await chain.ainvoke(
                {
                    "system_prompt": rewrite_flows_query_system_prompt,
                    "history": history,
                    "human_input": user_question,
                }
            ),
        )

        if resp["parsing_error"]:
            print(resp["parsing_error"])
            tries += 1

            history.append(
                SystemMessage(
                    content=f"Error: {resp['parsing_error']}\nPlease fix your mistakes."
                )
            )
        else:
            res = cast(FlowQueries, resp["parsed"])
            writer(
                AgentEvent(
                    event=RewrittenFlowsQueriesEvent(rewritten_flows_queries=res)
                )
            )
            return res
    else:
        raise RuntimeError("Failed to rewrite flows query after 3 attempts")


select_process_system_prompt = f"""
You are assisting a life cycle inventory (LCI) expert in browsing and querying the PROBAS
life cycle inventory database.

## Instructions ##
Your task is to select the best fitting process from the list of candidates.
You will receive a list of candidate processes and you must select the best fitting one.

## Output format ##
`justification`: {SelectedProcess.model_fields["justification"].description}
`index`: {SelectedProcess.model_fields["index"].description}
""".strip()


@task
async def select_process(
    candidates: list[ProcessData],
    user_question: str,
    writer: StreamWriter = noop_writer,
) -> SelectedProcess:
    writer(AgentEvent(event=SelectingProcessEvent()))

    llm = get_ollama().with_structured_output(SelectedProcess)
    chain = base_prompt | llm

    def _format_candidate(index: int, process: ProcessData) -> str:
        data = create_process_section(process, include_flows=False)

        return "<candidate index={index}>\n{data}\n</candidate>".format(
            index=index, data=json.dumps(data, indent=2)
        )

    human_template = (
        "<question>{question}</question>\n<processes>\n{candidates}\n</processes>"
    )

    res = cast(
        SelectedProcess,
        await chain.ainvoke(
            {
                "system_prompt": select_process_system_prompt,
                "human_input": human_template.format(
                    question=user_question,
                    candidates="\n\n".join(
                        _format_candidate(idx, c) for idx, c in enumerate(candidates)
                    ),
                ),
            }
        ),
    )
    writer(
        AgentEvent(
            event=SelectedProcessEvent(
                process=res,
                process_uuid=candidates[
                    res.index
                ].processInformation.dataSetInformation.UUID,
            )
        )
    )
    return res


@entrypoint(checkpointer=MemorySaver())
async def main(user_question: str, writer: StreamWriter) -> AgentOutput:
    rewritten_process_query_fut = cast(
        Awaitable[RewrittenProcessQuery], rewrite_process_query(user_question)
    )

    rewritten_flows_query_fut = cast(
        Awaitable[FlowQueries], rewrite_flows_query(user_question)
    )

    rewritten_process_query_resp = await rewritten_process_query_fut
    rewritten_process_query = rewritten_process_query_resp.query

    candidate_processes_docs = collections.processes.similarity_search(
        rewritten_process_query, k=5
    )
    candidate_processes = [
        ProcessData.from_uuid(doc.metadata["uuid"]) for doc in candidate_processes_docs
    ]
    writer(
        AgentEvent(event=ProcessCandidatesFetchedEvent(candidates=candidate_processes))
    )

    selected_process_resp = cast(
        SelectedProcess,
        await select_process(candidate_processes, rewritten_process_query),
    )

    selected_process = candidate_processes[selected_process_resp.index]
    selected_process_uuid = selected_process.processInformation.dataSetInformation.UUID

    rewritten_flows_queries = await rewritten_flows_query_fut

    model = load_tapas_model()
    tokenizer = load_tapas_tokenizer()
    flows_df = extract_process_flows(selected_process)

    all_indices = None
    aggregations = Counter()
    set_operator = (
        set.intersection if rewritten_flows_queries.join_type == "and" else set.union
    )

    tasks = []
    for query in rewritten_flows_queries.queries:
        tasks.append(
            asyncio.to_thread(
                lambda: retrieve_rows(
                    flows_df,
                    query.query,
                    model=model,
                    tokenizer=tokenizer,
                    threshold=0.75,
                )
            )
        )

    for flow_indices, aggregation in await asyncio.gather(*tasks):
        aggregations[aggregation] += 1

        if all_indices is None:
            all_indices = set(flow_indices)
        else:
            all_indices = set_operator(all_indices, set(flow_indices))

    assert all_indices is not None
    flow_indices = sorted(all_indices)
    aggregation = aggregations.most_common(1)[0][0]

    res: AgentOutput = {
        "initial_question": user_question,
        "rewritten_process_query": rewritten_process_query_resp,
        "selected_process": selected_process_resp,
        "selected_process_uuid": selected_process_uuid,
        "rewritten_flows_queries": rewritten_flows_queries,
        "flows_indices": flow_indices,
        "aggregation": aggregation,
    }

    writer(AgentEvent(event=AgentFinishedEvent(type="agent_finished", result=res)))
    return res
