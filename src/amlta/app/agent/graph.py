import json
from typing import TypedDict, cast

import streamlit as st
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
from pydantic import BaseModel, Field, field_validator
from transformers import (
    TapasForQuestionAnswering,
    TapasTokenizer,
)

from amlta.app.agent.core import collections
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
    return _load_tapas_model().model


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


rewrite_process_query_system_prompt = """
You are assisting a life cycle inventory (LCI) expert in browsing and querying the PROBAS
life cycle inventory database.

Your task is to rewrite the user question to make it more suitable for searching processes in the
PROBAS database.

The query should completely ignore specifics about flows (i.e., inputs/outputs) and focus only on
the process itself.
""".strip()


@task
def rewrite_process_query(user_question: str) -> str:
    llm = get_ollama()
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
        | inspect_prompt
        | llm
    )

    return cast(str, chain.invoke(user_question).content)


rewrite_flows_query_system_prompt = """
You are assisting a life cycle inventory (LCI) expert in browsing and querying the PROBAS
life cycle inventory database.

Given the user question, formulate a neutral natural question query -- that is, a query that
excludes the specific process itself. Furthermore, the query must match the syntax explained in the
schema field description:

The `query` must be in the format 'What <is/are> the [input/output] [<aggregation>] <query> of the process?'.

Examples:
- "What are the total output emissions to air of the process?"
- "What are the emissions of the process?"
- "What are the output values for carbon dioxide, methane, and nitrous oxide of the process?"
""".strip()


flow_query_field_description = """
The query must be in the format 'What <is/are> the [input/output] [<aggregation>] <query> of the process?'.

Examples:
- "What are the total output emissions to air of the process?"
- "What are the emissions of the process?"
- "What are the output values for carbon dioxide, methane, and nitrous oxide of the process?"
""".strip()


class FlowsQuery(BaseModel):
    query: str = Field(description=flow_query_field_description)
    justification: str = Field(
        description="Why this query is suitable for searching flows"
    )

    @field_validator("query", mode="after")
    @classmethod
    def validate_query(cls, query: str):
        if not query.endswith("the process?"):
            raise ValueError("query must end with 'the process?'")

        return query


@task
def rewrite_flows_query(user_question: str) -> FlowsQuery:
    llm = get_ollama().with_structured_output(
        FlowsQuery, method="json_schema", include_raw=True
    )
    chain = base_prompt | llm
    history = []

    tries = 0

    while tries < 3:
        resp = cast(
            dict,
            chain.invoke(
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
            return cast(FlowsQuery, resp["parsed"])
    else:
        raise RuntimeError("Failed to rewrite flows query after 3 attempts")


select_process_system_prompt = """
You are assisting a life cycle inventory (LCI) expert in browsing and querying the PROBAS
life cycle inventory database.

Your task is to select the best fitting process from the list of candidates.
You will receive a list of candidate processes and you must select the best fitting one.
""".strip()


class SelectedProcess(BaseModel):
    index: int
    justification: str = Field(
        description="Why this process is suitable for the user question"
    )


@task
def select_process(
    candidates: list[ProcessData], user_question: str
) -> SelectedProcess:
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

    return cast(
        SelectedProcess,
        chain.invoke(
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


class Output(TypedDict):
    initial_question: str
    rewritten_process_query: str
    selected_process: SelectedProcess
    selected_process_uuid: str
    rewritten_flows_query: FlowsQuery
    flows_indices: list[int]
    aggregation: str


@entrypoint(checkpointer=MemorySaver())
def main(user_question: str) -> Output:
    rewritten_process_query_fut = rewrite_process_query(user_question)
    rewritten_flows_query_fut = rewrite_flows_query(user_question)

    rewritten_process_query = rewritten_process_query_fut.result()

    candidate_processes_docs = collections.processes.similarity_search(
        rewritten_process_query, k=5
    )
    candidate_processes = [
        ProcessData.from_uuid(doc.metadata["uuid"]) for doc in candidate_processes_docs
    ]

    selected_process_resp = select_process(
        candidate_processes, rewritten_process_query
    ).result()

    print(selected_process_resp)

    selected_process = candidate_processes[selected_process_resp.index]
    selected_process_uuid = selected_process.processInformation.dataSetInformation.UUID

    rewritten_flows_query = rewritten_flows_query_fut.result()
    model = load_tapas_model()
    tokenizer = load_tapas_tokenizer()
    flows_df = extract_process_flows(selected_process)

    flow_indices, aggregation = retrieve_rows(
        flows_df,
        rewritten_flows_query.query,
        model=model,
        tokenizer=tokenizer,
        threshold=0.75,
    )
    print(flows_df.iloc[flow_indices])

    # filtered_df = flows_df.iloc[flow_indices].copy().reset_index(drop=True)

    return {
        "initial_question": user_question,
        "rewritten_process_query": rewritten_process_query,
        "selected_process": selected_process_resp,
        "selected_process_uuid": selected_process_uuid,
        "rewritten_flows_query": rewritten_flows_query,
        "flows_indices": flow_indices,
        "aggregation": aggregation,
    }
