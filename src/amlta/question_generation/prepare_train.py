import functools
import json
from os import PathLike
from pathlib import Path
from typing import (
    NamedTuple,
    Sequence,
    TypedDict,
)

import pandas as pd

from amlta.data_processing.tapas_flows import transform_flows_for_tapas
from amlta.probas import flows, processes
from amlta.question_generation.generate import LCIQuestion, get_generated_questions_path
from amlta.question_generation.query_params import FlowQueryParams, get_flows_for_query


@functools.lru_cache(maxsize=128)
def get_process(process_uuid: str) -> processes.ProcessData:
    return processes.ProcessData.from_uuid(process_uuid)


@functools.lru_cache(maxsize=128)
def get_flows_df(process_uuid: str) -> pd.DataFrame:
    process = get_process(process_uuid)
    return flows.extract_process_flows(process)


class LCIQuestionBatchResult(NamedTuple):
    batch_name: str
    process_uuid: str
    question_id: int
    query_params: FlowQueryParams
    question: LCIQuestion


class QuestionData(TypedDict):
    batch: str
    process_uuid: str
    question_id: int
    basic_query: str
    general_query: str
    specific_query: str
    flow_query_params: FlowQueryParams
    question: str
    question_replaced_basic: str
    question_replaced_general: str
    question_replaced_specific: str


def process_questions(
    questions: Sequence[LCIQuestionBatchResult],
) -> list[QuestionData]:
    data: list[QuestionData] = []

    assert questions
    batch_name = questions[0].batch_name

    for process_question in questions:
        question = process_question.question
        question_id = process_question.question_id
        specific_query = question.process_keywords.specific_search
        general_query = question.process_keywords.general_search
        basic_query = question.process_keywords.basic_search
        question_text = question.flow_question.question
        question_replaced_specific = question_text.replace(
            "<the process>", specific_query
        )
        question_replaced_general = question_text.replace(
            "<the process>", general_query
        )
        question_replaced_basic = question_text.replace("<the process>", basic_query)

        data.append(
            {
                "batch": process_question.batch_name,
                "process_uuid": process_question.process_uuid,
                "question_id": question_id,
                "basic_query": basic_query,
                "general_query": general_query,
                "specific_query": specific_query,
                "flow_query_params": process_question.query_params,
                "question": question_text,
                "question_replaced_basic": question_replaced_basic,
                "question_replaced_general": question_replaced_general,
                "question_replaced_specific": question_replaced_specific,
            }
        )

    qa_gen_dir = get_generated_questions_path()
    out_dir = qa_gen_dir / "out"
    out_dir.mkdir(exist_ok=True)

    out_file = out_dir / f"{batch_name}.jsonl"
    out_file.write_text("\n".join(json.dumps(item) for item in data))

    print(f"Saved {len(data)} questions to {out_file}")

    return data


def process_batch_results(batch_output_file: PathLike) -> list[QuestionData]:
    batch_output_file = Path(batch_output_file)
    batch_input_filename = batch_output_file.stem + "_input"

    qa_gen_dir = get_generated_questions_path()
    batch_input_file = (qa_gen_dir / "batch_inputs" / batch_input_filename).with_suffix(
        ".jsonl"
    )

    batch_input = [
        json.loads(line) for line in batch_input_file.read_text().strip().splitlines()
    ]

    data = [
        json.loads(line)
        for line in Path(batch_output_file).read_text().strip().splitlines()
    ]

    responses = []

    for item in data:
        process_uuid, question_id = item["custom_id"].split("/")
        question_id = int(question_id)

        response_raw_json = item["response"]["body"]["choices"][0]["message"]["content"]
        response = LCIQuestion.model_validate_json(response_raw_json)

        query_params = next(
            item
            for item in batch_input
            if item["process_uuid"] == process_uuid
            and item["question_id"] == question_id
        )["query_params"]

        responses.append(
            LCIQuestionBatchResult(
                batch_name=batch_output_file.stem,
                process_uuid=process_uuid,
                question_id=question_id,
                query_params=query_params,
                question=response,
            )
        )

    return process_questions(responses)


# TAPAS has a max token limit of 512, so we need to split the training data into batches
def get_training_batches(question: QuestionData, batch_size: int = 20) -> pd.DataFrame:
    dfs = []

    query_params = question["flow_query_params"]

    flows_df = get_flows_df(question["process_uuid"])
    flows_df_tapas = transform_flows_for_tapas(flows_df)
    flows_df_filtered = get_flows_for_query(flows_df, query_params)

    flow_uuids = flows_df_filtered["flow_uuid"].values.tolist()
    flows_col_index = flows_df_tapas.columns.get_loc("Amount")

    batch_gen = (
        (
            (i, min(len(flows_df), i + batch_size)),  # df start, stop
            flows_df.iloc[i : i + batch_size],  # df batch
        )
        for i in range(0, len(flows_df), batch_size)
    )

    for (start, stop), batch in batch_gen:
        batch = batch.reset_index(drop=True)
        batch_flows = batch.loc[batch["flow_uuid"].isin(flow_uuids)]
        batch_flows_rows = [int(x) for x in batch_flows.index.values.tolist()]
        coordinates = [(row, flows_col_index) for row in batch_flows_rows]
        answers = flows_df_tapas.iloc[start:stop]["Amount"].values[batch_flows_rows]

        dfs.append(
            pd.DataFrame(
                {
                    "batch": question["batch"],
                    "question_id": question["question_id"],
                    "question_template": question["question"],
                    "question_basic": question["question_replaced_basic"],
                    "question_general": question["question_replaced_general"],
                    "question_specific": question["question_replaced_specific"],
                    "process_uuid": question["process_uuid"],
                    "flows_start": start,
                    "flows_stop": stop,
                    "aggregation": query_params["aggregation"],
                    "coordinates": [coordinates],
                    "answers": [answers.tolist()],
                }
            )
        )

    return pd.concat(dfs, ignore_index=True).assign(
        # "list" should actually be "none"
        aggregation=lambda x: x["aggregation"].replace("list", "none").str.upper()
    )
