import json
import random
import time
from os import PathLike
from pathlib import Path
from typing import Literal, NamedTuple, Sequence

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from openai.lib._parsing import type_to_response_format_param
from pydantic import BaseModel, Field

from amlta.config import config
from amlta.formatting.data import create_process_section
from amlta.formatting.yaml import format_as_yaml
from amlta.probas.flows import extract_process_flows
from amlta.probas.processes import ProcessData, read_uuids

load_dotenv()


system_prompt = """
<instructions>
You are a helpful assistant and Life Cycle Inventory (LCI) expert.

You will be provided a Life Cycle Inventory (LCI) dataset process entry.

# Your Task
- Pretend a LCI analyst has queried for the process you are given.
- The analyst wants to retrieve information about the process's inputs/outputs ("exchanges") and their values.
- Take special attention to the process metadata like region and year and context.
- Find a realistic question the LCI analyst could have asked about the process flows.
- If the question entails multiple flows, be reminded to combine all relevant flows when compiling the list.
- The expected output should be quantifiable, i.e., be able to be retrieved objectively and disambiguously.

## Note
- While the data may be german or multilingual you must use only english however.

# Output
- Provide the question including context about the process.
- For the context please use a simplified paraphrased name for the process. I.e., the search query the analyst could have entered querying the database.
- List the flows the analyst would retrieve as an answer to the question.

Think step by step;
1. Think about the domain of the process.
2. What can be relevant?
3. What values can be interesting to know, and make sense to ask about?
4. What are the relevant flows?
5. Is the question about a group of flows?
5.1 Can the entire group be summarized using either the class level or the type of flow?
5.2 Use the relevant FlowQuery accordingly.
6. Output the question in the correct format and according to the descriptions.

# BAD QUESTIONS
- What are the relevant outputs of the process?
- What are the top 5 outputs by mean amount?
- What are the top three input flows?

# Good Questions
- What is the output amount of methane of <the process>?
- What is input amount of energy from hydro power of <the process>?
- What are the total toxic emissions to water of <the process>?

-> In case no good question can be defined, simply ask for 1 specific flow amount.
</instructions>
""".strip()

user_prompt = """
<process>
{process_description}
</process>

Given the provided LCI process data, what is a question an analyst could ask about the process flows?

Let's think step by step.
""".strip()


class LCIProcessQuery(BaseModel):
    """
    Possible queries that could have been used to find the current process.
    """

    specific: str = Field(
        description=(
            "A specific query for which it is likely that the current process "
            "is the top result or even the only result.\n"
            "Still, do not quote the process name verbatim."
        )
    )
    general: str = Field(
        description=(
            "A general query for which the current process is very relevant. But one can "
            "imagine other processes being just as relevant.\n"
            "This can e.g. be the general process paraphrased, like 'iron production'."
        )
    )


class FlowName(BaseModel):
    """
    Use if flows are best idenified by their name.
    """

    flow_name: str


class FlowClass(BaseModel):
    """
    Use if multiple flows ought to be extracted that all belong to a specific class (level).
    Cut off the class hierarchy at the relevant level.
    """

    flow_class: str


class FlowType(BaseModel):
    """
    Use if all flows of a specific type are to be extracted.
    """

    flow_type: str


class FlowQuery(BaseModel):
    flow: FlowName | FlowClass | FlowType = Field(
        description=(
            "The flow to extract. If the question targets multiple flows through their "
            "class level or their type, use the respective model.\n"
            "Examples:\n"
            "FlowName: What are the emissions for carbon dioxide and sulfur dioxide?\n"
            "FlowClass: What are the total emissions emissions to air?\n"
            "FlowType: What is the average waste flow?"
        )
    )
    direction: Literal["input", "output", "both"]


class LCIQuestion(BaseModel):
    """Respond with this"""

    thoughts: list[str]
    process_query: LCIProcessQuery
    question: str = Field(
        description=(
            "The question an analyst could ask about the process flows. "
            "The question should be process-agnostic; it must not include "
            "the process name or description, but use a placeholder '<the process>' "
            "that can be replaced be the generated process queries.\n"
            "Example: 'What are the total emissions emissions to air of <the process>?'"
        )
    )
    flows: list[FlowQuery]
    aggregation: Literal["none", "count", "sum", "average"] = Field(
        description=(
            "How to aggregate the flows. If 'none', the flows are listed individually. "
            "If 'count', the number of flows is returned. If 'sum', the sum of the "
            "flow amounts is returned. If 'average', the average of the flow amounts "
            "is returned.\n"
            "Remain aware that aggregation only makes sense to use if the flows are of the same "
            "type and unit."
        )
    )


class ProcessLCIQuestion(NamedTuple):
    process_uuid: str
    question: LCIQuestion


def process_responses(
    responses: Sequence[ProcessLCIQuestion],
    **kwargs,
) -> pd.DataFrame:
    dfs = []

    for response in responses:
        data = []

        process = ProcessData.from_uuid(response.process_uuid)
        process_flows = extract_process_flows(process)

        question = response.question
        specific_query = question.process_query.specific
        general_query = question.process_query.general
        question_text = question.question
        question_replaced_specific = question_text.replace(
            "<the process>", specific_query
        )
        question_replaced_general = question_text.replace(
            "<the process>", general_query
        )
        desired_flows = question.flows
        aggregation = question.aggregation

        for flow_output in desired_flows:
            direction = flow_output.direction.upper()
            if direction != "BOTH":
                process_flows = process_flows.loc[
                    process_flows["exchange_direction"] == direction
                ]

            if isinstance(flow_output.flow, FlowName):
                flow_name = flow_output.flow.flow_name
                process_flows = process_flows.loc[
                    process_flows["flow_description"] == flow_name
                ]
            elif isinstance(flow_output.flow, FlowClass):
                flow_class = flow_output.flow.flow_class
                process_flows = process_flows.loc[
                    process_flows["exchange_classification_hierarchy"]
                    .fillna("")
                    .str.contains(flow_class, case=False)
                ]
            elif isinstance(flow_output.flow, FlowType):
                flow_type = flow_output.flow.flow_type
                process_flows = process_flows.loc[
                    process_flows["exchange_type_of_flow"].str.lower()
                    == flow_type.lower()
                ]
            else:
                raise ValueError("Invalid flow type")

            for _, flow in process_flows.iterrows():
                data.append(
                    {
                        **kwargs,
                        "process_uuid": response.process_uuid,
                        "general_query": general_query,
                        "specific_query": specific_query,
                        "question": question_text,
                        "question_replaced_general": question_replaced_general,
                        "question_replaced_specific": question_replaced_specific,
                        "question_direction": direction,
                        "aggregation": aggregation,
                        "exchange_direction": flow["exchange_direction"],
                        "exchange_resulting_amount": flow["exchange_resulting_amount"],
                        "exchange_type_of_flow": flow["exchange_type_of_flow"],
                        "exchange_classification_hierarchy": flow[
                            "exchange_classification_hierarchy"
                        ],
                        "flow_uuid": flow["flow_uuid"],
                        "flow_description": flow["flow_description"],
                        "flow_property_uuid": flow["flow_property_uuid"],
                        "flow_property_name": flow["flow_property_name"],
                        "flow_property_unit": flow["flow_property_unit"],
                    }
                )

        dfs.append(
            pd.DataFrame(data).drop_duplicates(
                subset=[
                    "process_uuid",
                    "flow_uuid",
                    "aggregation",
                    "exchange_direction",
                ]
            )
        )

    return pd.concat(dfs)


client = OpenAI()


def generate_example(process: ProcessData, model: str = "gpt-4o"):
    process_description_data = create_process_section(process)
    process_description = format_as_yaml(process_description_data)
    process_user_prompt = user_prompt.format(process_description=process_description)

    return client.beta.chat.completions.parse(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": process_user_prompt,
            },
        ],
        response_format=LCIQuestion,
        model=model,
        temperature=0.6,
    )


def generate_random(model: str = "gpt-4o"):
    uuids = read_uuids()
    uuid = random.choice(uuids)
    process = ProcessData.from_uuid(uuid)

    return generate_example(process, model=model)


def prepare_batch(
    n: int,
    offset: int = 0,
    model: str = "gpt-4o",
    eval_dir: PathLike | None = None,
):
    if eval_dir is None:
        eval_dir = config.project_dir / "eval"
    else:
        eval_dir = Path(eval_dir)

    out_dir = eval_dir / "process_flows_questions" / "batch_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    rnd = random.Random(42)
    uuids = read_uuids()
    rnd.shuffle(uuids)
    stop = offset + n
    uuids = uuids[offset:stop]

    tasks = []

    for uuid in uuids:
        process = ProcessData.from_uuid(uuid)
        process_description_data = create_process_section(process)
        process_description = format_as_yaml(process_description_data)
        process_user_prompt = user_prompt.format(
            process_description=process_description
        )

        tasks.append(
            {
                "custom_id": uuid,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": process_user_prompt,
                        },
                    ],
                    "response_format": type_to_response_format_param(LCIQuestion),
                    "model": model,
                    "temperature": 0.6,
                },
            }
        )

    out_file = out_dir / f"batch_{model}_{offset}_{stop}.jsonl"
    out_file.write_text("\n".join(json.dumps(task) for task in tasks))

    return out_file


def send_batch(batch_file: PathLike):
    job_file = client.files.create(file=batch_file, purpose="batch")

    while (job_file := client.files.retrieve(job_file.id)).status == "uploaded":
        time.sleep(1)

    if job_file.status == "error":
        raise RuntimeError(f"Batch upload failed: {job_file.status_details}")

    batch = client.batches.create(
        completion_window="24h",
        endpoint="/v1/chat/completions",
        input_file_id=job_file.id,
    )

    return batch


def retrieve_batch_results(batch_id: str, eval_dir: PathLike | None = None):
    if eval_dir is None:
        eval_dir = config.project_dir / "eval"
    else:
        eval_dir = Path(eval_dir)

    out_dir = eval_dir / "process_flows_questions" / "batch_outputs"

    batch = client.batches.retrieve(batch_id)

    if batch.status != "completed":
        if batch.errors:
            raise RuntimeError(f"Batch not completed: {batch.errors}")

        print(f"Batch status: {batch.status}")
        print(batch)
        return

    input_file = client.files.retrieve(batch.input_file_id)

    assert batch.output_file_id is not None
    result = client.files.content(batch.output_file_id)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / input_file.filename
    out_file.write_bytes(result.content)

    return out_file


def process_batch_results(batch_output_file: PathLike):
    batch_output_file = Path(batch_output_file)
    data = [
        json.loads(line)
        for line in Path(batch_output_file).read_text().strip().splitlines()
    ]

    responses = []

    for item in data:
        # batch_req_id = item["id"]
        process_uuid = item["custom_id"]
        response_raw_json = item["response"]["body"]["choices"][0]["message"]["content"]
        response = LCIQuestion.model_validate_json(response_raw_json)
        responses.append(
            ProcessLCIQuestion(process_uuid=process_uuid, question=response)
        )

    return process_responses(responses, batch=batch_output_file.stem)
