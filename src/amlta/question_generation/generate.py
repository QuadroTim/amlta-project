import json
import random
import time
from os import PathLike
from pathlib import Path
from typing import (
    ClassVar,
)

from openai import OpenAI
from openai.lib._parsing import type_to_response_format_param
from openai.types import Batch
from openai.types.chat import ParsedChatCompletion
from pydantic import BaseModel, ConfigDict, Field

from amlta.config import config
from amlta.formatting.data import create_flows_section, create_process_section
from amlta.formatting.yaml import format_as_yaml
from amlta.probas.processes import ProcessData, read_uuids
from amlta.question_generation.query_params import generate_random_query_params

system_prompt = """
<instructions>
You are a helpful assistant and Life Cycle Inventory (LCI) expert.

You will be provided a Life Cycle Inventory (LCI) dataset process entry and the query params
for a question which you will generate.

# Your Task
- Pretend a LCI analyst has queried for the process you are given.
- The analyst wants to retrieve information about the process's inputs/outputs ("flows") and their values.
- Take special attention to the process metadata like region and year and context.
- Find a realistic question the LCI analyst could have asked about the process flows.
- The output should be quantifiable, i.e., be able to be retrieved objectively and disambiguously.
- You may use a different phrasing for the asked flows, classes, flow types, etc., than in the params.
    The meaning must be the same, however!

NOTE: While the data may be German or multilingual you however must use only English.

# Output
1. Provide the question where the process name (e.g., the process query you generated) is replaced
    by '<the process>'.
2. Provide three possible search queries for the process itself that can replace '<the process>'.

Think step by step;
**Flow Question**
1. Think about the domain of the process in combination with the query params.
2. What are possible words an analyst would use that match the meaning of the query params?
3. Phrase the question ACCORDING TO THE QUERY TYPE ('name', 'names', 'class', 'type') provided as
    well as the direction and aggregation. Unless the direction is 'both', make sure to hint at
    the direction in the question. The same applies for any other `flow_[...]` field in the params
    as well as the aggregation.

    Considering the different query types, please note:
    a) For 'name', the question should include the flow name, which can be rephrased to make a
        realistic query (e.g., 'CO2' instead of 'carbon dioxide').
    b) Analogous for 'names', however, ALL listed flows must be named in the question.
    c) For 'class', the question should include the flow class, which can be rephrased to make a
        realistic query (e.g., 'airborne emissions' instead of 'emissions to air').
        Moreover, pay attention to all class levels (separated by ' / ').
    d) For 'type', the question should include the flow type, which can be rephrased also.

**Process Search**
1. Think about the domain of the process.
2. In which context would the process be relevant? Think about different target users;
    e.g., researchers, manufacturers, policy makers.
3. What can be the keywords an analyst would use to find the process?
4. Use the process metadata and context to phrase searches with different levels of specificity.
5. The generated queries should be able to replace '<the process>' in the generated question.

# BAD QUESTIONS
- What are the relevant outputs of the process?
- What are the top 5 outputs by mean amount?
- What are the top three input flows?

# Good Questions
- What is the output amount of methane of <the process>?
- What is input amount of energy from hydro power of <the process>?
- What are the total toxic emissions to water of <the process>?
</instructions>
""".strip()

user_prompt = """
<process>
{process_description}
</process>

<query_params>
{query_params}
</query_params>

Given the provided LCI process data and query context, what is a question an analyst could ask
about the process flows that match the query params?

Let's think step by step.
""".strip()


class LCIProcessKeywords(BaseModel):
    """
    Possible query that could have been used to find the current process.
    This must be only about the process itself, not the flows.

    DO NOT include any query about the flows here, ONLY the process.

    Example: 'iron production in germany'
    """

    thoughts: list[str]

    specific_search: str = Field(
        description=(
            "A specific query for which it is likely that the current process "
            "is the top result or even the only result.\n"
            "Still, do not quote the process name verbatim."
        )
    )
    general_search: str = Field(
        description=(
            "A general query for which the current process is very relevant. But one can "
            "imagine other processes being just as relevant.\n"
            "This can e.g. be the general process paraphrased, like 'iron production'."
        )
    )

    basic_search: str = Field(
        description=(
            "A basic query that could be used to find the process in the same domain. "
            "This is a very general query that could be used to find a wide range of processes."
        )
    )


class LCIFlowQuestion(BaseModel):
    thoughts: list[str] = Field(
        description=(
            "Include at least (maybe more) all previously enumerated steps; 1., 2., 3., 4."
        )
    )

    question: str = Field(
        description=(
            "The question an analyst could ask about the process flows. "
            "The question should be process-agnostic; it must not include "
            "the process name or description, but use a placeholder '<the process>' "
            "that can be replaced be the generated process queries.\n"
            "Example: 'What are the total cadmium emissions of <the process>?'\n"
        )
    )


class LCIQuestion(BaseModel):
    model_config: ClassVar[ConfigDict] = {"title": "LCI Question"}

    flow_question: LCIFlowQuestion

    process_keywords: LCIProcessKeywords = Field(
        description=(
            "IGNORING THE FLOWS, what could have been the query to find this process? "
            "I.e., paraphrase the process name and context with different levels of specificity. "
            "It should be able to replace a placeholder '<the process>'."
        )
    )


def get_openai_client():
    from dotenv import load_dotenv

    load_dotenv()
    return OpenAI()


def generate_question(
    process: ProcessData, model: str = "gpt-4o"
) -> ParsedChatCompletion[LCIQuestion]:
    client = get_openai_client()

    process_description_data = create_process_section(process, include_flows=False)
    result_flows, query_params = generate_random_query_params(process)

    process_description = format_as_yaml(process_description_data)
    process_user_prompt = user_prompt.format(
        process_description=process_description,
        query_params=format_as_yaml(query_params, line_between_sections=False),
        # flows_result=format_as_yaml(create_flows_section(result_flows)),
    )

    print(process_user_prompt)

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
        temperature=0.9,
        top_p=0.9,
    )


def generate_random_process_question(
    model: str = "gpt-4o",
) -> ParsedChatCompletion[LCIQuestion]:
    uuids = read_uuids()
    uuid = random.choice(uuids)
    process = ProcessData.from_uuid(uuid)

    return generate_question(process, model=model)


def get_generated_questions_path():
    return config.generated_dir / "questions"


def prepare_batch(
    n: int,
    offset: int = 0,
    questions_per_process: int = 1,
    model: str = "gpt-4o",
) -> Path:
    qa_gen_dir = get_generated_questions_path()

    out_dir = qa_gen_dir / "batch_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)

    uuids = read_uuids()
    random.shuffle(uuids)
    stop = offset + n
    uuids = uuids[offset:stop]

    tasks = []
    payload = []

    for uuid in uuids:
        for i in range(questions_per_process):
            process = ProcessData.from_uuid(uuid)
            process_description_data = create_process_section(
                process, include_flows=False
            )
            result_flows, query_params = generate_random_query_params(process)

            process_description = format_as_yaml(process_description_data)
            process_user_prompt = user_prompt.format(
                process_description=process_description,
                query_params=format_as_yaml(query_params, line_between_sections=False),
                flows_result=format_as_yaml(create_flows_section(result_flows)),
            )

            task_id = f"{uuid}/{i}"

            tasks.append(
                {
                    "custom_id": task_id,
                    "process_uuid": uuid,
                    "question_id": i,
                    "query_params": query_params,
                }
            )

            payload.append(
                {
                    "custom_id": task_id,
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

    batch_out_file = out_dir / f"batch_{model}_{offset}_{stop}.jsonl"
    batch_out_file.write_text(
        "\n".join(json.dumps(task_payload) for task_payload in payload)
    )

    batch_input_file = batch_out_file.with_name(f"{batch_out_file.stem}_input.jsonl")
    batch_input_file.write_text("\n".join(json.dumps(task) for task in tasks))

    return batch_out_file


def send_batch(batch_file: PathLike) -> Batch:
    client = get_openai_client()

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


def retrieve_batch_results(
    batch_id: str, eval_dir: PathLike | None = None
) -> Path | None:
    client = get_openai_client()

    qa_gen_dir = get_generated_questions_path()

    out_dir = qa_gen_dir / "batch_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    batch = client.batches.retrieve(batch_id)
    print(batch)

    if batch.status not in {"completed", "expired"}:
        if batch.errors:
            raise RuntimeError(f"Batch not completed: {batch.errors}")

        print(f"Batch status: {batch.status}")
        return

    input_file = client.files.retrieve(batch.input_file_id)

    assert batch.output_file_id is not None
    result = client.files.content(batch.output_file_id)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / input_file.filename
    out_file.write_bytes(result.content)

    return out_file
