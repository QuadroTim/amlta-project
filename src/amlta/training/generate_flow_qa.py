import random
from typing import Literal, NamedTuple, cast

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from amlta.probas.extract import extract_process_flows
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
- Think about the domain of the process.
- What can be relevant?
- What values can be interesting to know, and make sense to ask about?

# BAD QUESTIONS
- What are the relevant outputs of the process?
- What are the top 5 outputs by mean amount?
- What are the top three input flows?

# Good Questions
- What is the output amount of <relevant flow>?
- What is input amount of energy from hydro power?
- What are the total toxic emissions to water?

-> In case no good question can be defined, simply ask for 1 specific flow amount.
</instructions>
""".strip()


# TODO: include description (`generalComment`) unless its the default useless one;
#   "Kurzinfo: Datensatz aus GEMIS. Negative Werte durch Gutschriftenrechnung..."
user_prompt = """
<process>
# Name
{process_name!r}

# Context
## Year
Start year of the time period for which the data set is valid (until year of "Data set valid until:"). For data sets that combine data from different
years, the most representative year is given regarding the overall environmental impact. In that case, the reference year is derived by expert
judgement.

=> {process_year}

## Geography
Location, country or region the data set represents. [Note 1: This field does not refer to e.g. the country in which a specific site is located that is
represented by this data set but to the actually represented country, region, or site. Note 2: Entry can be of type "two-letter ISO 3166 country code" for countries,
"seven-letter regional codes" for regions or continents, or "market areas and market organisations", as predefined for the ILCD. Also a name for e.g. a specific plant
etc. can be given here (e.g. "FR, Lyon, XY Company, Z Site"; user defined). Note 3: The fact whether the entry refers to production or to consumption / supply has to be
stated in the name-field "Mix and location types" e.g. as "Production mix".]

=> {process_geography}

## Classification
Hierarchical classification of the good, service, or process. (Note: This entry is NOT required for the identification of a Process. It should
nevertheless be avoided to use identical names for Processes in the same category.

=> {process_classification}

# Flows
Input/Output list of exchanges with the quantitative inventory data, as well as pre-calculated LCIA results.
- Amount: Final value to be used for calculation of the LCI results and in the product system: It is calculated as the product of the "Mean amount" value times the
    value of the "Variable". In case that no "Variable" entry is given, the "Resulting amount" is identical to the "Mean amount", i.e. a factor "1" is
    applied.

## Number of Flows
=> Input: {len_exchanges_input}
=> Output: {len_exchanges_output}

## Input Flows
{input_flows}

## Output Flows
{output_flows}
</process>

Given the provided LCI process data, what is a question an analyst could ask about the process flows?

Let's think step by step.
""".strip()


client = OpenAI()


class Flow(NamedTuple):
    exchange_direction: Literal["INPUT", "OUTPUT"]
    exchange_resulting_amount: float
    exchange_type_of_flow: str
    exchange_classification_hierarchy: str
    flow_description: str
    flow_property_name: str
    flow_property_unit: str


def flow_repr(flow: Flow) -> str:
    return '<"{name}" {direction}, {amount:g} {unit} ({property_name}), type={type_of_flow!r}, class={classification_hierarchy!r}>'.format(
        name=flow.flow_description,
        direction=flow.exchange_direction.lower(),
        type_of_flow=flow.exchange_type_of_flow,
        amount=flow.exchange_resulting_amount,
        classification_hierarchy=flow.exchange_classification_hierarchy,
        property_name=flow.flow_property_name,
        unit=flow.flow_property_unit,
    )


class LCIProcessQuery(BaseModel):
    """
    Possible queries that could have been used to find the current process.
    """

    specific: str = Field(
        description=(
            "A specific query for which it is likely that the current process "
            "is the top result or even the only result.\n"
            "Still, do not quote the process name verbatim, but make use of contextual information."
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


class FlowOutput(BaseModel):
    flow: FlowName | FlowClass | FlowType
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
            "that can be replaced be the generated process queries."
        )
    )
    flows: list[FlowOutput]
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


def generate_example(process: ProcessData):
    flows_df = (
        extract_process_flows(process)
        .sort_values(by="exchange_resulting_amount", ascending=False)
        .sort_values(by="exchange_classification_hierarchy")
    )
    process_name = cast(
        str, process.processInformation.dataSetInformation.name.baseName.get()
    )
    if process.processInformation.dataSetInformation.synonyms:
        synonyms = process.processInformation.dataSetInformation.synonyms.get()
        process_name += f" ({synonyms})"

    geography = process.processInformation.geography.locationOfOperationSupplyOrProduction.location
    class_info = process.processInformation.dataSetInformation.classificationInformation

    if class_info.classification:
        classes = []
        for classification in class_info.classification:
            classes.append(
                " - " + " / ".join(item.value for item in classification.class_)
            )
        classification = "\n".join(classes)
    else:
        classification = "Unknown"

    process_user_prompt = user_prompt.format(
        process_name=process_name,
        process_year=process.processInformation.time.referenceYear,
        process_geography=geography or "Unknown",
        process_classification=classification,
        input_flows="\n".join(
            flow_repr(cast(Flow, flow))
            for flow in flows_df.itertuples()
            if flow.exchange_direction == "INPUT"
        ),
        output_flows="\n".join(
            flow_repr(cast(Flow, flow))
            for flow in flows_df.itertuples()
            if flow.exchange_direction == "OUTPUT"
        ),
        len_exchanges_input=len(flows_df.loc[flows_df.exchange_direction == "INPUT"]),
        len_exchanges_output=len(flows_df.loc[flows_df.exchange_direction == "OUTPUT"]),
    )

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
        model="gpt-4o",
        temperature=0.6,
    )


def generate_random():
    uuids = read_uuids()
    uuid = random.choice(uuids)
    process = ProcessData.from_uuid(uuid)

    return generate_example(process)
