import random
from typing import Literal, NamedTuple, TypeAlias, cast

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

user_prompt = """
<process>
{process_description}
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
    flow_uuid: str
    flow_description: str
    flow_property_uuid: str
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


MarkdownValue: TypeAlias = "str | MarkdownData"
MarkdownData: TypeAlias = dict[str, "MarkdownValue | list[MarkdownValue]"]


def render_markdown(data: MarkdownData, level=0) -> str:
    _headings = [
        "#",
        "##",
        "###",
        "####",
        "#####",
        "######",
    ]

    def render_markdown_value(
        value: "MarkdownValue | list[MarkdownValue]", level=level
    ) -> str:
        if isinstance(value, str):
            return value
        elif isinstance(value, list):
            return "\n\n".join(
                render_markdown_value(item, level=level) for item in value
            )
        else:
            return "\n\n".join(
                f"{_headings[level]} {key}\n{render_markdown_value(val, level + 1)}"
                for key, val in value.items()
            )

    return render_markdown_value(data, level)


_default_general_comment = "Kurzinfo: Datensatz aus GEMIS. Negative Werte durch Gutschriftenrechnung. \n \nGEMIS steht f\u00fcr \u201cGlobales Emissions-Modell Integrierter Systeme\u201c; es ist ein Softwaretool des \u00d6ko-Instituts. GEMIS wurde 1987 erstmals angewendet und wird seitdem weiterentwickelt. \n \nDie GEMIS-Datens\u00e4tze beruhen - je nach Anwendung - auf unterschiedlichen Methoden; auch der zeitliche und der \u00f6rtliche Bezug der Datens\u00e4tze sind verschieden.\n \nMethode bei Prozessen mit mehreren Outputs:\n \nZur Modellierung der Datens\u00e4tze zu Multi-Output Prozessen wird in GEMIS die Methode der Systemerweiterung verwendet. Hierbei werden Datens\u00e4tze, in denen jeweils alle Inputs, alle Outputs und alle Umweltaspekte eines Multi-Output Prozesses ausgewiesen sind, als \u201cBrutto\u201c bezeichnet. Durch Subtraktion von \u201aBonus\u2019-Prozessen, die jeweils einen der Outputs auf herk\u00f6mmliche Weise bereitstellen, entsteht ein Nettoprozess, in denen das substituierte Nebenprodukt als Gutschrift erscheint. Die Gutschrift ist dabei kein realer Output des Prozesses, sondern ein rechnerischer \u201aMerker\u2019. \n \nBeispiel: \n \nMulti-Output Prozess Biogas-BZ-MC-HKW-D-2020/brutto: Output ist 1 TJ Elektrizit\u00e4t und 0,6 TJ W\u00e4rme, der \u201cNetto\u201c-Datensatz soll sich aber nur auf die Elektrizit\u00e4t beziehen. Durch Subtraktion des Bonusprozesses W\u00e4rme-Bonus-Gas-Hzg-D-2020 mit dem Output W\u00e4rme(0,6 TJ) entsteht der \u201cNetto\u201c-Datensatz Biogas-BZ-MC-HKW-D-2020/Gas, f\u00fcr den als Output 1 TJ Elektrizit\u00e4t und 0,6 TJ \u201aGutschrift W\u00e4rme-Bonus-f\u00fcr-KWK (Bio)-2020 bei W\u00e4rme-Bonus-Gas-Hzg-D-2020\u2019 angegeben werden; die Gutschrift stellt keinen Stoff- oder Energiefluss des Prozesses dar, sie ist allein rechnerisch begr\u00fcndet.\n \n\n \nTransport:\n \nAngaben zu den angesetzten Transportdistanzen werden nicht gegeben.\n \nAbschneidekriterien:\n \nWasser wird in der Regel nur auf der Inputseite angegeben (etwa als K\u00fchlwasser), auch wenn es den Prozess wieder verl\u00e4sst als Abwasser.\n Weitere Angaben zu angewendeten Abschneidekriterien werden nicht gegeben.\n \nBesondere Nomenklatur:\n \nZahlreiche Abk\u00fcrzungen f\u00fcr Brennstoffe aus Biomasse und entsprechende Technologien.\n \nBesonderheiten auf Datensatzebene:\n \nDie Datens\u00e4tze sind mit Vorketten-Datens\u00e4tzen verkn\u00fcpft, in denen die jeweils ben\u00f6tigten Vorprodukte, Energien und Transportleistungen erzeugt werden. Die Daten zu den Umweltaspekten werden erstens \u201cdirekt\u201c (d.h., nur aus dem jeweiligen Prozess, falls dieser direkt zu Umweltaspekten beitr\u00e4gt) als auch \u201cmit Vorkette\u201c (d.h., einschlie\u00dflich aller vorausgehenden Prozesse) ausgewiesen. \n Negative Werte f\u00fcr Stofffl\u00fcsse kommen in GEMIS regelm\u00e4\u00dfig vor; sie entstehen durch die Anwendung von Systemerweiterung um Multi-Output Prozesse in Single Output Prozesse umzurechnen. \n Teilweise werden Aufwendungen f\u00fcr Produktionsmittel (Anlagen, Fahrzeuge etc.) aufgef\u00fchrt (als Stofffl\u00fcsse im Input); diese sind jedoch nicht auf die funktionelle Einheit bezogen, sondern werden als absolute Werte angegeben; sie werden nur als Input und nicht als Output (Entsorgung der Betriebsmittel) angegeben. \n Die durch die Herstellung dieser Produktionsmittel verursachten Umweltaspekte sind dagegen \u00fcber Leistung, j\u00e4hrliche Auslastung und Lebensdauer auf die funktionelle Einheit bezogen \n \nWeiterf\u00fchrende Hinweise und Literatur:\n \n#1: Fritsche, U.R., Schmidt, K.: Globales Emissions-Modell Integrierter Systeme (GEMIS), Version 4.2, Handbuch, Darmstadt, August 2004.\n #2: Fritsche, U.R., Schmidt, K.: Globales Emissions-Modell Integrierter Systeme (GEMIS), Version 4.1, Handbuch, Darmstadt, Darmstadt, Januar 2003.\n #3: Fritsche, U., et al.: Stoffstromanalyse zur nachhaltigen energetischen Nutzung von Biomasse, Verbundprojekt gef\u00f6rdert vom BMU im Rahmen des ZIP, Projekttr\u00e4ger: FZ J\u00fclich, Mai 2004, Anhangband zum Endbericht.\n #4: Fritsche, U., et al.: Umweltanalyse von Energie-, Transport- und Stoffsystemen: Gesamt-Emissions-Modell integrierter Systeme (GEMIS) Version 2.1 - erweiterter und aktualisierter Endbericht, U. Fritsche u.a., i.A. des Hessischen Ministeriums f\u00fcr Umwelt, Energie und Bundesangelegenheiten (HMUEB), ver\u00f6ffentlicht durch HMUEB, Wiesbaden 1995"


def create_process_description(process: ProcessData) -> str:
    sections: MarkdownData = {}

    # Name
    # ====
    process_name = cast(
        str, process.processInformation.dataSetInformation.name.baseName.get()
    )
    if process.processInformation.dataSetInformation.synonyms:
        synonyms = process.processInformation.dataSetInformation.synonyms.get()
        process_name += f" ({synonyms})"

    sections["Name"] = process_name

    # Description
    # ===========
    if general_comment := process.processInformation.dataSetInformation.generalComment:
        general_comment = general_comment.get()
        if general_comment and _default_general_comment not in general_comment:
            sections["Description"] = general_comment

    # Year
    # ====
    year = process.processInformation.time.referenceYear
    until = process.processInformation.time.dataSetValidUntil

    time_desc = str(year)
    if until:
        time_desc += f" - {until}"

    sections["Year"] = [time_desc]
    if time_representativeness := (
        process.processInformation.time.timeRepresentativenessDescription
    ):
        time_representativeness = time_representativeness.get()
        assert time_representativeness is not None
        sections["Year"].append({"Representativeness": time_representativeness})

    # Geography
    # =========
    geography = process.processInformation.geography.locationOfOperationSupplyOrProduction.location
    sections["Geography"] = geography or "Unknown"

    # Classification
    # ==============
    class_info = process.processInformation.dataSetInformation.classificationInformation
    if class_info.classification:
        classes = []
        for classification in class_info.classification:
            classes.append(" / ".join(item.value for item in classification.class_))
        classification = "\n".join(classes)
    else:
        classification = "Unknown"

    sections["Class"] = classification

    # Technology
    # ==========
    technology = process.processInformation.technology
    technology_description = technology.technologyDescriptionAndIncludedProcesses
    technology_applicability = technology.technologicalApplicability

    sections["Technology"] = []

    if technology_description:
        technology_description = technology_description.get()
        assert technology_description is not None
        sections["Technology"].append(technology_description)

    if technology_applicability:
        technology_applicability = technology_applicability.get()
        assert technology_applicability is not None
        sections["Technology"].append({"Applicability": technology_applicability})

    if not sections["Technology"]:
        sections.pop("Technology")

    # (flows)
    flows_df = (
        extract_process_flows(process)
        .sort_values(by="exchange_resulting_amount", ascending=False)
        .sort_values(by="exchange_classification_hierarchy")
    )

    # Main Output
    # ===========
    if quantitative_reference := process.processInformation.quantitativeReference:
        if quantitative_reference.functionalUnitOrOther:
            functional_unit = quantitative_reference.functionalUnitOrOther.get()
        else:
            functional_unit = None

        output_flow_ids = quantitative_reference.referenceToReferenceFlow
        assert len(output_flow_ids) == 1
        output_flow_id = output_flow_ids[0]
        output_flow_uuid = next(
            flow.referenceToFlowDataSet.refObjectId
            for flow in process.exchanges.exchange
            if flow.dataSetInternalID == output_flow_id
        )
        output_flow = Flow(
            **flows_df.loc[flows_df["flow_uuid"] == output_flow_uuid].iloc[0].to_dict()
        )

        sections["Main Output"] = []
        if functional_unit:
            sections["Main Output"] = [functional_unit]

        sections["Main Output"].append({"Main Output Flow": flow_repr(output_flow)})

    # Flows
    # =====
    n_input_flows = len(flows_df.loc[flows_df.exchange_direction == "INPUT"])
    n_output_flows = len(flows_df.loc[flows_df.exchange_direction == "OUTPUT"])

    input_flows_list = "\n".join(
        flow_repr(cast(Flow, flow))
        for flow in flows_df.itertuples()
        if flow.exchange_direction == "INPUT"
    )
    output_flows_list = "\n".join(
        flow_repr(cast(Flow, flow))
        for flow in flows_df.itertuples()
        if flow.exchange_direction == "OUTPUT"
    )
    sections["Flows"] = {
        "Inputs": str(n_input_flows),
        "Outputs": str(n_output_flows),
        "Input Flows": input_flows_list,
        "Output Flows": output_flows_list,
    }

    return render_markdown(sections)


def generate_example(process: ProcessData):
    process_description = create_process_description(process)
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
        model="gpt-4o",
        temperature=0.6,
    )


def generate_random():
    uuids = read_uuids()
    uuid = random.choice(uuids)
    process = ProcessData.from_uuid(uuid)

    return generate_example(process)
