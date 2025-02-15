from typing import Sequence, TypeAlias, cast

import pandas as pd

from amlta.formatting.flows import flow_repr
from amlta.probas import flows
from amlta.probas.processes import ProcessData

SectionValue: TypeAlias = "str | int | float | SectionData | Sequence[SectionValue]"
SectionData: TypeAlias = dict[str, SectionValue]


_default_general_comment = "Kurzinfo: Datensatz aus GEMIS. Negative Werte durch Gutschriftenrechnung. \n \nGEMIS steht f\u00fcr \u201cGlobales Emissions-Modell Integrierter Systeme\u201c; es ist ein Softwaretool des \u00d6ko-Instituts. GEMIS wurde 1987 erstmals angewendet und wird seitdem weiterentwickelt. \n \nDie GEMIS-Datens\u00e4tze beruhen - je nach Anwendung - auf unterschiedlichen Methoden; auch der zeitliche und der \u00f6rtliche Bezug der Datens\u00e4tze sind verschieden.\n \nMethode bei Prozessen mit mehreren Outputs:\n \nZur Modellierung der Datens\u00e4tze zu Multi-Output Prozessen wird in GEMIS die Methode der Systemerweiterung verwendet. Hierbei werden Datens\u00e4tze, in denen jeweils alle Inputs, alle Outputs und alle Umweltaspekte eines Multi-Output Prozesses ausgewiesen sind, als \u201cBrutto\u201c bezeichnet. Durch Subtraktion von \u201aBonus\u2019-Prozessen, die jeweils einen der Outputs auf herk\u00f6mmliche Weise bereitstellen, entsteht ein Nettoprozess, in denen das substituierte Nebenprodukt als Gutschrift erscheint. Die Gutschrift ist dabei kein realer Output des Prozesses, sondern ein rechnerischer \u201aMerker\u2019. \n \nBeispiel: \n \nMulti-Output Prozess Biogas-BZ-MC-HKW-D-2020/brutto: Output ist 1 TJ Elektrizit\u00e4t und 0,6 TJ W\u00e4rme, der \u201cNetto\u201c-Datensatz soll sich aber nur auf die Elektrizit\u00e4t beziehen. Durch Subtraktion des Bonusprozesses W\u00e4rme-Bonus-Gas-Hzg-D-2020 mit dem Output W\u00e4rme(0,6 TJ) entsteht der \u201cNetto\u201c-Datensatz Biogas-BZ-MC-HKW-D-2020/Gas, f\u00fcr den als Output 1 TJ Elektrizit\u00e4t und 0,6 TJ \u201aGutschrift W\u00e4rme-Bonus-f\u00fcr-KWK (Bio)-2020 bei W\u00e4rme-Bonus-Gas-Hzg-D-2020\u2019 angegeben werden; die Gutschrift stellt keinen Stoff- oder Energiefluss des Prozesses dar, sie ist allein rechnerisch begr\u00fcndet.\n \n\n \nTransport:\n \nAngaben zu den angesetzten Transportdistanzen werden nicht gegeben.\n \nAbschneidekriterien:\n \nWasser wird in der Regel nur auf der Inputseite angegeben (etwa als K\u00fchlwasser), auch wenn es den Prozess wieder verl\u00e4sst als Abwasser.\n Weitere Angaben zu angewendeten Abschneidekriterien werden nicht gegeben.\n \nBesondere Nomenklatur:\n \nZahlreiche Abk\u00fcrzungen f\u00fcr Brennstoffe aus Biomasse und entsprechende Technologien.\n \nBesonderheiten auf Datensatzebene:\n \nDie Datens\u00e4tze sind mit Vorketten-Datens\u00e4tzen verkn\u00fcpft, in denen die jeweils ben\u00f6tigten Vorprodukte, Energien und Transportleistungen erzeugt werden. Die Daten zu den Umweltaspekten werden erstens \u201cdirekt\u201c (d.h., nur aus dem jeweiligen Prozess, falls dieser direkt zu Umweltaspekten beitr\u00e4gt) als auch \u201cmit Vorkette\u201c (d.h., einschlie\u00dflich aller vorausgehenden Prozesse) ausgewiesen. \n Negative Werte f\u00fcr Stofffl\u00fcsse kommen in GEMIS regelm\u00e4\u00dfig vor; sie entstehen durch die Anwendung von Systemerweiterung um Multi-Output Prozesse in Single Output Prozesse umzurechnen. \n Teilweise werden Aufwendungen f\u00fcr Produktionsmittel (Anlagen, Fahrzeuge etc.) aufgef\u00fchrt (als Stofffl\u00fcsse im Input); diese sind jedoch nicht auf die funktionelle Einheit bezogen, sondern werden als absolute Werte angegeben; sie werden nur als Input und nicht als Output (Entsorgung der Betriebsmittel) angegeben. \n Die durch die Herstellung dieser Produktionsmittel verursachten Umweltaspekte sind dagegen \u00fcber Leistung, j\u00e4hrliche Auslastung und Lebensdauer auf die funktionelle Einheit bezogen \n \nWeiterf\u00fchrende Hinweise und Literatur:\n \n#1: Fritsche, U.R., Schmidt, K.: Globales Emissions-Modell Integrierter Systeme (GEMIS), Version 4.2, Handbuch, Darmstadt, August 2004.\n #2: Fritsche, U.R., Schmidt, K.: Globales Emissions-Modell Integrierter Systeme (GEMIS), Version 4.1, Handbuch, Darmstadt, Darmstadt, Januar 2003.\n #3: Fritsche, U., et al.: Stoffstromanalyse zur nachhaltigen energetischen Nutzung von Biomasse, Verbundprojekt gef\u00f6rdert vom BMU im Rahmen des ZIP, Projekttr\u00e4ger: FZ J\u00fclich, Mai 2004, Anhangband zum Endbericht.\n #4: Fritsche, U., et al.: Umweltanalyse von Energie-, Transport- und Stoffsystemen: Gesamt-Emissions-Modell integrierter Systeme (GEMIS) Version 2.1 - erweiterter und aktualisierter Endbericht, U. Fritsche u.a., i.A. des Hessischen Ministeriums f\u00fcr Umwelt, Energie und Bundesangelegenheiten (HMUEB), ver\u00f6ffentlicht durch HMUEB, Wiesbaden 1995"


def create_flows_section(flows_df: pd.DataFrame) -> SectionData:
    n_input_flows = len(flows_df.loc[flows_df.exchange_direction == "INPUT"])
    n_output_flows = len(flows_df.loc[flows_df.exchange_direction == "OUTPUT"])

    input_flows_list = [
        flow_repr(cast(flows.Flow, flow))
        for flow in flows_df.itertuples()
        if flow.exchange_direction == "INPUT"
    ]
    output_flows_list = [
        flow_repr(cast(flows.Flow, flow))
        for flow in flows_df.itertuples()
        if flow.exchange_direction == "OUTPUT"
    ]

    return {
        "Flows": {
            "Inputs": n_input_flows,
            "Outputs": n_output_flows,
            "Input Flows": input_flows_list,
            "Output Flows": output_flows_list,
        }
    }


def create_process_section(
    process: ProcessData,
    include_flows: bool = True,
) -> SectionData:
    sections: SectionData = {}

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

    sections["Year"] = []
    sections["Year"].append(time_desc)
    if time_representativeness := (
        process.processInformation.time.timeRepresentativenessDescription
    ):
        if time_representativeness := time_representativeness.get():
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
        classification = classes
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
        if technology_description := technology_description.get():
            sections["Technology"].append(technology_description)

    if technology_applicability:
        if technology_applicability := technology_applicability.get():
            sections["Technology"].append({"Applicability": technology_applicability})

    if not sections["Technology"]:
        sections.pop("Technology")

    # (flows)
    flows_df = (
        flows.extract_process_flows(process)
        .sort_values(by="exchange_resulting_amount", ascending=False)
        .sort_values(by="exchange_classification_hierarchy")
    )

    # Main Output
    # ===========
    if main_exchange := process.get_main_output():
        quantitative_reference = process.processInformation.quantitativeReference

        assert quantitative_reference is not None
        if quantitative_reference.functionalUnitOrOther:
            functional_unit = quantitative_reference.functionalUnitOrOther.get()
        else:
            functional_unit = None

        output_flow_uuid = main_exchange.referenceToFlowDataSet.refObjectId
        output_flow = flows.Flow(
            **flows_df.loc[flows_df["flow_uuid"] == output_flow_uuid].iloc[0].to_dict()
        )

        sections["Main Output"] = []
        if functional_unit:
            sections["Main Output"].append(functional_unit)

        sections["Main Output"].append({"Main Output Flow": flow_repr(output_flow)})

    # Flows
    # =====
    if include_flows:
        sections |= create_flows_section(flows_df)

    return sections
