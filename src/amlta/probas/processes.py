from collections import UserList
from os import PathLike
from typing import (
    Any,
    ClassVar,
    Generic,
    Iterator,
    List,
    Literal,
    Self,
    TypeVar,
)

from pydantic import BaseModel as _BaseModel
from pydantic import ConfigDict, Field, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from amlta.config import config


class BaseModel(_BaseModel):
    model_config: ClassVar[ConfigDict] = {
        "use_attribute_docstrings": True,
        "extra": "forbid",
    }


# --------------------------
# Shared / Repeated Models
# --------------------------


class LocalizedText(BaseModel):
    value: str | None = None
    lang: str | None = None


class LocalizedTextList(UserList[LocalizedText]):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls, handler(List[LocalizedText])
        )

    def get(self, preferred_lang: str = "en") -> str | None:
        for text in self:
            if text.lang == preferred_lang and text.value:
                return text.value

        return next((text.value for text in self if text.value), None)


class AccessRestriction(BaseModel):
    value: str
    lang: str


class Other(BaseModel):
    anies: List[str]


ExternalDataSetTypeT = Literal[
    "source data set",
    "process data set",
    "flow data set",
    "flow property data set",
    "unit group data set",
    "contact data set",
    "LCIA method data set",
    "other external file",
]
TypeT = TypeVar("TypeT", bound=ExternalDataSetTypeT)


class ExternalReference(BaseModel, Generic[TypeT]):
    shortDescription: LocalizedTextList
    type: TypeT
    refObjectId: str
    uri: str | None = None
    version: str | None = None


SourceDataSet = ExternalReference[Literal["source data set"]]
ProcessDataSet = ExternalReference[Literal["process data set"]]
FlowDataSet = ExternalReference[Literal["flow data set"]]
FlowPropertyDataSet = ExternalReference[Literal["flow property data set"]]
UnitGroupDataSet = ExternalReference[Literal["unit group data set"]]
ContactDataSet = ExternalReference[Literal["contact data set"]]
LCIAMethodDataSet = ExternalReference[Literal["LCIA method data set"]]
OtherExternalFile = ExternalReference[Literal["other external file"]]


# --------------------------
# processInformation section
# --------------------------


class Name(BaseModel):
    baseName: LocalizedTextList
    functionalUnitFlowProperties: LocalizedTextList | None = None
    """
    Further, quantitative specifying information on the good, service or process in technical term(s): qualifying constituent(s)-content and / or
    energy-content per unit etc. as appropriate. Separated by commata. (Note: non-qualifying flow properties, CAS No, Synonyms, Chemical formulas etc. are documented
    exclusively in the "Flow data set".)
    """


class ClassificationItem(BaseModel):
    value: str
    level: int
    classId: str


class Classification(BaseModel):
    class_: List[ClassificationItem] = Field(..., alias="class")
    name: str
    classes: str


class ClassificationInformation(BaseModel):
    classification: List[Classification] | None = None


class DataSetInformation(BaseModel):
    UUID: str
    """
    Automatically generated Universally Unique Identifier of this data set. Together with the "Data set version", the UUID uniquely identifies each data
    set.
    """
    name: Name
    """
    General descriptive and specifying name of the process.
    """
    synonyms: LocalizedTextList | None = None
    """
    Synonyms / alternative names / brands of the good, service, or process. Separated by semicolon.
    """
    classificationInformation: ClassificationInformation
    """
    Hierarchical classification of the good, service, or process. (Note: This entry is NOT required for the identification of a Process. It should
    nevertheless be avoided to use identical names for Processes in the same category.
    """
    generalComment: LocalizedTextList
    """
    General information about the data set, including e.g. general (internal, not reviewed) quality statements as well as information sources used. (Note:
    Please also check the more specific fields e.g. on "Intended application", "Advice on data set use" and the fields in the "Modelling and validation" section to avoid
    overlapping entries.)
    """
    referenceToExternalDocumentation: List[SourceDataSet] | None = None
    """
    "Source data set(s)" of detailed LCA study on the process or product represented by this data set, as well as documents / files with overarching
    documentative information on technology, geographical and / or time aspects etc. (e.g. basic engineering studies, process simulation results, patents, plant
    documentation, model behind the parameterisation of the "Mathematical model" section, etc.) (Note: can indirectly reference to digital file.)
    """
    other: Other | None = None


class QuantitativeReference(BaseModel):
    referenceToReferenceFlow: List[int]
    """
    One or more of the Inputs or Outputs in case "Type of quantitative reference" is of type "Reference flow(s)". (Data set internal
    reference.)
    """
    functionalUnitOrOther: LocalizedTextList | None = None
    """
    Quantity, name, property/quality, and measurement unit of the Functional unit, Production period, or Other parameter, in case "Type of quantitative
    reference" is of one of these types. [Note: One or more functional units can also be given in addition to a reference flow.]
    """
    type: str
    """
    Type of quantitative reference of this data set.
    """


class Time(BaseModel):
    referenceYear: int
    """
    Start year of the time period for which the data set is valid (until year of "Data set valid until:"). For data sets that combine data from different
    years, the most representative year is given regarding the overall environmental impact. In that case, the reference year is derived by expert
    judgement.
    """
    dataSetValidUntil: int | None = None
    """
    End year of the time period for which the data set is still valid / sufficiently representative. This date also determines when a data set revision /
    remodelling is required or recommended due to expected relevant changes in environmentally or technically relevant inventory values, including in the background
    system.
    """
    timeRepresentativenessDescription: LocalizedTextList | None = None
    """
    Description of the valid time span of the data set including information on limited usability within sub-time spans (e.g.
    summer/winter).
    """


class LocationOfOperation(BaseModel):
    location: str | None = None
    """
    Location, country or region the data set represents. [Note 1: This field does not refer to e.g. the country in which a specific site is located that is
    represented by this data set but to the actually represented country, region, or site. Note 2: Entry can be of type "two-letter ISO 3166 country code" for countries,
    "seven-letter regional codes" for regions or continents, or "market areas and market organisations", as predefined for the ILCD. Also a name for e.g. a specific plant
    etc. can be given here (e.g. "FR, Lyon, XY Company, Z Site"; user defined). Note 3: The fact whether the entry refers to production or to consumption / supply has to be
    stated in the name-field "Mix and location types" e.g. as "Production mix".]
    """
    descriptionOfRestrictions: LocalizedTextList | None = None
    """
    Further explanations about additional aspects of the location: e.g. a company and/or site description and address, whether for certain sub-areas within
    the "Location" the data set is not valid, whether data is only valid for certain regions within the location indicated, or whether certain elementary flows or
    intermediate product flows are extrapolated from another geographical area.
    """


class Geography(BaseModel):
    locationOfOperationSupplyOrProduction: LocationOfOperation
    """
    Location, country or region the data set represents. [Note 1: This field does not refer to e.g. the country in which a specific site is located that is
    represented by this data set but to the actually represented country, region, or site. Note 2: Entry can be of type "two-letter ISO 3166 country code" for countries,
    "seven-letter regional codes" for regions or continents, or "market areas and market organisations", as predefined for the ILCD. Also a name for e.g. a specific plant
    etc. can be given here (e.g. "FR, Lyon, XY Company, Z Site"; user defined). Note 3: The fact whether the entry refers to production or to consumption / supply has to be
    stated in the name-field "Mix and location types" e.g. as "Production mix".]
    """


class Technology(BaseModel):
    technologyDescriptionAndIncludedProcesses: LocalizedTextList | None = None
    """
    Description of the technological characteristics including operating conditions of the process or product system. For the latter this includes the
    relevant upstream and downstream processes included in the data set. Professional terminology should be used.
    """
    technologicalApplicability: LocalizedTextList | None = None
    """
    Description of the intended / possible applications of the good, service, or process. E.g. for which type of products the material, represented by this
    data set, is used. Examples: "This high purity chemical is used for analytical laboratories only." or "This technical quality bulk chemical is used for large scale
    synthesis in chemical industry.". Or: "This truck is used only for long-distance transport of liquid bulk chemicals".
    """


class ProcessInformation(BaseModel):
    dataSetInformation: DataSetInformation
    """
    General data set information. Section covers all single fields in the ISO/TS 14048 "Process description", which are not part of the other sub-sections. In
    ISO/TS 14048 no own sub-section is foreseen for these entries.
    """

    quantitativeReference: QuantitativeReference
    """
    This section names the quantitative reference used for this data set, i.e. the reference to which the inputs and outputs quantiatively
    """

    time: Time
    """
    Provides information about the time representativeness of the data set.
    """

    geography: Geography
    """
    Provides information about the geographical representativeness of the data set.
    """

    technology: Technology
    """
    Provides information about the technological representativeness of the data set.
    """


# --------------------------
# modellingAndValidation section
# --------------------------


class LCIMethodAndAllocationType(BaseModel):
    typeOfDataSet: str
    """
    Type of the data set regarding systematic inclusion/exclusion of upstream or downstream processes, transparency and internal (hidden) multi-functionality,
    and the completeness of modelling.
    """


class DataSourcesTreatmentAndRepresentativeness(BaseModel):
    referenceToDataSource: List[SourceDataSet]
    """
    "Source data set"(s) of the source(s) used for deriving/compiling the inventory of this data set e.g. questionnaires, monographies, plant operation
    protocols, etc. For LCI results and Partly terminated systems the sources for relevant background system data are to be given, too. For parameterised data sets the
    sources used for the parameterisation / mathematical relations in the section "Mathematical model" are referenced here as well. [Note: If the data set stems from
    another database or data set publication and is only re-published: identify the origin of a converted data set in "Converted original data set from:" field in section
    "Data entry by" and its unchanged re-publication in "Unchanged re-publication of:" in the section "Publication and ownership". The data sources used to model a
    converted or re-published data set are nevertheless to be given here in this field, for transparency reasons.]
    """


class Review(BaseModel):
    referenceToNameOfReviewerAndInstitution: List[ContactDataSet] | None = None
    type: str
    """
    Type of review that has been performed regarding independency and type of review process.
    """


class Validation(BaseModel):
    review: List[Review]
    """
    Review information on data set.
    """


class Compliance(BaseModel):
    referenceToComplianceSystem: SourceDataSet
    approvalOfOverallCompliance: str
    nomenclatureCompliance: str
    methodologicalCompliance: str
    reviewCompliance: str
    documentationCompliance: str
    qualityCompliance: str


class ComplianceDeclarationsType(BaseModel):
    compliance: List[Compliance]


class ModellingAndValidation(BaseModel):
    LCIMethodAndAllocation: LCIMethodAndAllocationType
    """
    LCI methodological modelling aspects including allocation / substitution information.
    """
    dataSourcesTreatmentAndRepresentativeness: DataSourcesTreatmentAndRepresentativeness
    """
    Data selection, completeness, and treatment principles and procedures, data sources and market coverage information.
    """
    validation: Validation
    """
    Review / validation information on data set.
    """
    complianceDeclarations: ComplianceDeclarationsType | None = None
    """
    Statements on compliance of several data set aspects with compliance requirements as defined by the referenced compliance system (e.g. an EPD scheme,
    handbook of a national or international data network such as the ILCD, etc.).
    """


# --------------------------
# administrativeInformation section
# --------------------------


class CommissionerAndGoal(BaseModel):
    referenceToCommissioner: List[ContactDataSet] | None = None
    project: LocalizedTextList | None = None


class DataGenerator(BaseModel):
    referenceToPersonOrEntityGeneratingTheDataSet: List[ContactDataSet] | None = None


class DataEntryBy(BaseModel):
    referenceToDataSetFormat: List[SourceDataSet]
    referenceToPersonOrEntityEnteringTheData: ContactDataSet | None = None
    timeStamp: int | None = None


class PublicationAndOwnership(BaseModel):
    dataSetVersion: str
    permanentDataSetURI: str | None = None
    dateOfLastRevision: int
    referenceToOwnershipOfDataSet: ContactDataSet
    copyright: bool | None = None
    licenseType: str | None = None
    accessRestrictions: LocalizedTextList = Field(default_factory=LocalizedTextList)


class AdministrativeInformation(BaseModel):
    commissionerAndGoal: CommissionerAndGoal
    """
    Basic information about goal and scope of the data set.
    """
    dataGenerator: DataGenerator
    """
    Expert(s), that compiled and modelled the data set as well as internal administrative information linked to the data generation
    """
    dataEntryBy: DataEntryBy
    """
    Staff or entity, that documented the generated data set, entering the information into the database; plus administrative information linked to the data
    entry activity.
    """
    publicationAndOwnership: PublicationAndOwnership
    """
    Information related to publication and version management of the data set including copyright and access restrictions.
    """


# --------------------------
# exchanges section
# --------------------------


class FlowProperty(BaseModel):
    name: LocalizedTextList
    uuid: str
    referenceFlowProperty: bool
    meanValue: float
    referenceUnit: str
    unitGroupUUID: str


class ClassificationInfo(BaseModel):
    classHierarchy: str
    name: str


class Exchange(BaseModel):
    dataSetInternalID: int
    """
    Automated entry: internal ID, used in the "Quantitative reference" section to identify the "Reference flow(s)" in case the quantitative reference of this
    Process data set is of this type.
    """
    referenceToFlowDataSet: FlowDataSet
    """
    "Flow data set" of this Input or Output.
    """
    exchange_direction: str | None = Field(default=None, alias="exchange direction")
    """
    Direction of Input or Output flow.
    """
    meanAmount: float
    """
    Mean amount of the Input or Output. Only significant digits of the amount should be stated.
    """
    resultingAmount: float
    """
    Final value to be used for calculation of the LCI results and in the product system: It is calculated as the product of the "Mean amount" value times the
    value of the "Variable". In case that no "Variable" entry is given, the "Resulting amount" is identical to the "Mean amount", i.e. a factor "1" is
    applied.
    """
    uncertaintyDistributionType: str | None = None
    """
    Defines the kind of uncertainty distribution that is valid for this particular object or parameter.
    """
    referenceFlow: bool | None = None
    other: Other | None = None
    resolvedFlowVersion: str | None = None
    resultingflowAmount: float
    flowProperties: List[FlowProperty]
    typeOfFlow: str
    locationOfSupply: str | None = None
    classification: ClassificationInfo | None = None
    generalComments: LocalizedTextList | None = None


class Exchanges(BaseModel):
    exchange: List[Exchange]
    """
    Input/Output list of exchanges with the quantitative inventory data as well as pre-calculated LCIA results.
    """


# --------------------------
# LCIAResults section
# --------------------------


class LCIAResultType(BaseModel):
    referenceToLCIAMethodDataSet: LCIAMethodDataSet
    """
    "LCIA method data set" applied to calculate the LCIA results.
    """
    meanAmount: float
    """
    Mean amount of the LCIA result of the inventory, calculated for this LCIA method. Only significant digits should be stated.
    """
    uncertaintyDistributionType: str | None = None
    """
    Defines the kind of uncertainty distribution that is valid for this LCIA result.
    """


class LCIAResultsType(BaseModel):
    LCIAResult: List[LCIAResultType]
    """
    LCIA result
    """


# --------------------------
# Top-level Root Model
# --------------------------


import functools


@functools.cache
def _uuid_to_file():
    return {
        (file.stem.split("_")[0] if "_" in file.stem else file.stem): file
        for file in config.ilcd_processes_json_dir.iterdir()
    }


class ProcessData(BaseModel):
    model_config: ClassVar[ConfigDict] = {"extra": "forbid"}

    processInformation: ProcessInformation
    """
    Corresponds to the ISO/TS 14048 section "Process description". It comprises the following six sub-sections: 1) "Data set information" for data set
    identification and overarching information items, 2) "Quantitative reference", 3) "Time", 4) "Geography", 5) "Technology" and 6) "Mathematical
    relations".
    """

    modellingAndValidation: ModellingAndValidation
    """
    Covers the five sub-sections 1) LCI method and allocation, 2) Data sources, treatment and representativeness, 3) Completeness, 4) Validation, and 5)
    Compliance. (Section refers to LCI modelling and data treatment aspects etc., NOT the modeling of e.g. the input/output-relationships of a parameterised data
    set.)
    """

    administrativeInformation: AdministrativeInformation
    """
    Information on data set management and administration.
    """

    exchanges: Exchanges
    """
    Input/Output list of exchanges with the quantitative inventory data, as well as pre-calculated LCIA results.
    """

    LCIAResults: LCIAResultsType
    """
    List with the pre-calculated LCIA results of the Input/Output list of this data set. May contain also inventory-type results such as primary energy
    consumption etc.
    """

    version: str
    """
    Indicates, which version of the ILCD format is used
    """

    locations: str | None = None
    """
    contains reference to used location table for this dataset
    """

    def get_main_output(self) -> Exchange | None:
        if quantitative_reference := self.processInformation.quantitativeReference:
            if quantitative_reference.functionalUnitOrOther:
                functional_unit = quantitative_reference.functionalUnitOrOther.get()
            else:
                functional_unit = None

            output_flow_ids = quantitative_reference.referenceToReferenceFlow
            assert len(output_flow_ids) == 1
            output_flow_id = output_flow_ids[0]
            output_flow_uuid = next(
                flow.referenceToFlowDataSet.refObjectId
                for flow in self.exchanges.exchange
                if flow.dataSetInternalID == output_flow_id
            )
            output_flow = next(
                flow
                for flow in self.exchanges.exchange
                if flow.referenceToFlowDataSet.refObjectId == output_flow_uuid
            )

            return output_flow

    @classmethod
    def from_json_file(cls: type[Self], file_path: PathLike) -> Self:
        with open(file_path, "r") as file:
            return cls.model_validate_json(file.read())

    @classmethod
    def _find_files_by_uuids(cls: type[Self], uuids: list[str]) -> Iterator[PathLike]:
        yield from (_uuid_to_file()[uuid] for uuid in uuids)

    @classmethod
    def from_uuid(cls: type[Self], uuid: str) -> Self:
        return cls.from_json_file(next(cls._find_files_by_uuids([uuid])))

    @classmethod
    def from_uuids(cls: type[Self], uuids: list[str]) -> Iterator[Self]:
        for file in cls._find_files_by_uuids(uuids):
            try:
                yield cls.from_json_file(file)
            except Exception:
                print(f"Failed to parse {file!s}")
                raise

    @classmethod
    def iter_all(cls: type[Self], lci_results_only: bool = True) -> Iterator[Self]:
        return cls.from_uuids(read_uuids(lci_results_only=lci_results_only))


def read_uuids(lci_results_only=True) -> list[str]:
    filename = "lci-results-uuids.txt" if lci_results_only else "uuids.txt"
    return (config.data_dir / filename).read_text().splitlines()
