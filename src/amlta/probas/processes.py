from collections import UserList
from os import PathLike
from typing import Any, ClassVar, Iterator, List, Optional, Self

from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from amlta.config import config

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

    def get(self, preferred_lang: str = "en") -> str:
        for text in self:
            if text.lang == preferred_lang:
                return text.value if text.value else ""

        return next((text.value for text in self if text.value), "")


class AccessRestriction(BaseModel):
    value: str
    lang: str


class ContactData(BaseModel):
    shortDescription: LocalizedTextList
    type: str
    refObjectId: str


# --------------------------
# processInformation section
# --------------------------


class BaseNameWrapper(BaseModel):
    baseName: LocalizedTextList


class ClassificationItem(BaseModel):
    value: str
    level: int
    classId: str


class Classification(BaseModel):
    # "class" is a reserved word in Python so we use "class_" and set alias.
    class_: List[ClassificationItem] = Field(..., alias="class")
    name: str
    classes: str


class ClassificationInformation(BaseModel):
    classification: List[Classification] | None = None


class DataSetInformation(BaseModel):
    UUID: str
    name: BaseNameWrapper
    classificationInformation: ClassificationInformation
    generalComment: LocalizedTextList
    referenceToExternalDocumentation: List[ContactData] | None = None


class QuantitativeReference(BaseModel):
    referenceToReferenceFlow: List[int]
    functionalUnitOrOther: LocalizedTextList | None = None
    type: str


class Time(BaseModel):
    referenceYear: int


class LocationOfOperation(BaseModel):
    location: str | None = None


class Geography(BaseModel):
    locationOfOperationSupplyOrProduction: LocationOfOperation


class Technology(BaseModel):
    technologyDescriptionAndIncludedProcesses: LocalizedTextList | None = None


class ProcessInformation(BaseModel):
    dataSetInformation: DataSetInformation
    quantitativeReference: QuantitativeReference
    time: Time
    geography: Geography
    technology: Technology


# --------------------------
# modellingAndValidation section
# --------------------------


class LCIMethodAndAllocation(BaseModel):
    typeOfDataSet: str


class DataSourcesTreatmentAndRepresentativeness(BaseModel):
    referenceToDataSource: List[ContactData]


class Review(BaseModel):
    # reusing ContactData for reviewer info
    referenceToNameOfReviewerAndInstitution: List[ContactData] | None = None
    type: str


class Validation(BaseModel):
    review: List[Review]


class ModellingAndValidation(BaseModel):
    LCIMethodAndAllocation: LCIMethodAndAllocation
    dataSourcesTreatmentAndRepresentativeness: DataSourcesTreatmentAndRepresentativeness
    validation: Validation


# --------------------------
# administrativeInformation section
# --------------------------


class CommissionerAndGoal(BaseModel):
    referenceToCommissioner: List[ContactData] | None = None
    project: LocalizedTextList | None = None


class DataGenerator(BaseModel):
    referenceToPersonOrEntityGeneratingTheDataSet: List[ContactData] | None = None


class DataSetFormat(BaseModel):
    shortDescription: LocalizedTextList
    type: str
    refObjectId: str


class DataEntryBy(BaseModel):
    referenceToDataSetFormat: List[DataSetFormat]
    referenceToPersonOrEntityEnteringTheData: ContactData | None = None


class PublicationAndOwnership(BaseModel):
    dataSetVersion: str
    permanentDataSetURI: str | None = None
    dateOfLastRevision: int
    referenceToOwnershipOfDataSet: ContactData
    copyright: bool | None = None
    licenseType: str | None = None
    accessRestrictions: List[AccessRestriction] | None = None


class AdministrativeInformation(BaseModel):
    commissionerAndGoal: CommissionerAndGoal
    dataGenerator: DataGenerator
    dataEntryBy: DataEntryBy
    publicationAndOwnership: PublicationAndOwnership


# --------------------------
# exchanges section
# --------------------------


class ReferenceToFlowDataSet(BaseModel):
    shortDescription: LocalizedTextList
    type: str
    refObjectId: str


class FlowProperty(BaseModel):
    name: LocalizedTextList
    uuid: str
    referenceFlowProperty: bool
    meanValue: float
    referenceUnit: str
    unitGroupUUID: str


class Other(BaseModel):
    anies: List[str]


class ClassificationInfo(BaseModel):
    classHierarchy: str
    name: str


class Exchange(BaseModel):
    dataSetInternalID: int
    referenceToFlowDataSet: ReferenceToFlowDataSet
    # "exchange direction" contains a space so we use an alias:
    exchange_direction: str | None = Field(default=None, alias="exchange direction")
    meanAmount: float
    resultingAmount: float
    uncertaintyDistributionType: str | None = None
    other: Other | None = None
    resolvedFlowVersion: str | None = None
    resultingflowAmount: float
    flowProperties: List[FlowProperty]
    typeOfFlow: str
    classification: ClassificationInfo | None = None


class Exchanges(BaseModel):
    exchange: List[Exchange]


# --------------------------
# LCIAResults section
# --------------------------


class ReferenceToLCIAMethodDataSet(BaseModel):
    shortDescription: LocalizedTextList
    type: str
    refObjectId: str


class LCIAResult(BaseModel):
    referenceToLCIAMethodDataSet: ReferenceToLCIAMethodDataSet
    meanAmount: float
    uncertaintyDistributionType: str | None = None


class LCIAResults(BaseModel):
    LCIAResult: List[LCIAResult]


# --------------------------
# Top-level Root Model
# --------------------------


class ProcessData(BaseModel):
    model_config: ClassVar[ConfigDict] = {"extra": "forbid"}

    processInformation: ProcessInformation
    modellingAndValidation: ModellingAndValidation
    administrativeInformation: AdministrativeInformation
    exchanges: Exchanges
    LCIAResults: LCIAResults
    version: str
    locations: str | None = None

    @classmethod
    def from_json_file(cls: type[Self], file_path: PathLike) -> Self:
        with open(file_path, "r") as file:
            return cls.model_validate_json(file.read())

    @classmethod
    def from_uuids(cls: type[Self], uuids: list[str]) -> Iterator[Self]:
        file_to_uuid = lambda file: (
            file.stem.split("_")[0] if "_" in file.stem else file.stem
        )
        process_files = {
            file_to_uuid(file): file
            for file in config.ilcd_processes_json_dir.iterdir()
        }

        for uuid in uuids:
            try:
                yield cls.from_json_file(process_files[uuid])
            except Exception:
                print(f"Failed to parse {process_files[uuid]!s}")
                raise


def read_uuids(lci_results_only=True) -> list[str]:
    filename = "lci-results-uuids.txt" if lci_results_only else "uuids.txt"
    return (config.data_dir / filename).read_text().splitlines()
