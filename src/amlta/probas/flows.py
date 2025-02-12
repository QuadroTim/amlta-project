from typing import Iterable, Literal, NamedTuple, cast

import pandas as pd

from amlta.probas import processes


class Flow(NamedTuple):
    exchange_direction: Literal["INPUT", "OUTPUT", None]
    exchange_resulting_amount: float
    exchange_type_of_flow: str
    exchange_classification_hierarchy: str | None
    flow_uuid: str
    flow_description: str
    flow_property_uuid: str
    flow_property_name: str
    flow_property_unit: str


def extract_process_flows(process: processes.ProcessData) -> pd.DataFrame:
    # TODO: maybe add `locationOfSupply`
    process_flows = pd.DataFrame(
        [
            {
                "exchange_direction": exchange.exchange_direction,
                # what are the differences?
                # "exchange_mean_amount": exchange.meanAmount,
                "exchange_resulting_amount": exchange.resultingAmount,
                # "exchange_resulting_flow_amount": exchange.resultingflowAmount,
                "exchange_type_of_flow": exchange.typeOfFlow,
                "exchange_classification_hierarchy": (
                    exchange.classification.classHierarchy
                    if exchange.classification
                    else None
                ),
                "flow_uuid": exchange.referenceToFlowDataSet.refObjectId,
                "flow_description": (
                    exchange.referenceToFlowDataSet.shortDescription.get("en")
                ),
                "flow_property_uuid": flow_property.uuid,
                "flow_property_name": flow_property.name.get("en"),
                "flow_property_unit": flow_property.referenceUnit,
            }
            for exchange in process.exchanges.exchange
            for flow_property in exchange.flowProperties
        ]
    )

    return process_flows


def iter_flows(process: processes.ProcessData) -> Iterable[Flow]:
    def _assert_get(value: processes.LocalizedTextList) -> str:
        val = value.get("en")
        assert val is not None
        return val

    for exchange in process.exchanges.exchange:
        for flow_property in exchange.flowProperties:
            yield Flow(
                exchange_direction=cast(
                    Literal["INPUT", "OUTPUT"], exchange.exchange_direction
                ),
                exchange_resulting_amount=exchange.resultingAmount,
                exchange_type_of_flow=exchange.typeOfFlow,
                exchange_classification_hierarchy=(
                    exchange.classification.classHierarchy
                    if exchange.classification
                    else None
                ),
                flow_uuid=exchange.referenceToFlowDataSet.refObjectId,
                flow_description=_assert_get(
                    exchange.referenceToFlowDataSet.shortDescription
                ),
                flow_property_uuid=flow_property.uuid,
                flow_property_name=_assert_get(flow_property.name),
                flow_property_unit=flow_property.referenceUnit,
            )
