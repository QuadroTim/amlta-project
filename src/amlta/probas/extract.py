import pandas as pd

from amlta.probas import processes


def extract_process_flows(process: processes.ProcessData) -> pd.DataFrame:
    process_flows = pd.DataFrame(
        [
            {
                "exchange_direction": exchange.exchange_direction,
                # what are the differences?
                "exchange_mean_amount": exchange.meanAmount,
                "exchange_resulting_amount": exchange.resultingAmount,
                "exchange_resulting_flow_amount": exchange.resultingflowAmount,
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
