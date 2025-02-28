import pandas as pd


def transform_flows_for_tapas(flows_df: pd.DataFrame):
    return (
        flows_df[
            [
                "exchange_direction",
                "exchange_resulting_amount",
                "exchange_type_of_flow",
                "exchange_classification_hierarchy",
                "flow_description",
                "flow_property_name",
                "flow_property_unit",
            ]
        ]
        .rename(
            columns={
                "exchange_direction": "Direction",
                "exchange_resulting_amount": "Amount",
                "exchange_type_of_flow": "Type",
                "exchange_classification_hierarchy": "Class",
                "flow_description": "Name",
                "flow_property_name": "Property",
                "flow_property_unit": "Unit",
            }
        )
        .replace(
            {
                "Direction": {
                    "INPUT": "Input",
                    "OUTPUT": "Output",
                },
            }
        )
        .pipe(
            lambda df: df.assign(
                **{col: df[col].round(3) for col in df.select_dtypes("number").columns}
            )
        )
    ).astype(str)
