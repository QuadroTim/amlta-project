import random
from typing import (
    Literal,
    NotRequired,
    TypeAlias,
    TypedDict,
    cast,
)

import pandas as pd

from amlta.probas.flows import extract_process_flows
from amlta.probas.processes import ProcessData

FlowQueryParamType: TypeAlias = Literal["name", "names", "class", "type"]
FlowQueryParamDirection: TypeAlias = Literal["input", "output", "both"]
FlowQueryParamAggregation: TypeAlias = Literal["list", "count", "sum", "average"]


class FlowQueryParams(TypedDict):
    query_type: FlowQueryParamType
    flow_name: NotRequired[str]
    flow_names: NotRequired[list[str]]
    flow_class: NotRequired[str]
    flow_type: NotRequired[str]
    flow_unit: NotRequired[str]
    direction: FlowQueryParamDirection
    aggregation: FlowQueryParamAggregation


def base_flow_target_query(
    query_type: FlowQueryParamType,
) -> FlowQueryParams:
    return {
        "query_type": query_type,
        "direction": "both",
        "aggregation": "list",
    }


def get_flows_for_query(flows_df: pd.DataFrame, query: FlowQueryParams) -> pd.DataFrame:
    query_type = query["query_type"]
    if (direction := query["direction"]) != "both":
        flows_df = flows_df.loc[flows_df["exchange_direction"] == direction.upper()]

    if query_type in {"name", "names"}:
        names = query.get("flow_name") or query.get("flow_names")
        if isinstance(names, str):
            names = [names]

        assert names
        flows_df = flows_df.loc[flows_df["flow_description"].isin(names)]

    if flow_class := query.get("flow_class"):
        flows_df = flows_df.loc[
            flows_df["exchange_classification_hierarchy"]
            .fillna("")
            .str.contains(flow_class, case=False, regex=False)
        ]

    if flow_type := query.get("flow_type"):
        flows_df = flows_df.loc[
            flows_df["exchange_type_of_flow"].fillna("").str.lower()
            == flow_type.lower()
        ]

    if flow_unit := query.get("flow_unit"):
        flows_df = flows_df.loc[
            flows_df["flow_property_unit"].fillna("").str.lower() == flow_unit.lower()
        ]

    return flows_df


def _uniquify_flow_type_and_unit(flows_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    query_target_update = {}

    flow_types_n = flows_df["exchange_type_of_flow"].nunique()
    if flow_types_n > 1:
        type_of_flow, flows_df = random.choice(
            list(flows_df.groupby("exchange_type_of_flow"))
        )
        query_target_update["flow_type"] = cast(str, type_of_flow)

    units_n = flows_df["flow_property_unit"].nunique()
    if units_n > 1:
        unit, flows_df = random.choice(list(flows_df.groupby("flow_property_unit")))
        query_target_update["flow_unit"] = cast(str, unit)

    return flows_df, query_target_update


def generate_random_query_params(
    process: ProcessData,
) -> tuple[pd.DataFrame, FlowQueryParams]:
    flows_df = extract_process_flows(process)

    if main_output_flow := process.get_main_output():
        flows_df = flows_df.loc[
            flows_df["flow_uuid"] != main_output_flow.referenceToFlowDataSet.refObjectId
        ]

    flow_target_query_type: FlowQueryParamType = random.choice(
        ["name", "names", "class", "type"]
    )

    query_params = base_flow_target_query(flow_target_query_type)

    possible_aggregations: list[FlowQueryParamAggregation]

    if random.random() > 0.1:
        # most questions would be about the same flow direction
        grouping_cols = ["exchange_direction"]
        possible_aggregations = ["list", "count", "sum", "average"]
    else:
        grouping_cols = []
        possible_aggregations = ["list", "count"]

    aggregation: FlowQueryParamAggregation = random.choice(possible_aggregations)
    query_params["aggregation"] = aggregation

    if aggregation in {"sum", "average"}:
        should_be_unique_type_and_unit = True
    else:
        should_be_unique_type_and_unit = False

    grouping_cols += ["exchange_type_of_flow"]

    if flow_target_query_type == "class":
        class_levels = flows_df["exchange_classification_hierarchy"].str.split(
            " / ", expand=True
        )
        class_levels.columns = [
            f"class_{i}" for i in range(1, class_levels.shape[1] + 1)
        ]
        flows_df = pd.concat([flows_df, class_levels], axis=1)

        grouping_cols += ["class_1"]

        group_key, group = random.choice(list(flows_df.groupby(grouping_cols)))
        group_data = dict(zip(grouping_cols, group_key))

        if should_be_unique_type_and_unit:
            group, query_target_update = _uniquify_flow_type_and_unit(group)
            query_params.update(query_target_update)

        # if "flow_property_uuid" in group_data:
        #     query_target["flow_type"] = group["exchange_type_of_flow"].iloc[0]

        if "exchange_direction" in group_data:
            query_params["direction"] = group_data["exchange_direction"].lower()

        group_class = [group_data["class_1"]]

        sub_grouping_cols = []
        # if deeper class levels exist, choose a random level:
        for i in range(2, class_levels.shape[1] + 1):
            if f"class_{i}" in group.columns and group[f"class_{i}"].notna().any():
                sub_grouping_cols.append(f"class_{i}")
                break

        if sub_grouping_cols:
            depth = random.randint(0, len(sub_grouping_cols))
            sub_grouping_cols = sub_grouping_cols[:depth]

        if sub_grouping_cols:
            _sub_group_key, group = random.choice(
                list(group.groupby(sub_grouping_cols))
            )
            group_class.extend(_sub_group_key)

        query_params["flow_class"] = " / ".join(group_class)

    elif flow_target_query_type == "type":
        group_key, group = random.choice(list(flows_df.groupby(grouping_cols)))
        group_data = dict(zip(grouping_cols, group_key))

        if "exchange_direction" in group_data:
            query_params["direction"] = group_data["exchange_direction"].lower()

        if should_be_unique_type_and_unit:
            group, query_target_update = _uniquify_flow_type_and_unit(group)
            query_params.update(query_target_update)

        if "flow_type" not in query_params:
            query_params["flow_type"] = group_data["exchange_type_of_flow"]

    elif flow_target_query_type == "name":
        group_key, group = random.choice(list(flows_df.groupby("flow_description")))
        query_params["flow_name"] = cast(str, group_key)

    elif flow_target_query_type == "names":
        group_key, group = random.choice(list(flows_df.groupby(grouping_cols)))
        group_data = dict(zip(grouping_cols, group_key))

        if should_be_unique_type_and_unit:
            group_filtered, query_target_update = _uniquify_flow_type_and_unit(group)
            # No need to update query_params with unique type and unit
            # query_target.update(query_target_update)

            # Instead, limit to names only appearing once
            group = group.loc[
                group["flow_description"].isin(group_filtered["flow_description"])
            ]
            group = group.groupby("flow_description").filter(lambda x: len(x) == 1)

        n = min(random.randint(2, 5), len(group))
        query_params["flow_names"] = group["flow_description"].sample(n).tolist()
        group = group.loc[group["flow_description"].isin(query_params["flow_names"])]

        if "exchange_direction" in group_data:
            query_params["direction"] = group_data["exchange_direction"].lower()

    assert (  # sanity check
        group.values == get_flows_for_query(flows_df, query_params).values
    ).all()

    # group = transform_flows_for_tapas(group)
    return group, query_params
