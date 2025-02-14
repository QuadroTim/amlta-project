import functools

import pandas as pd

from amlta.data_processing.tapas_flows import transform_flows_for_tapas
from amlta.probas import flows, processes
from amlta.question_generation.process_flows import QuestionData


@functools.lru_cache(maxsize=128)
def get_process(process_uuid: str) -> processes.ProcessData:
    return processes.ProcessData.from_uuid(process_uuid)


@functools.lru_cache(maxsize=128)
def get_flows_df(process_uuid: str) -> pd.DataFrame:
    process = get_process(process_uuid)
    return flows.extract_process_flows(process)


def get_training_batches(question: QuestionData, batch_size: int = 20):
    dfs = []

    flows_df = get_flows_df(question["process_uuid"])
    flows_df_tapas = transform_flows_for_tapas(flows_df)

    flow_uuids = [item["uuid"] for item in question["flows"]]
    flows_col_index = flows_df_tapas.columns.get_loc("Amount")

    batch_gen = (
        (
            (i, min(len(flows_df), i + batch_size)),  # df start, stop
            flows_df.iloc[i : i + batch_size],  # df batch
        )
        for i in range(0, len(flows_df), batch_size)
    )

    for (start, stop), batch in batch_gen:
        batch = batch.reset_index(drop=True)
        batch_flows = batch.loc[batch["flow_uuid"].isin(flow_uuids)]
        batch_flows_rows = [int(x) for x in batch_flows.index.values.tolist()]
        coordinates = [(row, flows_col_index) for row in batch_flows_rows]
        answers = flows_df_tapas.iloc[start:stop]["Amount"].values[batch_flows_rows]

        dfs.append(
            pd.DataFrame(
                {
                    "question_general": question["question_replaced_general"],
                    "question_specific": question["question_replaced_specific"],
                    "process_uuid": question["process_uuid"],
                    "flows_start": start,
                    "flows_stop": stop,
                    "coordinates": [coordinates],
                    "answers": [answers.tolist()],
                }
            )
        )

    return pd.concat(dfs, ignore_index=True)
