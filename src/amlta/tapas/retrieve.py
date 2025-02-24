import collections
import warnings

import pandas as pd
import torch
from transformers import TapasForQuestionAnswering

from amlta.data_processing.tapas_flows import transform_flows_for_tapas
from amlta.tapas.base import id2aggregation
from amlta.tapas.model import CustomTapasTokenizer

warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_tapas_chunks(
    df: pd.DataFrame,
    *,
    shuffle_rows=True,
    chunk_size: int = 15,
):
    table = (
        transform_flows_for_tapas(df)
        .reset_index(drop=False)
        .rename(columns={"index": "original_index"})
    )
    if shuffle_rows:
        table = table.sample(frac=1, random_state=42)

    batch_gen = (
        (
            (i, min(len(table), i + chunk_size)),  # df start, stop
            table.iloc[i : i + chunk_size],  # df batch
        )
        for i in range(0, len(table), chunk_size)
    )

    for (start, stop), batch in batch_gen:
        batch = batch.reset_index(drop=True)

        yield batch


def retrieve_rows_from_chunk(
    chunk: pd.DataFrame,
    query: str,
    *,
    model: TapasForQuestionAnswering,
    tokenizer: CustomTapasTokenizer,
    threshold=0.15,
):
    inputs = tokenizer(
        table=chunk.drop(columns=["original_index"]),
        queries=query,
        padding="max_length",
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs.to(device))

    predicted_answer_coordinates, predicted_aggregation_indices = (  # type: ignore
        tokenizer.convert_logits_to_predictions(
            inputs.to("cpu"),
            outputs.logits.detach().to("cpu"),
            outputs.logits_aggregation.detach().to("cpu"),
            cell_classification_threshold=threshold,
        )
    )

    predicted_answer_coordinates = predicted_answer_coordinates[0]
    predicted_aggregation_index = predicted_aggregation_indices[0]

    aggregation = id2aggregation[predicted_aggregation_index]

    row_indices = [
        int(chunk.iloc[row_index]["original_index"])
        for row_index, _ in predicted_answer_coordinates
    ]

    return row_indices, aggregation


def retrieve_rows(
    df: pd.DataFrame,
    query: str,
    *,
    shuffle_rows=True,
    model: TapasForQuestionAnswering,
    tokenizer: CustomTapasTokenizer,
    chunk_size: int = 15,
    threshold=0.15,
):
    row_indices = []
    aggregation_candidates = collections.Counter()

    for batch in generate_tapas_chunks(
        df,
        shuffle_rows=shuffle_rows,
        chunk_size=chunk_size,
    ):
        row_indices, predicted_aggregation = retrieve_rows_from_chunk(
            batch,
            query,
            model=model,
            tokenizer=tokenizer,
            threshold=threshold,
        )

        aggregation_candidates[predicted_aggregation] += 1
        row_indices.extend(row_indices)

    aggregation, _ = aggregation_candidates.most_common(1)[0]

    return row_indices, aggregation
