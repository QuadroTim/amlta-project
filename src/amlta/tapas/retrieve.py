import collections
import itertools
import warnings
from typing import Literal, cast, overload

import pandas as pd
import torch
from transformers import TapasForQuestionAnswering

from amlta.tapas.base import id2aggregation
from amlta.tapas.model import CustomTapasTokenizer
from amlta.tapas.preprocessing import transform_flows_for_tapas

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


@overload
def retrieve_rows_from_chunk(
    chunk: pd.DataFrame,
    query: str,
    *,
    model: TapasForQuestionAnswering,
    tokenizer: CustomTapasTokenizer,
    threshold: float = 0.15,
    return_probabilities: Literal[True] = True,
) -> tuple[list[int], str, list[float]]: ...
@overload
def retrieve_rows_from_chunk(
    chunk: pd.DataFrame,
    query: str,
    *,
    model: TapasForQuestionAnswering,
    tokenizer: CustomTapasTokenizer,
    threshold: float = 0.15,
    return_probabilities: Literal[False] = False,
) -> tuple[list[int], str]: ...
def retrieve_rows_from_chunk(
    chunk: pd.DataFrame,
    query: str,
    *,
    model: TapasForQuestionAnswering,
    tokenizer: CustomTapasTokenizer,
    threshold=0.15,
    return_probabilities=False,
):
    inputs = tokenizer(
        table=chunk.drop(columns=["original_index"]),
        queries=query,
        padding="max_length",
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs.to(device))

    if return_probabilities:
        predicted_answer_coordinates_w_probs, predicted_aggregation_indices = (
            tokenizer.convert_logits_to_probabilities(
                inputs.to("cpu"),
                outputs.logits.detach().to("cpu"),
                outputs.logits_aggregation.detach().to("cpu"),
            )
        )
        predicted_answer_probabilities = []
        predicted_answer_coordinates = []

        for coords, probs in predicted_answer_coordinates_w_probs[0]:
            if probs > threshold:
                predicted_answer_probabilities.append(probs)
                predicted_answer_coordinates.append(coords)
    else:
        predicted_answer_coordinates, predicted_aggregation_indices = (  # type: ignore
            tokenizer.convert_logits_to_predictions(
                inputs.to("cpu"),
                outputs.logits.detach().to("cpu"),
                outputs.logits_aggregation.detach().to("cpu"),
                cell_classification_threshold=threshold,
            )
        )
        predicted_answer_probabilities = None
        predicted_answer_coordinates = predicted_answer_coordinates[0]

    predicted_aggregation_index = predicted_aggregation_indices[0]
    aggregation = id2aggregation[predicted_aggregation_index]

    if return_probabilities:
        max_predicted_answer_probabilities = []
        row_indices = []

        for row, grp in itertools.groupby(
            zip(
                predicted_answer_coordinates, cast(list, predicted_answer_probabilities)
            ),
            key=lambda x: x[0][0],  # group by row
        ):
            row_indices.append(int(chunk.iloc[row]["original_index"]))
            # store the max probability for each row
            max_predicted_answer_probabilities.append(max([x[1] for x in grp]))

        predicted_answer_probabilities = max_predicted_answer_probabilities

    else:
        row_indices = []
        last_row_index = -1

        for row_index, _ in predicted_answer_coordinates:
            if row_index != last_row_index:
                last_row_index = row_index
                row_indices.append(int(chunk.iloc[row_index]["original_index"]))

    if return_probabilities:
        return (
            row_indices,
            aggregation,
            cast(list[float], predicted_answer_probabilities),
        )
    else:
        return row_indices, aggregation


@overload
def retrieve_rows(
    df: pd.DataFrame,
    query: str,
    *,
    shuffle_rows=True,
    model: TapasForQuestionAnswering,
    tokenizer: CustomTapasTokenizer,
    chunk_size: int = 15,
    threshold=0.15,
    return_probabilities: Literal[True] = True,
) -> tuple[list[int], str, list[float]]: ...
@overload
def retrieve_rows(
    df: pd.DataFrame,
    query: str,
    *,
    shuffle_rows=True,
    model: TapasForQuestionAnswering,
    tokenizer: CustomTapasTokenizer,
    chunk_size: int = 15,
    threshold=0.15,
    return_probabilities: Literal[False] = False,
) -> tuple[list[int], str]: ...


def retrieve_rows(
    df: pd.DataFrame,
    query: str,
    *,
    shuffle_rows=True,
    model: TapasForQuestionAnswering,
    tokenizer: CustomTapasTokenizer,
    chunk_size: int = 15,
    threshold=0.15,
    return_probabilities=False,
):
    all_row_indices = []
    all_probabilities = []
    all_aggregations = collections.Counter()

    for batch in generate_tapas_chunks(
        df,
        shuffle_rows=shuffle_rows,
        chunk_size=chunk_size,
    ):
        row_indices, predicted_aggregation, *rest = retrieve_rows_from_chunk(
            batch,
            query,
            model=model,
            tokenizer=tokenizer,
            threshold=threshold,
            return_probabilities=return_probabilities,
        )

        all_aggregations[predicted_aggregation] += 1
        all_row_indices.extend(row_indices)
        if return_probabilities:
            all_probabilities.extend(rest[0])

    aggregation, _ = all_aggregations.most_common(1)[0]

    if return_probabilities:
        return all_row_indices, aggregation, all_probabilities
    else:
        return all_row_indices, aggregation
