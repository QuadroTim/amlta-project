import collections
import warnings

import pandas as pd
import torch
from transformers import TapasForQuestionAnswering, TapasTokenizer

from amlta.data_processing.tapas_flows import transform_flows_for_tapas
from amlta.probas.flows import Flow
from amlta.tapas.base import id2aggregation

warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def retrieve_rows(
    df: pd.DataFrame,
    query: str,
    *,
    model: TapasForQuestionAnswering,
    tokenizer: TapasTokenizer,
    chunk_size: int = 15,
    threshold=0.15,
):
    table = transform_flows_for_tapas(df)

    row_indices = []
    aggregation_candidates = collections.Counter()

    batch_gen = (
        (
            (i, min(len(table), i + chunk_size)),  # df start, stop
            table.iloc[i : i + chunk_size],  # df batch
        )
        for i in range(0, len(table), chunk_size)
    )

    model.eval()

    for (start, stop), batch in batch_gen:
        batch = batch.reset_index(drop=True)
        inputs = tokenizer(
            table=batch,
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

        aggregation_candidates[id2aggregation[predicted_aggregation_index]] += 1

        row_indices += [
            start + row_index
            for row_index, _ in predicted_answer_coordinates
            if row_index < len(batch)
        ]

    aggregation, _ = aggregation_candidates.most_common(1)[0]
    filtered_df = df.iloc[row_indices]

    flows = [Flow(**row.to_dict()) for _, row in filtered_df.iterrows()]

    return flows, aggregation
