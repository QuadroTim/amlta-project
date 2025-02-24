import collections
import warnings

import numpy as np
import torch
from transformers import (
    PretrainedConfig,
    TapasConfig,
    TapasForQuestionAnswering,
    TapasTokenizer,
)

warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tapas_wikisql_name = "google/tapas-base-finetuned-wikisql-supervised"
hf_finetuned_name = "woranov/tapas-finetuned-probas-supervised-2"


# XXX: custom methods unused
class CustomTapasTokenizer(TapasTokenizer):
    def _get_cell_token_probs_selected_rows(
        self, probabilities, segment_ids, row_ids, column_ids, selected_rows
    ):
        """
        Yields token indices and probabilities only for tokens whose row (0-indexed)
        is in the provided selected_rows set.
        """
        for i, p in enumerate(probabilities):
            segment_id = segment_ids[i]
            col = column_ids[i] - 1
            row = row_ids[i] - 1
            if col >= 0 and row >= 0 and segment_id == 1 and (row in selected_rows):
                yield i, p

    def _get_mean_row_probs(
        self, probabilities, segment_ids, row_ids, column_ids, selected_rows
    ):
        """
        Aggregates token probabilities by row (only for rows in selected_rows)
        and returns a dictionary mapping row indices to the average probability.
        """
        row_to_probs = collections.defaultdict(list)
        for i, prob in self._get_cell_token_probs_selected_rows(
            probabilities, segment_ids, row_ids, column_ids, selected_rows
        ):
            row = row_ids[i] - 1
            row_to_probs[row].append(prob)
        return {row: np.mean(probs) for row, probs in row_to_probs.items()}

    def convert_logits_to_predictions_selected_rows(
        self,
        data,
        logits,
        logits_agg=None,
        cell_classification_threshold=0.5,
        selected_rows=None,
    ):
        """
        Customized conversion that:
        - Computes probabilities as before,
        - Only aggregates over tokens in the provided selected rows, and
        - Returns, for each example in the batch, a tuple of:
                (selected_row_indices, corresponding_row_probabilities)
        If selected_rows is None, all rows (as defined by the data) are used.
        """
        # Convert to numpy arrays.
        logits = logits.numpy()
        if logits_agg is not None:
            logits_agg = logits_agg.numpy()
        data = {key: value.numpy() for key, value in data.items() if key != "training"}

        # Prevent overflow in exponential.
        logits[logits < -88.7] = -88.7

        # Compute probabilities and mask out padding tokens.
        probabilities = 1 / (1 + np.exp(-logits)) * data["attention_mask"]

        # Define token type names.
        token_types = [
            "segment_ids",
            "column_ids",
            "row_ids",
            "prev_labels",
            "column_ranks",
            "inv_column_ranks",
            "numeric_relations",
        ]

        input_ids = data["input_ids"]
        segment_ids = data["token_type_ids"][:, :, token_types.index("segment_ids")]
        row_ids = data["token_type_ids"][:, :, token_types.index("row_ids")]
        column_ids = data["token_type_ids"][:, :, token_types.index("column_ids")]

        num_batch = input_ids.shape[0]
        predictions = []

        for i in range(num_batch):
            # Convert per-example data to lists.
            probabilities_example = probabilities[i].tolist()
            segment_ids_example = segment_ids[i].tolist()
            row_ids_example = row_ids[i].tolist()
            column_ids_example = column_ids[i].tolist()

            # Determine the available rows (assuming row ids > 0).
            max_height = max(row_ids_example) if row_ids_example else 0

            # If no specific rows provided, use all available rows (converted to 0-index).
            if selected_rows is None:
                selected_rows_example = set(range(max_height))
            else:
                # Ensure the provided rows are treated as 0-indexed.
                selected_rows_example = set(selected_rows)

            # Get the average probability for each selected row.
            row_probs = self._get_mean_row_probs(
                probabilities_example,
                segment_ids_example,
                row_ids_example,
                column_ids_example,
                selected_rows_example,
            )

            # Filter rows based on the threshold and collect indices & probabilities.
            selected_row_indices = []
            selected_row_probabilities = []
            for row, prob in row_probs.items():
                if prob > cell_classification_threshold:
                    selected_row_indices.append(row)
                    selected_row_probabilities.append(prob)

            predictions.append((selected_row_indices, selected_row_probabilities))

        output = (predictions,)

        if logits_agg is not None:
            predicted_aggregation_indices = logits_agg.argmax(axis=-1)
            output = (predictions, predicted_aggregation_indices.tolist())

        return output


def load_tapas_config(name=hf_finetuned_name) -> PretrainedConfig:
    return TapasConfig.from_pretrained(name)


def load_tapas_tokenizer(name=tapas_wikisql_name) -> CustomTapasTokenizer:
    return CustomTapasTokenizer.from_pretrained(name)


def load_tapas_model(name=hf_finetuned_name) -> TapasForQuestionAnswering:
    return TapasForQuestionAnswering.from_pretrained(
        name,
        config=load_tapas_config(name),
    ).to(device)  # type: ignore
