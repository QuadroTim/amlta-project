import warnings

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


def load_tapas_config(name=hf_finetuned_name) -> PretrainedConfig:
    return TapasConfig.from_pretrained(name)


def load_tapas_tokenizer(name=tapas_wikisql_name) -> TapasTokenizer:
    return TapasTokenizer.from_pretrained(name)


def load_tapas_model(name=hf_finetuned_name) -> TapasForQuestionAnswering:
    return TapasForQuestionAnswering.from_pretrained(
        name,
        config=load_tapas_config(name),
    ).to(device)  # type: ignore
