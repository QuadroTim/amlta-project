import warnings

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from transformers import (
    PretrainedConfig,
    TapasConfig,
    TapasForQuestionAnswering,
    TapasTokenizer,
)

from amlta.tapas.base import tapas_ft_checkpoints_dir

warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tapas_wikisql_name = "google/tapas-base-finetuned-wikisql-supervised"
tapas_base_name = "google/tapas-base"

tapas_wikisql_name = "google/tapas-base-finetuned-wikisql-supervised"
tapas_base_name = "google/tapas-base"
checkpoint = tapas_ft_checkpoints_dir / "tapas-epoch=00-val_loss=0.38.ckpt"


class TapasLightningModule(pl.LightningModule):
    def __init__(self, model, learning_rate=5e-5):
        """
        Args:
            model: A pretrained TAPAS model.
            learning_rate: The learning rate for the optimizer.
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

    def forward(self, batch):
        # Forward pass that expects a batch dictionary with all required keys.
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=batch["labels"],
            numeric_values=batch["numeric_values"],
            numeric_values_scale=batch["numeric_values_scale"],
            aggregation_labels=batch["aggregation_labels"],
        )

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss = outputs.loss
        # Log training loss on both step and epoch levels.
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss = outputs.loss
        # Log validation loss only at the epoch level.
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):  # Add this method
        outputs = self.forward(batch)
        loss = outputs.loss
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        return optimizer


def load_tapas_config() -> PretrainedConfig:
    return TapasConfig.from_pretrained(tapas_wikisql_name)


def load_tapas_tokenizer() -> TapasTokenizer:
    return TapasTokenizer.from_pretrained(tapas_wikisql_name)


def load_tapas_model() -> TapasLightningModule:
    base_model = TapasForQuestionAnswering.from_pretrained(
        tapas_base_name,
        config=load_tapas_config(),
    ).to(device)  # type: ignore

    return TapasLightningModule.load_from_checkpoint(
        str(checkpoint),
        model=base_model,
    ).to(device)
