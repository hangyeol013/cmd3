from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from pytorch_lightning import LightningModule
import torchmetrics

from .vggish import VGGish

NUM_CLASSES = 3


class CMD3Audio(LightningModule):
    def __init__(
        self,
        model_name: str,
        model_path: str,
        feature_extraction: bool,
        optimizer: str,
        learning_rate: float,
        weight_decay: float,
        momentum: float,
    ):
        super(CMD3Audio, self).__init__()
        self.optimizer = optimizer
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum

        self.num_classes = NUM_CLASSES
        self.feature_extraction = feature_extraction

        self.model = VGGish(model_path)
        self.model.embeddings[4] = nn.Linear(4096, self.num_classes)

        self.criterion = CrossEntropyLoss()
        metric_params = {"num_classes": self.num_classes, "average": "macro"}

        self.train_accuracy = torchmetrics.Accuracy(**metric_params)
        self.val_accuracy = torchmetrics.Accuracy(**metric_params)

        self.train_precision = torchmetrics.Precision(**metric_params)
        self.val_precision = torchmetrics.Precision(**metric_params)

        self.train_recall = torchmetrics.Recall(**metric_params)
        self.val_recall = torchmetrics.Recall(**metric_params)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.model(x)
            return x

    def shared_log_step(
        self,
        mode: str,
        loss: torch.Tensor,
        accuracy: torch.Tensor,
        precision: torch.Tensor,
        recall: torch.Tensor,
    ):
        on_step = True if mode == 'train' else False
        self.log(
            f"{mode}/loss",
            loss,
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{mode}/accuracy",
            accuracy,
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{mode}/precision",
            precision,
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f"{mode}/recall",
            recall,
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

    def _shared_eval_step(
        self, batch: Tuple[torch.Tensor], batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluation step."""
        targets = batch["label"]
        inputs = batch["audio"]
        outputs = self.model(inputs).squeeze()
        loss = self.criterion(outputs, targets)
        preds = torch.argmax(outputs, dim=1)

        return loss, preds, targets

    def training_step(self, batch: torch.Tensor, batch_idx: torch.Tensor):
        """Training loop."""
        loss, preds, targets = self._shared_eval_step(batch, batch_idx)
        accuracy = self.train_accuracy(preds, targets)
        precision = self.train_precision(preds, targets)
        recall = self.train_recall(preds, targets)
        self._shared_log_step("train", loss, accuracy, precision, recall)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: torch.Tensor, batch_idx: torch.Tensor):
        """Validation loop."""
        loss, preds, targets = self._shared_eval_step(batch, batch_idx)
        accuracy = self.val_accuracy(preds, targets)
        precision = self.val_precision(preds, targets)
        recall = self.val_recall(preds, targets)
        self._shared_log_step("val", loss, accuracy, precision, recall)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: torch.Tensor, batch_idx: torch.Tensor):
        """Test loop."""
        targets = batch["label"]
        inputs = batch["audio"]

        clip_ids = batch["id"]
        preds = self.model(inputs).squeeze()
        loss = self.criterion(preds, targets)

        return {
            "loss": loss,
            "preds": preds,
            "targets": targets,
            "id": clip_ids,
        }

    def test_epoch_end(self, outputs: torch.Tensor):
        self.test_outputs = outputs
        return outputs

    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        """Define optimizers and LR schedulers."""
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                weight_decay=self.weight_decay,
                momentum=self.momentum,
                lr=self.lr,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", 0.1, verbose=False
            )

        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), self.lr)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda x: x  # Identity, only to monitor
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss",
        }