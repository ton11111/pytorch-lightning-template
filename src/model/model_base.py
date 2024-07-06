from typing import Dict, Tuple

import torch
from torch import nn
from torch.optim import Adam
from lightning import LightningModule

from common.enums import ModelLoss
from utils.model_utils import get_metrics


class ModelBase(LightningModule):
    def __init__(self, **kwargs):
        super(ModelBase, self).__init__()
        self.save_hyperparameters()
        self.train_metrics, self.valid_metrics, self.test_metrics = get_metrics(
            task=self.hparams.task, num_classes=self.hparams.num_labels
        )

        self.criterion = nn.CrossEntropyLoss()

    def _get_pred_and_target(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        x, y = batch
        return self(x), y

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor, loss_type: ModelLoss) -> torch.Tensor:
        loss = self.criterion(input=pred, target=target)
        self.log_dict({loss_type.value: loss})

        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        pred, target = self._get_pred_and_target(batch)
        self.train_metrics.update(pred, target)

        return self._compute_loss(pred, target, ModelLoss.TRAIN_LOSS)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        pred, target = self._get_pred_and_target(batch)
        self.valid_metrics.update(pred, target)

        return self._compute_loss(pred, target, ModelLoss.VALID_LOSS)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        pred, target = self._get_pred_and_target(batch)
        self.test_metrics.update(pred, target)

        return self._compute_loss(pred, target, ModelLoss.TEST_LOSS)

    def on_train_epoch_end(self) -> None:
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.valid_metrics.compute())
        self.valid_metrics.reset()

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def configure_optimizers(self) -> Adam:
        return Adam(self.parameters(), lr=self.hparams.lr)
