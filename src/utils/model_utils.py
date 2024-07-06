from typing import Tuple, Optional, Literal

from torchmetrics import MetricCollection, Precision, Recall, F1Score, AUROC
from torchmetrics.classification import (
    BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryRecall,
    MultilabelAccuracy, MultilabelF1Score, MultilabelPrecision, MultilabelRecall, MultilabelAUROC
)

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from common.enums import ModelLoss


def get_metrics(
    task: Literal["binary", "multiclass", "multilabel"],
    num_classes: Optional[int] = None
) -> Tuple[MetricCollection]:
    metrics = MetricCollection([
        F1Score(task=task, num_classes=num_classes),
        Precision(task=task, num_classes=num_classes),
        Recall(task=task, num_classes=num_classes),
        AUROC(task=task, num_classes=num_classes),
    ])

    train_metrics = metrics.clone(prefix='train_')
    valid_metrics = metrics.clone(prefix='valid_')
    test_metrics = metrics.clone(prefix='test_')

    return train_metrics, valid_metrics, test_metrics


def get_model_checkpoint(filename: str) -> ModelCheckpoint:
    return ModelCheckpoint(
        monitor=ModelLoss.VALID_LOSS.value,
        filename=filename
    )


def get_early_stopping(patience: int) -> EarlyStopping:
    return EarlyStopping(
        monitor=ModelLoss.VALID_LOSS.value,
        patience=patience
    )
