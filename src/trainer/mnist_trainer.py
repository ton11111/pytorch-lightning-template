from lightning import Trainer
from lightning.pytorch.callbacks import TQDMProgressBar

from utils.model_utils import get_model_checkpoint, get_early_stopping
from model.mnist_model import MNISTModel
from data_module.mnist_data_module import MNISTDataModule


class MNISTTrainer:

    def __init__(self, hparams):
        self.hparams = hparams

    def train(self):
        model = MNISTModel(**vars(self.hparams))
        data_module = MNISTDataModule(batch_size=32)

        trainer = Trainer(
            max_epochs=self.hparams.max_epochs,
            callbacks=[
                get_model_checkpoint(filename="mnist_model"),
                get_early_stopping(patience=self.hparams.patience),
                TQDMProgressBar(refresh_rate=self.hparams.refresh_rate)
            ]
        )
        trainer.fit(model=model, datamodule=data_module)
        trainer.test(datamodule=data_module)
