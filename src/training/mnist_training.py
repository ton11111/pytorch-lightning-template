from argparse import ArgumentParser

from trainer.mnist_trainer import MNISTTrainer


def get_hparams():
    hparams = ArgumentParser()
    hparams.add_argument("--lr", default=0.01)
    hparams.add_argument("--batch_size", default=32)
    hparams.add_argument("--max_epochs", default=3)
    hparams.add_argument("--patience", default=3)
    hparams.add_argument("--refresh_rate", default=20)

    return hparams.parse_args()


if __name__ == '__main__':
    trainer = MNISTTrainer(hparams=get_hparams())
    trainer.train()
