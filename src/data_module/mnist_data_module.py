from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from common.logger import logger


class MNISTDataModule(LightningDataModule):

    def __init__(self, batch_size: int, data_dir: str = "./data") -> None:
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def prepare_data(self) -> None:
        logger.info("Downloading MNIST data")
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str) -> None:
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.train_dataset, self.val_dataset = random_split(mnist_full, [0.7, 0.3])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_dataset = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
