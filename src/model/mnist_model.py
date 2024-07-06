import torch
from torch import nn

from model.model_base import ModelBase


class MNISTModel(ModelBase):
    def __init__(self, **kwargs):
        super(MNISTModel, self).__init__(task="multiclass", num_labels=10, **kwargs)

        self.layers = nn.Sequential(
            # Conv Layers
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            # FC Layers
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)
