from enum import Enum


class ModelLoss(Enum):
    TRAIN_LOSS = "train_loss"
    VALID_LOSS = "valid_loss"
    TEST_LOSS = "test_loss"
