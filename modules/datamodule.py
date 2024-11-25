import os

from datasets.retinadataset import RetinaDataset
from torch.utils.data import DataLoader
from augmentation.augmentationreferral import (
    train_transform,
    valid_transform,
)
from lightning.pytorch import LightningDataModule


class DataModule(LightningDataModule):

    def __init__(
            self,
            path: str,
            input_size: int,
            batch_size: int,
            workers: int
        ) -> None:
        super().__init__()
        self.path = path
        self.input_size = input_size
        self.batch_size = batch_size
        self.workers = workers

    def setup(self, stage: str) -> None:
        train_dataset = RetinaDataset(
            root=os.path.join(self.path, 'train'),
            transform=train_transform(size=self.input_size)
        )
        self.train_loader = DataLoader(dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers
        )
        val_dataset = RetinaDataset(
            root=os.path.join(self.path, 'validation'),
            transform=valid_transform(size=self.input_size)
        )
        self.val_loader = DataLoader(dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers
        )

    def train_dataloader(self) -> DataLoader:
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        return self.val_loader
