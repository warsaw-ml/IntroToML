from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms

from src.data.face_age_dataset import FaceAgeDataset


class FaceAgeDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        img_size: Tuple[int, int] = (224, 224),
        label_clipping=(0, 80),
        normalize_labels=True,
        train_val_test_split: Tuple[int, int, int] = (19708, 2000, 2000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters()

        # data transformations
        # self.imagenet_mean = (0.485, 0.456, 0.406)
        # self.imagenet_std = (0.229, 0.224, 0.225)
        # self.transform = [transforms.Normalize(self.imagenet_mean, self.imagenet_std)]

        # self.mean = (0.5961, 0.4564, 0.3906)
        # self.std = (0.2587, 0.2307, 0.2262)
        # self.transform = [transforms.Normalize(self.mean, self.std)]

        self.transform = None

        # TODO: add augmentations to transforms

        # TODO: Undersample of Oversample the data

        # TODO: Add data stratification

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = FaceAgeDataset(
                data_dir=self.hparams.data_dir,
                img_size=self.hparams.img_size,
                label_clipping=self.hparams.label_clipping,
                normalize_labels=self.hparams.normalize_labels,
                transform=self.transform,
            )
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
