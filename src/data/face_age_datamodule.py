from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.face_age_dataset_from_path import FaceAgeDatasetFromPath


class FaceAgeDataModule(LightningDataModule):
    """
    LightningDataModule for our FaceAge dataset.

    A DataModule implements 4 key methods:
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader

    This allows to share a full dataset without explaining how to download,
    split, transform and process the data.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        img_size: tuple = (224, 224),
        normalize_age_by: int = 80,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """
        Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        """
        if not self.data_train and not self.data_val and not self.data_test:

            # image transformations
            transform_list = []
            transform_list.append(transforms.Resize(self.hparams.img_size))
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
            transform = transforms.Compose(transform_list)

            self.data_train = FaceAgeDatasetFromPath(
                img_dir="data/face_age_dataset/train",
                normalize_age_by=self.hparams.normalize_age_by,
                transform=transform,
            )
            self.data_val = FaceAgeDatasetFromPath(
                img_dir="data/face_age_dataset/val",
                normalize_age_by=self.hparams.normalize_age_by,
                transform=transform,
            )
            self.data_test = FaceAgeDatasetFromPath(
                img_dir="data/face_age_dataset/test",
                normalize_age_by=self.hparams.normalize_age_by,
                transform=transform,
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
