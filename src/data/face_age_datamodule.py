from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from torchvision import transforms
from src.data.face_age_dataset import FaceAgeDataset, FaceAgeDatasetAugmented
from PIL import Image, ImageOps
from torchvision.transforms.functional import rotate

MAX_DATA_CLASS = 700

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
            
            data = list(dataset)

            #cut data to have MAX_DATA_CLASS per class
            new_data = list()
            test_data = list()
            val_data = list()
            count_occurences = dict()
            for img, tensor in data:
                label = tensor.item()
                if label not in count_occurences:
                    count_occurences[label] = 1
                else:
                    count_occurences[label] += 1
                
                if count_occurences[label] < 25:
                    test_data.append((img, tensor))
                elif count_occurences[label] < 50:
                    val_data.append((img, tensor))
                elif count_occurences[label] <= MAX_DATA_CLASS:
                    new_data.append((img, tensor))
                
            
            
            #data argumentation on new_data(train data)
            mirror_data = list()
            for img, tensor in new_data:
                copy_img = transforms.functional.to_pil_image(img)
                mirror_img = ImageOps.mirror(copy_img)
                mirror_data.append((transforms.functional.to_tensor(mirror_img), tensor))
            
            #rotate data by 25 degrees
            rotate_data = list()
            for img, tensor in new_data:
                rotated_tensor = rotate(img, 25)
                rotate_data.append((rotated_tensor, tensor))
                rotated_tensor = rotate(img, -25)
                rotate_data.append((rotated_tensor, tensor))

            for img, tensor in mirror_data:
                rotated_tensor = rotate(img, 25)
                rotate_data.append((rotated_tensor, tensor))
                rotated_tensor = rotate(img, -25)
                rotate_data.append((rotated_tensor, tensor))

            #val_data - validation
            #test_data - test
            #new_data - train
            new_data.append(mirror_data)
            new_data.append(rotate_data)

            self.data_test = FaceAgeDatasetAugmented(test_data)
            self.data_val = FaceAgeDatasetAugmented(val_data)
            self.data_train = FaceAgeDatasetAugmented(new_data)

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
