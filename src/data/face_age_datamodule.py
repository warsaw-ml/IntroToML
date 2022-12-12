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
                
                if count_occurences[label] < 50:
                    if count_occurences[label] %2 == 0:
                        test_data.append((img, tensor))
                    else:
                        val_data.append((img, tensor))
                elif count_occurences[label] <= MAX_DATA_CLASS:
                    new_data.append((img, tensor))
                
            print(f"Val data size: {val_data.__sizeof__()}")
            print(f"Test data size: {test_data.__sizeof__()}")
            print(f"Train data before agumentation: {new_data.__sizeof__()}")
            
                
            #data argumentation on new_data(train data)
            mirror_data = list()
            for img, tensor in new_data:
                if count_occurences[tensor.item()] > MAX_DATA_CLASS:
                    continue
                count_occurences[tensor.item()] += 1
                copy_img = transforms.functional.to_pil_image(img)
                mirror_img = ImageOps.mirror(copy_img)
                mirror_data.append((transforms.functional.to_tensor(mirror_img), tensor))

            new_data.extend(mirror_data)
            #rotate data by random degrees from -30 to 30
            rotate_data = list()
            rotater = transforms.RandomRotation(degrees=(-30,30))

            for img, tensor in new_data:
                if count_occurences[tensor.item()] > MAX_DATA_CLASS:
                    continue
                count_occurences[tensor.item()] += 2
                rotated_tensors = [(rotater(img), tensor) for _ in range(2)]
                rotate_data.append(rotated_tensors[0])
                rotate_data.append(rotated_tensors[1])
                
            new_data.extend(rotate_data)

            #color jitter
            jitter = transforms.ColorJitter(brightness=.5, hue=.3)
            jitter_data = list()
            for img, tensor in new_data:
                if count_occurences[tensor.item()] > MAX_DATA_CLASS:
                    continue
                count_occurences[tensor.item()] += 2
                jitter_tensors = [(jitter(img), tensor) for _ in range(2)]
                jitter_data.append(jitter_tensors[0])
                jitter_data.append(jitter_tensors[1])
            new_data.extend(jitter_data)
            
            #random perspective  
            perspective = transforms.RandomPerspective(distortion_scale=0.5, p=1)
            perspective_data = list()
            for img, tensor in new_data:
                if count_occurences[tensor.item()] > MAX_DATA_CLASS:
                    continue
                count_occurences[tensor.item()] += 2
                perspective_tensors = [(perspective(img), tensor) for _ in range(2)]
                perspective_data.append(perspective_tensors[0])
                perspective_data.append(perspective_tensors[1])
            new_data.extend(perspective_data)


            #random posterize
            #val_data - validation
            #test_data - test
            #new_data - train
            print("Data per class:")            
            print((count_occurences))

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
