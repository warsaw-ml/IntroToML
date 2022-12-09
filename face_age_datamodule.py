from typing import Optional, Tuple
import os
from PIL import Image
from tqdm import tqdm

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms


class FaceAgeDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir="data/"):
        self.data_dir = data_dir
        self.dataset_dir = os.path.join(self.data_dir, "UTKFace")
        self.img_dir = os.path.join(self.dataset_dir, "images")

        self.images = None
        self.labels = None

        self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(100, 100))])

        self.load_data()

        self.mean = torch.mean(self.images)
        self.std = torch.std(self.images)

    def load_data(self):
        # load data from torch saves if it was already processed
        images_tensor_path = os.path.join(self.dataset_dir, "images.pt")
        labels_tensor_path = os.path.join(self.dataset_dir, "labels.pt")
        if os.path.exists(images_tensor_path) and os.path.exists(labels_tensor_path):
            self.images = torch.load(images_tensor_path)
            self.labels = torch.load(labels_tensor_path)
            return

        # read filenames and ages from folder
        paths = []
        ages = []
        for filename in os.listdir(self.img_dir):
            if filename.split(".")[-1] == "jpg":
                paths.append(os.path.join(self.img_dir, filename))
                ages.append(int(filename.split("_")[0]))

        # load images
        images = [Image.open(path) for path in tqdm(paths, desc="Loading images...")]

        # convert to tensors
        images = torch.stack([self.transforms(img) for img in tqdm(images, desc="Converting images to tensors...")])
        labels = torch.stack(
            [torch.tensor([age], dtype=torch.long) for age in tqdm(ages, desc="Converting labels to tensors...")]
        )

        self.images = images
        self.labels = labels

        # save tensors
        torch.save(self.images, images_tensor_path)
        torch.save(self.labels, labels_tensor_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class FaceAgeDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (20708, 2990, 10),
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.mean = 0.4810
        self.std = 0.2544

        # data transformations
        self.transforms = transforms.Compose([transforms.Normalize((self.mean,), (self.std,))])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = FaceAgeDataset(data_dir=self.hparams.data_dir)
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


if __name__ == "__main__":
    dm = FaceAgeDataModule()
    dm.setup()
    for i in dm.train_dataloader():
        print(i[0].shape, i[1].shape)
        break
