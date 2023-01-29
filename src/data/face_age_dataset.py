import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class FaceAgeDataset(Dataset):
    def __init__(
        self,
        data_dir="data/",
        img_size=(100, 100),
        label_clipping=(0, 80),
        normalize_labels=False,
        transform=None,
    ):
        self.data_dir = Path(data_dir)
        self.dataset_dir = self.data_dir / "archive/UTKFace"
        self.img_dir = self.dataset_dir

        self.img_size = img_size
        self.label_clipping = label_clipping
        self.normalize_labels = normalize_labels

        # transformations applied when returning datapoints
        base_transform = [transforms.ToTensor(), transforms.Resize(size=self.img_size)]
        self.transform = base_transform + transform if transform else base_transform
        self.transform = transforms.Compose(self.transform)

        # setup img paths and labels
        self.img_paths = None
        self.labels = None
        self.load_data()

    def load_data(self):
        """Read image names and labels from folder."""
        self.img_paths = []
        self.labels = []
        for filename in os.listdir(self.img_dir):
            if filename.split(".")[-1] == "jpg":
                self.img_paths.append(os.path.join(self.img_dir, filename))
                self.labels.append(int(filename.split("_")[0]))  # age is element of filename

        self.labels = torch.LongTensor(self.labels)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        label = self.labels[idx].unsqueeze(0)

        if self.transform:
            img = self.transform(img)

        if self.label_clipping:
            label = label.clip(min=self.label_clipping[0], max=self.label_clipping[1])

        if self.normalize_labels:
            label = label / 80

        return img, label, idx


if __name__ == "__main__":
    import pyrootutils

    data_dir = pyrootutils.find_root() / "data/"
    dataset = FaceAgeDataset(data_dir=data_dir, img_size=(100, 100))
    x, y = dataset[0]

    labels = dataset.labels.float()
    print("stats:", labels.min(), labels.max(), labels.mean(), labels.std())
