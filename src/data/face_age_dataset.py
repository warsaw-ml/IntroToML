import os
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import transforms

MAX_AGE = 80
class FaceAgeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir="data/",
        img_size=(224, 224),
        label_clipping=(0, MAX_AGE),
        normalize_labels=True,
        transform=None,
    ):
        self.data_dir = Path(data_dir)
        self.dataset_dir = self.data_dir / "UTKFace"
        self.img_dir = self.dataset_dir / "images"

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
            label = label / MAX_AGE

        return img, label

class FaceAgeDatasetAugmented(torch.utils.data.Dataset):
    def __init__(self, dataset, augmentations):
        self.dataset = dataset
        self.augmentations = augmentations
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]        
        return img, label

if __name__ == "__main__":
    import pyrootutils

    # imagenet_mean = (0.485, 0.456, 0.406)
    # imagenet_std = (0.229, 0.224, 0.225)
    # transform = [transforms.Normalize(mean=imagenet_mean, std=imagenet_std)]
    transform = None

    data_dir = pyrootutils.find_root() / "data/"
    dataset = FaceAgeDataset(data_dir=data_dir, transform=transform, img_size=(100, 100))
    x, y = dataset[0]

    labels = dataset.labels.float()
    print("stats:", labels.min(), labels.max(), labels.mean(), labels.std())

    # calculate MAE against tensor full of labels.mean()
    labels_mean = torch.full_like(labels, labels.mean())
    mae = torch.abs(labels - labels_mean).mean()
    print("MAE:", mae)

    # from tqdm import tqdm
    # for x in tqdm(dataset):
    #     pass
