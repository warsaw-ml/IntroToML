import os

import torch
from torch.utils.data import Dataset


class FaceAgeDataset(Dataset):
    def __init__(
        self,
        img_dir="data/face_age_dataset/train",
        normalize_age_by=80,
        transform=None,
    ):
        if type(img_dir) == str:
            self.img_dirs = [img_dir]
        else:
            self.img_dirs = img_dir

        if type(self.img_dirs) != list:
            raise TypeError("img_dir must be a string or list of strings")

        self.normalize_age_by = normalize_age_by

        self.transform = transform

        self.img_paths = None
        self.labels = None

        self.load_data()

    def load_data(self):
        """Read image names and labels from folder."""
        self.img_paths = []
        self.labels = []

        for img_dir in self.img_dirs:
            assert os.path.exists(img_dir), f"Image directory {img_dir} does not exist."

        for img_dir in self.img_dirs:
            for filename in os.listdir(img_dir):
                if filename.split(".")[-1] == "pt":
                    self.img_paths.append(os.path.join(img_dir, filename))
                    self.labels.append(int(filename.split("_")[-1].split(".")[0]))  # age is element of filename

        assert self.img_paths, f"No images with '.pt' extension found in {self.img_dirs}"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = torch.load(self.img_paths[idx])
        label = self.labels[idx]

        img = torch.FloatTensor(img)
        label = torch.FloatTensor([label])

        if self.normalize_age_by:
            label = label / self.normalize_age_by

        if self.transform:
            img = self.transform(img)

        return img, label


if __name__ == "__main__":
    dataset = FaceAgeDataset(img_dir="data/face_age_dataset/val", normalize_age_by=1)

    x, y = dataset[0]
    print(x.shape, y.shape)

    data_x = torch.stack([x for x, y in dataset])
    data_y = torch.stack([y for x, y in dataset])
    print(data_x.shape, data_y.shape)

    data_y = data_y.float()
    print(data_y.min(), data_y.max(), data_y.mean(), data_y.std())
    print(data_x.mean())
