import pytorch_lightning as pl
import torch
from face_age_datamodule import FaceAgeDataset

from train import LitModel


def main():

    # standardize img before passing to model
    # self.imagenet_mean = (0.485, 0.456, 0.406)
    # self.imagenet_std = (0.229, 0.224, 0.225)
    # self.transform = [transforms.Normalize(self.imagenet_mean, self.imagenet_std)]

    model = LitModel.load_from_checkpoint("checkpoints/last.ckpt")
    model.eval()
    model.freeze()
    data = FaceAgeDataset()
    img, y = data[0]
    print(y)
    img = img.reshape(1, 3, 100, 100)
    print(img.shape)
    prediction = model.forward(img)
    print(prediction)


if __name__ == "__main__":
    main()
