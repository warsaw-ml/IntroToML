import torch
import pytorch_lightning as pl

from train import LitModule
from face_age_datamodule import FaceAgeDataset

def main():
    model = LitModule.load_from_checkpoint("checkpoints/last-v2.ckpt")
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