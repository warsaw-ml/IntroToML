import pytorch_lightning as pl
import torch
from torchvision import transforms
from data.face_age_dataset import FaceAgeDataset
from PIL import Image

from train import FaceAgeModule

class Predict():
    def __init__(self):
        self.model = FaceAgeModule.load_from_checkpoint("models/best-checkpoint.ckpt")
        self.model.eval()
        self.model.freeze()
        self.convert_tensor = transforms.ToTensor()
    def predict(self, image) -> float:
        image = image.resize((100, 100), Image.ANTIALIAS)
        img = self.convert_tensor(image)
        img = img.reshape(1, 3, 100, 100)
        prediction = self.model.forward(img)
        return prediction.item()*80


def main():

    # standardize img before passing to model
    # self.imagenet_mean = (0.485, 0.456, 0.406)
    # self.imagenet_std = (0.229, 0.224, 0.225)
    # self.transform = [transforms.Normalize(self.imagenet_mean, self.imagenet_std)]

    model = FaceAgeModule.load_from_checkpoint("models/best-checkpoint.ckpt")
    model.eval()
    model.freeze()
    data = FaceAgeDataset()
    img, y, idx = data[0]
    print(y)
    img = img.reshape(1, 3, 100, 100)
    print(img.shape)
    prediction = model.forward(img)
    print(prediction.item()*80)


if __name__ == "__main__":
    main()
