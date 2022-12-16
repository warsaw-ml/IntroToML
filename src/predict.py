import pytorch_lightning as pl
import torch
from torchvision import transforms
from src.data.face_age_dataset import FaceAgeDataset
from PIL import Image

from src.train import FaceAgeModule

class Predict():
    def __init__(self):
        self.model = FaceAgeModule.load_from_checkpoint("models/best-checkpoint.ckpt")
        self.model.eval()
        self.model.freeze()
        transform_list = [
            transforms.ToTensor(),
            transforms.Resize((100, 100)),
            transforms.Resize((224, 224)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
        self.transform = transforms.Compose(transform_list)
    def predict(self, image) -> float:
        img = self.transform(image)
        img = img.reshape(1, 3, 224, 224)
        prediction = self.model.forward(img)
        prediction_rescaled = prediction * 80
        prediction_rescaled = prediction_rescaled.clip(1, 80)
        return prediction_rescaled.item()
