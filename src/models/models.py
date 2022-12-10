import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from src.models.resnet import resnet50


class SimpleConvNet_100x100(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 25 * 25, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 25 * 25)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleConvNet_224x224(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# TODO
# class PretrainedEfficientNet(nn.Module):
#     def __init__(self):
# super().__init__()

#         self.model = EfficientNet.from_pretrained("efficientnet-b0")
#         self.fc = nn.Linear(1280, 1)

#     def forward(self, x):
#         x = self.model(x)
#         x = x.view(-1, 1280)
#         x = self.fc(x)
#         return x


class PretrainedResnetVGGFace2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(pretrained=False, remove_classifier=True)

        state_dict = torch.load("models/flr_r50_vgg_face.pth", map_location="cpu")["state_dict"]

        # drop fc.weight and fc.bias
        # state_dict.pop("fc.weight")
        # state_dict.pop("fc.bias")

        # remove "module.base_net." prefix
        state_dict = {k.replace("module.base_net.", ""): v for k, v in state_dict.items()}

        # remove "projection_net." keys
        state_dict = {k: v for k, v in state_dict.items() if "module." not in k}

        # load pretrained weights
        self.model.load_state_dict(state_dict)

        self.fc = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.model(x)
        x = torch.relu(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = PretrainedResnetVGGFace2()

    # mock input data
    x = torch.randn(1, 3, 224, 224)

    output = model.forward(x)

    print(output.shape)
    print(output.std())
    print(output.mean())
