from typing import Any, List

import pytorch_lightning as pl
import torch
from torchmetrics import MeanAbsoluteError as MAE
from torchmetrics import MeanMetric, MinMetric
from src.models import models


class FaceAgeModule(pl.LightningModule):
    """
    FaceAgeModule is a PyTorch Lightning module for training a model to predict the age of a face in an image.
    It uses a pre-trained model (either SimpleConvNet_100x100, SimpleConvNet_224x224, or PretrainedEfficientNet)
    and fine-tunes it on the input dataset. The module has several methods for training, validation, and testing,
    as well as for logging metrics such as mean absolute error (MAE) and loss.
    """

    def __init__(self, net: str = "EffNet_224x224", rescale_age_by: int = 80.0):
        """
        Initializes the FaceAgeModule with the specified rescale value for the labels.
        The rescale value is used to convert the predicted age value from a range of [0,1] to [0, rescale_age_by].
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters()

        # architecture
        if net == "SimpleConvNet_100x100":
            self.net = models.SimpleConvNet_100x100()
        elif net == "SimpleConvNet_224x224":
            self.net = models.SimpleConvNet_224x224()
        elif net == "EffNet_224x224":
            self.net = models.PretrainedEfficientNet()
        else:
            raise ValueError(f"Unknown net: {net}")

        # loss function
        self.criterion = torch.nn.MSELoss()
        # self.criterion = torch.nn.SmoothL1Loss()

        # metric objects for calculating and averaging MAE across batches
        self.train_mae = MAE()
        self.val_mae = MAE()
        self.test_mae = MAE()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation maeuracy
        self.val_mae_best = MinMetric()

    def forward(self, x: torch.Tensor):
        """
        The forward method is called during training, validation, and testing to make predictions on the input data.
        It takes in a tensor 'x' and returns the model's predictions.
        """
        return self.net(x)

    def predict(self, batch):
        """
        The predict method is called to make predictions on a single batch of data.
        It takes in a batch of data as input and returns the model's predictions.
        """
        x, y = batch
        preds = self.forward(x)
        preds = preds.clip(0, 1)
        return preds

    def on_train_start(self):
        """
        The on_train_start method is called before the training process begins.
        It resets the val_mae_best metric to ensure that it doesn't store any values from the validation step sanity checks.
        """
        self.val_mae_best.reset()

    def model_step(self, batch: Any):
        """
        The model_step method is called during training, validation, and testing to make predictions and calculate loss.
        It takes in a batch of input data and returns the calculated loss and predictions.
        """
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)

        # clip prediction to [0-1]
        preds = preds.clip(0, 1)

        # rescale prediction from [0-1] to [0-80]
        if self.hparams.rescale_age_by:
            preds = preds * self.hparams.rescale_age_by
            y = y * self.hparams.rescale_age_by
            preds = preds.clip(1, self.hparams.rescale_age_by)

        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        """
        The training_step method is called during training to calculate the loss and update the model's parameters.
        It also logs the loss and mean absolute error metric to track training progress.
        """
        loss, preds, targets = self.model_step(batch)

        self.train_loss(loss)
        self.train_mae(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        """
        The validation_step method is called during validation to calculate the loss and update metrics.
        It also logs the loss and mean absolute error metric to track validation progress.
        """
        loss, preds, targets = self.model_step(batch)

        self.val_loss(loss)
        self.val_mae(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, outputs: List[Any]):
        """
        The validation_epoch_end method is called at the end of each validation epoch.
        It updates the best mean absolute error metric and logs it.
        """
        self.val_mae_best(self.val_mae.compute())
        self.log("val/mae_best", self.val_mae_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        """
        The test_step method is called during testing to calculate the loss and update metrics.
        It also logs the loss and mean absolute error metric to track data, and returns the calculated loss, predictions, and targets.
        """
        loss, preds, targets = self.model_step(batch)

        self.test_loss(loss)
        self.test_mae(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """
        The configure_optimizers method is used to configure the optimizers used for training.
        This method should return a single optimizer or a list of optimizers.
        In this implementation, it returns an instance of the Adam optimizer with a learning rate of 0.01.
        """
        return torch.optim.Adam(self.parameters(), lr=0.01)
