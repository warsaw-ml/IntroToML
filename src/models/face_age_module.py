from typing import Any, List

import pytorch_lightning as pl
import torch
from torchmetrics import MeanAbsoluteError as MAE
from torchmetrics import MeanMetric, MinMetric
from src.models import models


class FaceAgeModule(pl.LightningModule):
    def __init__(self, net: str = "EffNet_224x224", rescale_age_by: int = 80.0):
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

        # metric objects for calculating and averaging maeuracy across batches
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
        return self.net(x)

    def predict(self, batch):
        x, y = batch
        preds = self.forward(x)
        preds = preds.clip(0, 1)
        return preds

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_mae_best doesn't store mae from these checks
        self.val_mae_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)

        preds = preds.clip(0, 1)

        # rescale prediction from [0-1] to [0-80]
        if self.hparams.rescale_age_by:
            preds = preds * self.hparams.rescale_age_by
            y = y * self.hparams.rescale_age_by
            preds = preds.clip(1, self.hparams.rescale_age_by)

        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        self.train_loss(loss)
        self.train_mae(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        self.val_loss(loss)
        self.val_mae(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, outputs: List[Any]):
        self.val_mae_best(self.val_mae.compute())
        self.log("val/mae_best", self.val_mae_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        self.test_loss(loss)
        self.test_mae(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)
