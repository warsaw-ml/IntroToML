import pytorch_lightning as pl
from face_age_datamodule import FaceAgeDataModule

from typing import Any, List

import torch
from torchmetrics import MaxMetric, MeanMetric, MinMetric

from torchmetrics import MeanAbsoluteError as MAE

import torch.nn as nn

import torch.nn.functional as F


class Net(nn.Module):
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


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # this line allows to maeess init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        # declare network with 1 output unit with rgb image input 100x100
        self.net = Net()

        # loss function
        # self.criterion = torch.nn.MSELoss
        self.criterion = torch.nn.SmoothL1Loss()

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

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_mae_best doesn't store maeuracy from these checks
        self.val_mae_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        preds = self.forward(x)
        # preds = preds * 90 + 1
        loss = self.criterion(preds, y)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        self.train_loss(loss)
        self.train_mae(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        self.val_loss(loss)
        self.val_mae(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        mae = self.val_mae.compute()  # get current val mae
        self.val_mae_best(mae)  # update best so far val mae
        # log `val_mae_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/mae_best", self.val_mae_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        self.test_loss(loss)
        self.test_mae(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


def main():
    model = LitModel()
    datamodule = FaceAgeDataModule(num_workers=4)

    # model ckpt callback
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/mae_best",
        dirpath="checkpoints",
        filename="model-{epoch:02d}-{val/mae_best:.2f}",
        save_top_k=1,
        save_last=True,
        mode="min",
        save_weights_only=True,
    )
    callbacks = [ckpt_callback]

    # get wandb logger
    logger = pl.loggers.WandbLogger(project="face-age", name="face-age", save_dir="logs")

    trainer = pl.Trainer(accelerator="cpu", max_epochs=10, callbacks=callbacks, val_check_interval=0.5, logger=logger)

    trainer.fit(model=model, datamodule=datamodule)

    trainer.test(model=model, datamodule=datamodule)

    # LitModel.load_from_checkpoint(ckpt_callback.best_model_path)

    # save ckpt to onnx
    # trainer.to_onnx(model=model, ckpt_path=...)

    # trainer.predict(model=model, datamodule=datamodule, ckpt_path=...)


if __name__ == "__main__":
    main()
