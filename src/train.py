import pyrootutils

root = pyrootutils.setup_root(__file__, pythonpath=True, cwd=True)

from typing import Any, List

import pytorch_lightning as pl
import torch
from torchmetrics import MeanAbsoluteError as MAE
from torchmetrics import MeanMetric, MinMetric

from src.models import models
from src.data.face_age_datamodule import FaceAgeDataModule


class LitModel(pl.LightningModule):
    def __init__(self, rescale_labels: bool = False):
        super().__init__()

        # this line allows to init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        # declare network with 1 output unit with rgb image input 100x100
        self.net = models.SimpleConvNet_100x100()
        # self.net = models.SimpleConvNet_224x224()
        # self.net = models.PretrainedResnetVGGFace2()
        # self.net = models.PretrainedEfficientNet()

        # loss function
        # self.criterion = torch.nn.MSELoss()
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

    def predict(self, batch):
        x, y = batch
        preds = self.forward(x)
        return preds

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_mae_best doesn't store mae from these checks
        self.val_mae_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)

        # rescale prediction from [0-1] to [0-80]
        if self.hparams.rescale_labels:
            preds = preds * 80
            y = y * 80

        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        self.train_loss(loss)
        self.train_mae(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        pass

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

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


def main():
    pl.seed_everything(42)

    data_dir = root / "data"
    logs_dir = root / "logs"

    datamodule = FaceAgeDataModule(
        data_dir=data_dir,
        num_workers=0,
        batch_size=64,
        img_size=(100, 100),
        # img_size=(224, 224),
        # label_clipping=None,
        label_clipping=(0, 80),
        normalize_labels=False,
        # normalize_labels=True,
    )

    model = LitModel(rescale_labels=False)

    callbacks = []
    loggers = []

    # model checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/mae",
        dirpath=logs_dir / "checkpoints",
        save_top_k=1,
        save_last=True,
        mode="min",
        save_weights_only=True,
        filename="best-checkpoint",
    )
    callbacks.append(checkpoint_callback)

    # experiment logger
    # wandb_logger = pl.loggers.WandbLogger(
    #     project="face-age",
    #     name="100x100+convnet",
    #     save_dir=logs_dir,
    # )
    # loggers.append(wandb_logger)

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=logs_dir,
        accelerator="cpu",
        max_epochs=10,
        val_check_interval=0.25,
    )

    trainer.validate(model=model, datamodule=datamodule)

    trainer.fit(model=model, datamodule=datamodule)

    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
