import pyrootutils

root = pyrootutils.setup_root(__file__, pythonpath=True, cwd=True)

from typing import Any, List

import pytorch_lightning as pl
import torch
import wandb
from torchmetrics import MeanAbsoluteError as MAE
from torchmetrics import MeanMetric, MinMetric

from src.data.face_age_datamodule import FaceAgeDataModule
from src.models import models


class FaceAgeModule(pl.LightningModule):
    def __init__(self, rescale_labels_by: int = 1.0):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters()

        # self.net = models.SimpleConvNet_100x100()
        # self.net = models.SimpleConvNet_224x224()
        # self.net = models.PretrainedResnetVGGFace2()
        self.net = models.PretrainedEfficientNet()

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
        if self.hparams.rescale_labels_by:
            preds = preds * self.hparams.rescale_labels_by
            y = y * self.hparams.rescale_labels_by
            preds = preds.clip(1, self.hparams.rescale_labels_by)

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

    age_norm_value = 80

    datamodule = FaceAgeDataModule(
        data_dir=data_dir,
        normalize_age_by=age_norm_value,
        num_workers=12,
        batch_size=32,
    )

    for i in range(0, 5):
        pl.seed_everything(i)

        model = FaceAgeModule(rescale_labels_by=age_norm_value)

        callbacks = []
        loggers = []

        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                monitor="val/loss",
                dirpath=logs_dir / "checkpoints" / str(i),
                save_top_k=1,
                save_last=True,
                mode="min",
                save_weights_only=True,
                filename="best-checkpoint",
            )
        )

        loggers.append(
            pl.loggers.WandbLogger(
                project="face-age",
                save_dir=logs_dir,
                name="224x224+EffNet-b0-+balanced-val+cut500+clip80+label-norm+mirror-augment+MSELoss",
                group="224x224+EffNet-b0+balanced-val+cut500+clip80+label-norm+mirror-augment+MSELoss",
            )
        )

        trainer = pl.Trainer(
            callbacks=callbacks,
            logger=loggers,
            default_root_dir=logs_dir,
            accelerator="gpu",
            max_epochs=10,
            val_check_interval=0.1,
        )

        trainer.validate(model=model, datamodule=datamodule)

        trainer.fit(model=model, datamodule=datamodule)

        trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

        wandb.finish()


if __name__ == "__main__":
    main()
