import pyrootutils
import pytorch_lightning as pl
import wandb

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

from src.data.face_age_datamodule import FaceAgeDataModule
from src.models.face_age_module import FaceAgeModule
from src.models import models


def main():
    pl.seed_everything(42)

    data_dir = root / "data"
    log_dir = root / "logs"

    use_wandb = False
    age_norm_value = 80

    net = models.SimpleConvNet_100x100()
    img_size = (100, 100)

    # net = models.SimpleConvNet_224x224()
    # img_size = (224, 224)

    # net = models.PretrainedEfficientNet()
    # img_size = (224, 224)

    datamodule = FaceAgeDataModule(
        data_dir=data_dir,
        normalize_age_by=age_norm_value,
        img_size=img_size,
        num_workers=6,
        batch_size=32,
        pin_memory=False,
    )

    model = FaceAgeModule(net=net, rescale_age_by=age_norm_value)

    callbacks = []
    loggers = []

    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            monitor="val/loss",
            dirpath=log_dir / "checkpoints",
            save_top_k=1,
            save_last=True,
            mode="min",
            save_weights_only=True,
            filename="best-checkpoint",
        )
    )

    if use_wandb:
        loggers.append(
            pl.loggers.WandbLogger(
                project="face-age",
                save_dir=log_dir,
                # name=f"EffNet+{img_size}+age_norm_{age_norm_value}",
                # group=f"EffNet+{img_size}+age_norm_{age_norm_value}",
            )
        )

    trainer = pl.Trainer(
        # accelerator="gpu",
        default_root_dir=log_dir,
        callbacks=callbacks,
        logger=loggers,
        max_epochs=10,
        # val_check_interval=0.1,  # frequency of validation epoch
    )

    # validate before training
    trainer.validate(model=model, datamodule=datamodule)

    # train
    trainer.fit(model=model, datamodule=datamodule)

    # test
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
