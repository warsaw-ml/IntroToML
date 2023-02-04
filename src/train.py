import pyrootutils
import pytorch_lightning as pl
import wandb

# set pythonpath and working directory to folder containing .project-root file
root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

from src.data.face_age_datamodule import FaceAgeDataModule
from src.models.face_age_module import FaceAgeModule


def main():
    """
    The main function is the entry point of the program. It sets up the data and model,
    creates callbacks and loggers, and then runs the training, validation, and testing process using the
    Pytorch Lightning Trainer.
    """

    # set seed for reproducibility
    pl.seed_everything(2)

    data_dir = root / "data"
    log_dir = root / "logs"

    use_wandb = False
    age_norm_value = 80

    # choose one of the architectures by uncommenting the set of corresponding hyperparameters below

    # 1
    net = "SimpleConvNet_100x100"
    img_size = (100, 100)
    imagenet_normalization = False
    exp_name = f"SimpleConvNet+{img_size}"

    # 2
    # net = "SimpleConvNet_224x224"
    # img_size = (224, 224)
    # imagenet_normalization = False
    # exp_name = f"SimpleConvNet+{img_size}"

    # 3
    # net = "EffNet_224x224"
    # img_size = (224, 224)
    # imagenet_normalization = True
    # exp_name = f"EffNet+{img_size}"

    datamodule = FaceAgeDataModule(
        data_dir=data_dir,
        img_size=img_size,
        imagenet_normalization=imagenet_normalization,
        normalize_age_by=age_norm_value,
        num_workers=0,
        batch_size=32,
        pin_memory=False,
    )

    model = FaceAgeModule(net=net, rescale_age_by=age_norm_value)

    callbacks = []
    loggers = []

    # this controls how checkpoints are saved
    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            monitor="val/loss",
            dirpath=log_dir / "checkpoints",
            save_top_k=1,  # save the best checkpoint
            save_last=True,  # additionally the save the last checkpoint
            mode="min",
            save_weights_only=True,
            filename="best-checkpoint",
        )
    )

    # this configurates optional weights&biases logger
    if use_wandb:
        loggers.append(
            pl.loggers.WandbLogger(
                project="face-age",
                save_dir=log_dir,
                name=exp_name,
                group=exp_name,
            )
        )

    # trainer setup
    trainer = pl.Trainer(
        accelerator="cpu",
        # accelerator="gpu",
        default_root_dir=log_dir,
        callbacks=callbacks,
        logger=loggers,
        max_epochs=10,
        # val_check_interval=0.1,  # frequency of validation epoch
    )

    # optiomally validate before training
    # trainer.validate(model=model, datamodule=datamodule)

    # train
    trainer.fit(model=model, datamodule=datamodule)

    # test
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
