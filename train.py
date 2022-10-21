import pytorch_lightning as pl
from face_age_datamodule import FaceAgeDataModule

class LitModel(pl.LightningModule):
    
    def __self__(self):
        super().__init__()
        
    # TODO


def main():
    model = LitModel()
    datamodule = FaceAgeDataModule()
    
    trainer = pl.Trainer()
    
    # trainer.fit(model=model, datamodule=datamodule)
    

if __name__ == "__main__":
    main()
