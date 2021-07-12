from argparse import Namespace
from typing import List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from handwritten_digit_string_recognition.data import MultiDigitMNIST
from handwritten_digit_string_recognition.lit_models import CTCLitModel


@hydra.main(config_path="../", config_name="config")
def main(cfg: DictConfig):
    datamodule = MultiDigitMNIST(**cfg.data)
    datamodule.prepare_data()
    datamodule.setup()

    cfg.lit_model.padding_index = datamodule.padding_index
    cfg.lit_model.blank_index = datamodule.blank_index
    lit_model = CTCLitModel(**cfg.lit_model)

    callbacks: List[Callback] = []
    if cfg.callbacks.model_checkpoint:
        callbacks.append(ModelCheckpoint(**cfg.callbacks.model_checkpoint))
    if cfg.callbacks.early_stopping:
        callbacks.append(EarlyStopping(**cfg.callbacks.early_stopping))

    logger: Optional[WandbLogger] = None
    if cfg.logger:
        logger = WandbLogger(**cfg.logger)

    trainer = Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)

    if trainer.logger:
        trainer.logger.log_hyperparams(Namespace(**cfg))

    trainer.tune(lit_model, datamodule=datamodule)
    trainer.fit(lit_model, datamodule=datamodule)
    trainer.test(lit_model, datamodule=datamodule)


if __name__ == "__main__":
    main()
