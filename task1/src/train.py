import hydra
from hydra.utils import instantiate
from lib.lightning_modules import ClassifierModule
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.loggers import Logger
from torch.nn import Module


@hydra.main(config_path="config", config_name="train")
def main(cfg: DictConfig):
    # Prepare the data
    data_module: LightningDataModule = instantiate(cfg.data_module)

    # Prepare the model
    model: Module = instantiate(cfg.model)
    lightning_module = ClassifierModule(model=model, labels=cfg.labels)

    # Prepare the trainer
    logger: Logger = instantiate(cfg.logger)
    ckpt_callback = instantiate(cfg.callbacks.checkpoint)
    callbacks = [ckpt_callback]
    trainer: Trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    # Train phase
    trainer.fit(lightning_module, data_module)

    # Test phase
    best_model = ckpt_callback.best_model_path
    print(f"Going to load the model for testing from '{best_model}'")
    lightning_module = ClassifierModule.load_from_checkpoint(best_model, model=model)
    trainer.test(lightning_module, data_module)


if __name__ == "__main__":
    main()
