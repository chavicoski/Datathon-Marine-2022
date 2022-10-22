import hydra
from hydra.utils import instantiate
from lib.lightning_modules import ClassifierModule
from lib.utils import set_all_seeds
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.loggers import Logger
from torch.nn import Module


@hydra.main(config_path="config", config_name="train")
def main(cfg: DictConfig):
    # Set random seeds for reproducibility
    set_all_seeds(cfg.seed)

    # Prepare the data
    data_module: LightningDataModule = instantiate(cfg.data_module)

    # Prepare the model
    model: Module = instantiate(cfg.model, n_classes=len(cfg.labels))
    lightning_module = ClassifierModule(model=model, labels=cfg.labels)

    # Prepare the trainer
    logger: Logger = instantiate(cfg.logger)
    logger.log_hyperparams(cfg)
    ckpt_callback = instantiate(cfg.callbacks.checkpoint)
    callbacks = [ckpt_callback]
    trainer: Trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    # Train phase
    trainer.fit(lightning_module, data_module)

    # Test phase
    best_model = ckpt_callback.best_model_path
    print(f"Going to load the model for testing from '{best_model}'")
    lightning_module = ClassifierModule.load_from_checkpoint(
        best_model, model=model, labels=cfg.labels
    )
    trainer.test(lightning_module, data_module)


if __name__ == "__main__":
    main()
