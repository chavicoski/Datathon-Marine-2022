"""Script to transform the raw dataset into a format ready for training
the machine learning models"""

import hydra
from hydra.utils import instantiate
from lib.data.preprocessing import PreprocPipeline
from lib.utils import set_all_seeds
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="config", config_name="preprocess_data")
def main(cfg: DictConfig):
    # Set random seeds for reproducibility
    set_all_seeds(cfg.seed)

    pipeline: PreprocPipeline = instantiate(cfg.preprocessing_pipeline)
    pipeline.preprocess_data()


if __name__ == "__main__":
    main()
