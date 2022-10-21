"""Script to transform the raw dataset into a format ready for training
the machine learning models"""

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="preprocess_data")
def main(cfg: DictConfig):
    pass


if __name__ == "__main__":
    main()
