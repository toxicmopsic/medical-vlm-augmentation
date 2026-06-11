from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from thesis_rrg.training.pipeline import run_training


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    run_training(cfg)


if __name__ == "__main__":
    main()
