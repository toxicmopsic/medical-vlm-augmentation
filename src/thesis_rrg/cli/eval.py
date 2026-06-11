from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from thesis_rrg.evaluation.pipeline import run_evaluation


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
