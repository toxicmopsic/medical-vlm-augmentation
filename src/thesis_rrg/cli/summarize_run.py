from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf



def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    output_dir = Path(str(cfg.paths.output_dir))

    train_metrics = _read_json(output_dir / "metrics" / "train_metrics.json")
    valid_metrics = _read_json(output_dir / "metrics" / "valid_metrics.json")
    test_metrics = _read_json(output_dir / "metrics" / "test_metrics.json")

    summary = {
        "output_dir": str(output_dir),
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
