from pathlib import Path


def test_train_config_exists():
    cfg = Path("configs/train.yaml")
    assert cfg.exists()
