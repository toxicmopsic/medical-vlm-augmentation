from pathlib import Path


def test_prepare_config_exists():
    cfg = Path("configs/prepare.yaml")
    assert cfg.exists()
