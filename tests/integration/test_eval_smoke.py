from pathlib import Path


def test_eval_config_exists():
    cfg = Path("configs/eval.yaml")
    assert cfg.exists()
