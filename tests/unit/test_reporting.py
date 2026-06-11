from omegaconf import OmegaConf

from thesis_rrg.training.reporting import resolve_report_to


def test_resolve_report_to_prefers_training_config():
    cfg = OmegaConf.create(
        {
            "training": {"report_to": "tensorboard"},
            "logging": {"report_to": "none"},
        }
    )
    assert resolve_report_to(cfg) == "tensorboard"


def test_resolve_report_to_falls_back_to_logging_config():
    cfg = OmegaConf.create(
        {
            "training": {},
            "logging": {"report_to": "tensorboard"},
        }
    )
    assert resolve_report_to(cfg) == "tensorboard"
