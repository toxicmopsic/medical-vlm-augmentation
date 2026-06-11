import types
import sys

from thesis_rrg.metrics.radeval_metrics import compute_radeval_metrics


def test_radeval_disabled_returns_stub():
    out = compute_radeval_metrics(
        preds=["a"],
        refs=["b"],
        cfg={"enabled": False},
    )
    assert out["enabled"] is False
    assert out["available"] is False
    assert out["reason"] == "disabled_by_config"


def test_radeval_happy_path_with_fake_module(monkeypatch):
    class FakeEvaluator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self, refs, hyps):
            assert refs == ["ref"]
            assert hyps == ["pred"]
            return {
                "bertscore": 0.1,
                "ratescore": 0.2,
                "radcliq_v1": 0.3,
                "green": 0.4,
                "radgraph_simple": 0.5,
                "radgraph_partial": 0.6,
                "radgraph_complete": 0.7,
                "f1chexbert_5_micro_f1": 0.8,
            }

    fake_mod = types.SimpleNamespace(RadEval=FakeEvaluator)
    monkeypatch.setitem(sys.modules, "RadEval", fake_mod)

    out = compute_radeval_metrics(
        preds=["pred"],
        refs=["ref"],
        cfg={"enabled": True, "do_green": True},
    )
    assert out["available"] is True
    assert out["selected_metrics"]["radgraph_partial"] == 0.6
    assert out["selected_metrics"]["f1chexbert"]["f1chexbert_5_micro_f1"] == 0.8
