from thesis_rrg.metrics import text_metrics


class _Metric:
    def __init__(self, payload):
        self.payload = payload

    def compute(self, **kwargs):
        return self.payload



def test_text_metrics(monkeypatch):
    monkeypatch.setattr(text_metrics, "_rouge", lambda: _Metric({"rougeL": 0.5}))
    monkeypatch.setattr(text_metrics, "_bleu", lambda: _Metric({"bleu": 0.2}))

    out = text_metrics.compute_text_metrics(["a"], ["b"])

    assert out["n"] == 1
    assert out["rouge"]["rougeL"] == 0.5
    assert out["sacrebleu"]["bleu"] == 0.2
