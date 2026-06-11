from thesis_rrg.metrics.radeval_metrics import _sanitize_ratescore_text


def test_sanitize_ratescore_unescapes_punctuation():
    src = r"No edema \. No pleural effusion \, stable \(mild\)."
    out = _sanitize_ratescore_text(src)
    assert r"\." not in out
    assert r"\," not in out
    assert r"\(" not in out
    assert r"\)" not in out


def test_sanitize_ratescore_collapses_whitespace():
    src = "A  \n  B\t\tC"
    out = _sanitize_ratescore_text(src)
    assert out == "A B C"
