from thesis_rrg.metrics.radgraph_metrics import sanitize_for_radgraph


def test_sanitize_empty_text():
    out = sanitize_for_radgraph("")
    assert isinstance(out, str)
    assert len(out) > 0
    assert out.endswith(".")


def test_sanitize_one_token_text():
    out = sanitize_for_radgraph("normal")
    assert out.endswith(".")
    assert len(out.split()) >= 2


def test_sanitize_keeps_meaningful_content():
    src = "No pleural effusion. . !!! Lungs are clear."
    out = sanitize_for_radgraph(src)
    assert "pleural" in out.lower() or "lungs" in out.lower()
    assert out.endswith(".")
