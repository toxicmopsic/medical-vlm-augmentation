from thesis_rrg.data.preprocess.report_cleaning import parse_pred_findings_impression


def test_parse_with_impression_marker():
    findings, impression = parse_pred_findings_impression("No edema. Impression: Mild cardiomegaly.")
    assert findings == "No edema."
    assert impression == "Mild cardiomegaly."


def test_parse_without_impression_marker():
    findings, impression = parse_pred_findings_impression("No focal consolidation")
    assert findings == "No focal consolidation"
    assert impression == ""


def test_parse_strips_leading_slash_before_impression():
    findings, impression = parse_pred_findings_impression("/ impression: No pneumothorax.")
    assert findings == ""
    assert impression == "No pneumothorax."
