import pandas as pd

from thesis_rrg.evaluation.error_analysis import generation_pathology_report, generation_pathology_summary


def test_generation_pathology_flags_repeated_time_and_punctuation():
    df = pd.DataFrame(
        [
            {
                "study_id": "1",
                "raw_pred": "impression: No acute process.___ at 10:00 a.m.___ at 10:00 a.m.___ at 10:00 a.m.___ at 10:00 a.m.___ at 10:00 a.m.",
                "pred_findings": "",
                "pred_impression": "No acute process.___ at 10:00 a.m.",
                "ref_impression": "No acute process.",
            },
            {
                "study_id": "2",
                "raw_pred": "impression: Mild edema." + "-" * 40,
                "pred_findings": "",
                "pred_impression": "Mild edema.",
                "ref_impression": "Mild edema.",
            },
        ]
    )

    flags = generation_pathology_report(df)
    summary = generation_pathology_summary(flags)

    assert flags.loc[0, "has_time_like"]
    assert flags.loc[0, "has_ngram_repeat_ge_5"]
    assert flags.loc[1, "long_punct_run"]
    assert summary["has_time_like"] == 1
    assert summary["long_punct_run"] == 1
