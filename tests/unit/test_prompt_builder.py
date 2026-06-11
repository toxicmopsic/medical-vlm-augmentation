from thesis_rrg.data.preprocess.prompt_targets import build_mimic_prompt, build_target_text


def test_prompt_builder_uses_indication():
    p = build_mimic_prompt("shortness of breath", template="<start_of_image> {indication} findings:")
    assert "shortness of breath" in p


def test_target_builder_concatenates_findings_and_impression():
    class _Tok:
        eos_token = "<eos>"

    t = build_target_text("No edema", "Mild bibasilar atelectasis", _Tok())
    assert t == "No edema impression: Mild bibasilar atelectasis<eos>"
