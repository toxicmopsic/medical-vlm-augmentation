from __future__ import annotations

import re



def parse_pred_findings_impression(gen_text: str) -> tuple[str, str]:
    t = "" if gen_text is None else str(gen_text).strip()
    t = re.sub(r"^[\s/\\|]+", "", t)
    if not t:
        return "", ""

    m = re.search(r"(?i)\bimpression\s*:\s*", t)
    if not m:
        return t, ""

    findings = t[: m.start()].strip()
    impression = t[m.end() :].strip()
    return findings, impression
