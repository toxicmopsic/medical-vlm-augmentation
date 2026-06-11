from __future__ import annotations

import re
from typing import Iterable

_ws = re.compile(r"\s+")



def minimal_normalize(text: str | None) -> str:
    if not isinstance(text, str):
        return ""
    return _ws.sub(" ", text.strip())



def extract_section(report_text: str, section_names: Iterable[str]) -> str:
    if not report_text:
        return ""

    txt = report_text.replace("\r\n", "\n").replace("\r", "\n")
    for name in section_names:
        pattern = rf"(?ims)^\s*{re.escape(name)}\s*:\s*(.*?)(?=^\s*[A-Z][A-Z0-9 /\-()]+?\s*:|\Z)"
        m = re.search(pattern, txt)
        if m:
            return minimal_normalize(m.group(1))
    return ""
