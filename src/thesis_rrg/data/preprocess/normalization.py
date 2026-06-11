from __future__ import annotations

import re

_WS = re.compile(r"\s+")



def minimal_normalize(text: str | None) -> str:
    if not isinstance(text, str):
        return ""
    return _WS.sub(" ", text.strip())
