from __future__ import annotations

import hashlib
from pathlib import Path



def file_sha1(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha1()
    with p.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
