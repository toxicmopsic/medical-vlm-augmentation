from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional



def try_get_git_commit(cwd: str | Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None
