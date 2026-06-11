from __future__ import annotations

from typing import Any

try:
    from rich import print as rich_print
except Exception:  # pragma: no cover
    rich_print = print



def cprint(*args: Any, **kwargs: Any) -> None:
    rich_print(*args, **kwargs)
