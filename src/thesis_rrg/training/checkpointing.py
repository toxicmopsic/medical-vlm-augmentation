from __future__ import annotations

from pathlib import Path

from thesis_rrg.utils.io import ensure_dir, write_text



def prepare_checkpoint_dirs(output_dir: str | Path) -> tuple[Path, Path]:
    root = Path(output_dir)
    last_dir = ensure_dir(root / "checkpoints" / "last")
    best_dir = ensure_dir(root / "checkpoints" / "best")
    return last_dir, best_dir



def write_best_checkpoint_pointer(output_dir: str | Path, best_checkpoint: str | None) -> None:
    if best_checkpoint:
        write_text(Path(output_dir) / "checkpoints" / "best" / "source_checkpoint.txt", best_checkpoint)
