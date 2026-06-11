from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from thesis_rrg.data.readers.dicom_reader import dicom_to_uint8_grayscale



def pad_resize_to_square(img: Image.Image, out_size: int = 896) -> Image.Image:
    w, h = img.size
    scale = out_size / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    img = img.resize((new_w, new_h), resample=Image.BICUBIC)
    canvas = Image.new("RGB", (out_size, out_size), (0, 0, 0))

    left = (out_size - new_w) // 2
    top = (out_size - new_h) // 2
    canvas.paste(img, (left, top))
    return canvas



def safe_open_mimic_dicom(dcm_path: str | Path, out_size: int = 896, augment: Optional[Callable] = None) -> Image.Image:
    arr = dicom_to_uint8_grayscale(Path(dcm_path))
    img = Image.fromarray(arr, mode="L").convert("RGB")
    img = pad_resize_to_square(img, out_size=out_size)

    if augment is not None:
        img = augment(img)
    return img

def safe_open_canonical_png(image_path: str | Path, out_size: int = 896, augment: Optional[Callable] = None) -> Image.Image:
    img = Image.open(Path(image_path)).convert("RGB")
    img = pad_resize_to_square(img, out_size=out_size)

    if augment is not None:
        img = augment(img)
    return img

def safe_open_report_image(row: dict, out_size: int = 896, augment: Optional[Callable] = None) -> Image.Image:
    image_source = str(row.get("image_source", "")).strip().lower()
    image_path = str(row.get("image_path", "")).strip()
    if image_source in {"synthetic", "synthetic_png", "png"} or image_path:
        return safe_open_canonical_png(image_path, out_size=out_size, augment=augment)

    return safe_open_mimic_dicom(Path(row["dcm_path"]), out_size=out_size, augment=augment)
