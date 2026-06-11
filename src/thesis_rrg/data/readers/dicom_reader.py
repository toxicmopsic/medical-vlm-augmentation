from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut

def dicom_to_uint8_grayscale(dcm_path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(dcm_path))
    arr = ds.pixel_array  # usually int16/uint16 from DICOM decoder

    try:
        # dtype after modality LUT may become float64 depending on tags
        arr = apply_modality_lut(arr, ds)
    except Exception:
        # keep raw decoded array if LUT is unavailable/broken
        arr = np.asarray(arr)

    # unify numeric type for stable normalization math: int16/uint16/float64 -> float32
    arr_f = np.asarray(arr, dtype=np.float32)
    mn, mx = float(arr_f.min()), float(arr_f.max())

    # float32 in [0, 1]
    if mx > mn:
        arr01 = (arr_f - mn) / (mx - mn)
    else:
        arr01 = np.zeros_like(arr_f, dtype=np.float32)

    arr01 = np.clip(arr01, 0.0, 1.0)  # float32 in [0, 1]

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr01 = 1.0 - arr01  # still float32 in [0, 1]

    return (arr01 * 255.0).clip(0, 255).astype(np.uint8)  # final uint8
