import pytest

pydicom = pytest.importorskip("pydicom")

from thesis_rrg.data.readers.dicom_reader import _apply_window


def test_apply_window_bounds():
    import numpy as np

    arr = np.array([0.0, 10.0, 20.0], dtype=np.float32)
    out = _apply_window(arr, center=10.0, width=20.0)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
