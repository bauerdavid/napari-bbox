"""
This module is an example of a barebones writer plugin for napari.

It implements the Writer specification.
see: https://napari.org/stable/plugins/guides.html?#writers

Replace code below according to your needs.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Sequence, Tuple, Union
import numpy as np

if TYPE_CHECKING:
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = Tuple[DataType, dict, str]


def write_single_bbox(path: str, data: Any, meta: dict) -> List[str]:
    """Writes a single image layer"""
    corners = np.asarray(
        list(map(lambda bb: np.concatenate([bb.min(axis=0), bb.max(axis=0)]), data)))

    np.savetxt(path, corners, "%.4f", ",")
    return [path]

