from pathlib import Path

from ._classes import CompressorMap, TurbineMap
from . import _data as from_ref
from ._data import load_map

__all__ = [
    "CompressorMap",
    "TurbineMap",
    "from_ref",
    "load_map"
]
