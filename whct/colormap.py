"""Turning the packed color-dimension bits into screen colors.

Each pixel's C column (up to 24 slices) is packed into one integer, one bit per
slice. A mapping file from generateMapping/ turns every 24-bit pattern into an
RGB color. With fewer than 24 slices, the slices are spread across the 24 bits so
the colors still use the whole mapping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .state import MAX_C_RES

REPO_DIR = Path(__file__).resolve().parent.parent
MAPPING_DIR = REPO_DIR / "generateMapping"
RAW_BITS = "raw bits"
DEFAULT_MAPPING = "centered_bgr"


def bit_positions(c_res: int) -> np.ndarray:
    """Which of the 24 color bits each C slice lands on."""
    c_res = int(np.clip(c_res, 1, MAX_C_RES))
    if c_res == MAX_C_RES:
        return np.arange(MAX_C_RES)
    if c_res == 1:
        return np.array([MAX_C_RES // 2])
    return np.round(np.arange(c_res) * (MAX_C_RES - 1) / (c_res - 1)).astype(int)


def _display_name(stem: str) -> str:
    name = stem
    for suffix in ("_mapping", "mapping"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name or stem


def list_mappings() -> Dict[str, Path]:
    """Display name -> mapping file (.pt preferred over .npz). Raw bits is always available."""
    found: Dict[str, Path] = {}
    if MAPPING_DIR.is_dir():
        for path in sorted(MAPPING_DIR.iterdir()):
            if path.suffix in (".pt", ".npz"):
                name = _display_name(path.stem)
                if name not in found or path.suffix == ".pt":
                    found[name] = path
    ordered: Dict[str, Path] = {}
    if DEFAULT_MAPPING in found:
        ordered[DEFAULT_MAPPING] = found.pop(DEFAULT_MAPPING)
    ordered.update(found)
    return ordered


def _load_lut(path: Path) -> np.ndarray:
    pt = path.with_suffix(".pt")
    if pt.exists():
        return torch.load(pt, map_location="cpu").numpy().astype(np.uint32)
    data = np.load(path, allow_pickle=True)
    boid_to_rgb = data["boid_to_rgb"].astype(np.int64)
    side = int(data["side"])
    if boid_to_rgb.shape[0] != 1 << 24:
        raise ValueError(f"{path.name} has {boid_to_rgb.shape[0]} entries; expected 2**24.")
    if side == 256:
        r, g, b = boid_to_rgb[:, 0], boid_to_rgb[:, 1], boid_to_rgb[:, 2]
    else:
        d = max(1, side - 1)
        r, g, b = (boid_to_rgb[:, i] * 255 // d for i in range(3))
    lut = ((r << 16) | (g << 8) | b).astype(np.int32)
    torch.save(torch.from_numpy(lut), pt)
    return lut.astype(np.uint32)


class ColorMapper:
    def __init__(self):
        self.available = list_mappings()
        self.names: List[str] = list(self.available) + [RAW_BITS]
        self._luts: Dict[str, np.ndarray] = {}
        self.name = self.names[0]

    def set_mapping(self, name: str) -> None:
        if name not in self.names:
            raise ValueError(f"Unknown color mapping: {name}")
        self.name = name

    def _lut(self) -> Optional[np.ndarray]:
        if self.name == RAW_BITS:
            return None
        if self.name not in self._luts:
            self._luts[self.name] = _load_lut(self.available[self.name])
        return self._luts[self.name]

    def to_rgb(self, index: np.ndarray) -> np.ndarray:
        """(H, W) packed bits -> (H, W, 3) uint8 RGB."""
        lut = self._lut()
        packed = index if lut is None else lut[index]
        rgb = np.empty(index.shape + (3,), dtype=np.uint8)
        rgb[..., 0] = (packed >> 16) & 0xFF
        rgb[..., 1] = (packed >> 8) & 0xFF
        rgb[..., 2] = packed & 0xFF
        return rgb
