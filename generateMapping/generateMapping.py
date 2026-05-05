#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generateMapping.py

This file creates the color mapping files used by the WHCT / XYTC viewer.

The viewer can pack the color dimension into a 24-bit integer. This script decides
how each one of those 24-bit values should become an actual RGB color. In other
words, it builds a bijective mapping between:

    boid_id / 24-bit color-dimension pattern  <->  (R, G, B) coordinate

The saved .npz file contains:

    boid_to_rgb:  shape (N, 3)
                  boid_to_rgb[i] = the RGB coordinate assigned to boid/value i

    rgb_to_boid:  shape (N,)
                  rgb_to_boid[linear(R, G, B)] = the boid/value at that RGB voxel

    side:         the side length of the RGB cube
                  For the normal 24-bit case, side=256 and N=256^3=2^24.

    meta:         a small JSON string describing how the mapping was generated

Mapping modes
-------------
There are two types of mapping modes in this file.

1. Feature / transport modes

   These first describe each 24-bit value using three features, then arrange the
   values into the RGB cube using monotone transport and optional CUDA neighbor
   swaps.

   - rgb_bytes_generalized
       Splits the bitstring into 3 chunks and uses those chunks as the starting
       feature values.

   - SPC
       Uses popcount, average position of the 1s, and transition count.

2. Direct mapping modes

   These directly construct boid_to_rgb and rgb_to_boid. They do not go through
   the feature / transport pipeline, because that would change the exact mapping
   they are trying to preserve.

   - centered_bgr
       A direct bit permutation. The left side of the color-dimension bits maps
       toward blue, the middle maps toward green, and the right side maps toward
       red.

   - layered_popcount
       The old generateLayeredMapping.py behavior. Values are grouped by how many
       1s they have in binary, then assigned through diagonal RGB layers where
       R + G + B increases.

Usage examples
--------------
    python generateMapping.py --mapping-mode rgb_bytes_generalized --output mapping.npz
    python generateMapping.py --mapping-mode SPC --max-iters 6 --output spc_mapping.npz
    python generateMapping.py --mapping-mode centered_bgr --output centered_bgr_mapping.npz
    python generateMapping.py --mapping-mode layered_popcount --output layered_popcount_mapping.npz

The old --feature-mode flag is still accepted as an alias for --mapping-mode, but
--mapping-mode is the clearer name now because not every mode is feature-based.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Iterator, Tuple

import numpy as np
import torch
from numba import njit  # Used for the CPU counting sort in the transport modes.


TRANSPORT_MAPPING_MODES = ("rgb_bytes_generalized", "SPC")
DIRECT_MAPPING_MODES = ("centered_bgr", "layered_popcount")
MAPPING_MODES = TRANSPORT_MAPPING_MODES + DIRECT_MAPPING_MODES


# --------------------------------------------------------------------------------------
# -------------------------------- Shared save helpers ---------------------------------
# --------------------------------------------------------------------------------------

def save_mapping_npz(
    output: str,
    boid_to_rgb: np.ndarray,
    rgb_to_boid: np.ndarray,
    side: int,
    mapping_mode: str,
    description: str,
) -> None:
    """
    Save a mapping in the format expected by the WHCT / XYTC viewer.

    boid_to_rgb and rgb_to_boid should already be built and validated by the
    mapping function. This function just does one final shape check and writes
    the compressed .npz file.
    """
    n_expected = int(side) ** 3

    if boid_to_rgb.shape != (n_expected, 3):
        raise ValueError(
            f"boid_to_rgb should have shape {(n_expected, 3)}, got {boid_to_rgb.shape}"
        )
    if rgb_to_boid.shape != (n_expected,):
        raise ValueError(
            f"rgb_to_boid should have shape {(n_expected,)}, got {rgb_to_boid.shape}"
        )

    meta = {
        "side": int(side),
        "N": int(n_expected),
        "mapping_mode": mapping_mode,
        # Kept for compatibility with older mapping files / notes that used this name.
        "feature_mode": mapping_mode,
        "description": description,
    }

    np.savez_compressed(
        output,
        boid_to_rgb=boid_to_rgb.astype(np.uint16, copy=False),
        rgb_to_boid=rgb_to_boid.astype(np.uint32, copy=False),
        side=np.array(side, dtype=np.int32),
        meta=np.array(json.dumps(meta), dtype=object),
    )
    print("Saved", output)


def write_preview_csv(output: str, boid_to_rgb: np.ndarray) -> None:
    """
    Write a small CSV preview of the mapping.

    This is intentionally optional because a full side=256 mapping has 16,777,216
    rows, which is not a fun CSV to open or move around.
    """
    csv_path = os.path.splitext(output)[0] + "_preview.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("boid_id,R,G,B\n")
        for boid_id, (r, g, b) in enumerate(boid_to_rgb):
            f.write(f"{boid_id},{int(r)},{int(g)},{int(b)}\n")
    print("Saved", csv_path)


# --------------------------------------------------------------------------------------
# --------------------------- Feature computation modes --------------------------------
# --------------------------------------------------------------------------------------

def compute_features(
    boid_ids: np.ndarray,
    side: int,
    mode: str = "SPC",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (fx, fy, fz) in [0, side-1] for the feature / transport modes.

    These features are not final colors. They are only the starting description
    that the transport pipeline uses to arrange values into the RGB cube.

    Modes:
      - "SPC":  fx = popcount
                fy = average position of 1s   (0..bitlen-1; LSB=0)
                fz = transitions count        (0..bitlen-1)

      - "rgb_bytes_generalized":
                Split the bitstring into 3 contiguous chunks (LSB -> MSB)
                as evenly as possible, then use each chunk's integer value.
    """
    if mode not in TRANSPORT_MAPPING_MODES:
        raise ValueError(
            f"{mode!r} is not a feature / transport mode. "
            f"Choose one of {TRANSPORT_MAPPING_MODES}."
        )

    assert side >= 2
    n = int(side) ** 3
    bitlen = math.ceil(math.log2(n))
    assert bitlen <= 64, "bitlen too large for uint64 path"

    x = boid_ids.astype(np.uint64) & ((1 << bitlen) - 1)

    def to_side_scale(values_0_255: np.ndarray) -> np.ndarray:
        # Map [0, 255] -> [0, side - 1] by floor(value * side / 256).
        return (values_0_255.astype(np.uint32) * side // 256).astype(np.uint16)

    if mode == "rgb_bytes_generalized":
        # Split bitlen into three chunks as evenly as possible.
        # The first chunk gets the least significant bits.
        base = bitlen // 3
        remainder = bitlen % 3
        s1 = base + (1 if remainder > 0 else 0)
        s2 = base + (1 if remainder > 1 else 0)
        s3 = base

        if s1 > 0:
            m1 = (1 << s1) - 1
            c1 = (x >> 0) & m1
            c1max = m1
        else:
            c1 = np.zeros_like(x)
            c1max = 1

        if s2 > 0:
            m2 = (1 << s2) - 1
            c2 = (x >> s1) & m2
            c2max = m2
        else:
            c2 = np.zeros_like(x)
            c2max = 1

        if s3 > 0:
            m3 = (1 << s3) - 1
            c3 = (x >> (s1 + s2)) & m3
            c3max = m3
        else:
            c3 = np.zeros_like(x)
            c3max = 1

        def to_byte(values: np.ndarray, vmax: int) -> np.ndarray:
            vmax = int(vmax)
            if vmax <= 0:
                return np.zeros_like(values, dtype=np.uint16)
            return np.floor(values.astype(np.float64) * (255.0 / vmax) + 0.5).astype(
                np.uint16
            )

        fx_raw = to_byte(c1, c1max)
        fy_raw = to_byte(c2, c2max)
        fz_raw = to_byte(c3, c3max)
        return to_side_scale(fx_raw), to_side_scale(fy_raw), to_side_scale(fz_raw)

    if mode == "SPC":
        # Build a bit matrix, LSB first: bit position 0..bitlen-1.
        positions = np.arange(bitlen, dtype=np.uint64)
        bits = ((x[:, None] >> positions[None, :]) & 1).astype(np.uint8)

        # S: popcount.
        pop = bits.sum(axis=1).astype(np.float64)

        # P: average position of 1s. If there are no 1s, use 0.
        weighted_sum = (bits * positions[None, :]).sum(axis=1).astype(np.float64)
        avgpos = np.where(pop > 0.0, weighted_sum / pop, 0.0)

        # C: count transitions between neighboring bits.
        if bitlen > 1:
            trans = (bits[:, 1:] != bits[:, :-1]).sum(axis=1).astype(np.float64)
        else:
            trans = np.zeros_like(pop, dtype=np.float64)

        fx_raw = np.floor(pop * (255.0 / max(1, bitlen)) + 0.5).astype(np.uint16)
        fy_raw = np.floor(avgpos * (255.0 / max(1, bitlen - 1)) + 0.5).astype(
            np.uint16
        )
        fz_raw = np.floor(trans * (255.0 / max(1, bitlen - 1)) + 0.5).astype(
            np.uint16
        )
        return to_side_scale(fx_raw), to_side_scale(fy_raw), to_side_scale(fz_raw)

    # This should be unreachable because of the earlier mode check.
    raise ValueError(f"Unknown feature mode: {mode}")


# --------------------------------------------------------------------------------------
# ------------------------------- Direct mapping modes ---------------------------------
# --------------------------------------------------------------------------------------

def centered_bgr_direct_mapping(side: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Direct 24-bit bit-permutation mapping.

    This keeps the mapping extremely literal: each bit in the 24-bit input still
    becomes one bit in the final 24-bit color. The only thing that changes is
    which output color bit it controls.

    The 24 input bits are interpreted left-to-right across the color dimension as:

      bits 0..7    -> blue byte,  low-to-high:  1, 2, 4, 8, 16, 32, 64, 128
      bits 8..15   -> green byte, centered:     1, 4, 16, 64, 128, 32, 8, 2
      bits 16..23  -> red byte,   high-to-low: 128, 64, 32, 16, 8, 4, 2, 1

    Output is stored as boid_to_rgb[i] = (R, G, B).
    """
    if side != 256:
        raise ValueError(
            "centered_bgr is an exact 24-bit 0xRRGGBB bit permutation "
            "and currently requires --side 256."
        )

    n = side ** 3
    boid_ids = np.arange(n, dtype=np.uint32)

    r = np.zeros(n, dtype=np.uint32)
    g = np.zeros(n, dtype=np.uint32)
    b = np.zeros(n, dtype=np.uint32)

    # Each list entry says: source bit index -> output channel bit weight.
    bit_layout = [
        # Blue: low -> high
        ("b", 1),
        ("b", 2),
        ("b", 4),
        ("b", 8),
        ("b", 16),
        ("b", 32),
        ("b", 64),
        ("b", 128),
        # Green: low/outside -> high/center -> low/outside
        ("g", 1),
        ("g", 4),
        ("g", 16),
        ("g", 64),
        ("g", 128),
        ("g", 32),
        ("g", 8),
        ("g", 2),
        # Red: high -> low
        ("r", 128),
        ("r", 64),
        ("r", 32),
        ("r", 16),
        ("r", 8),
        ("r", 4),
        ("r", 2),
        ("r", 1),
    ]

    for src_bit, (channel, weight) in enumerate(bit_layout):
        bit_is_on = ((boid_ids >> src_bit) & 1).astype(np.uint32)

        if channel == "r":
            r |= bit_is_on * weight
        elif channel == "g":
            g |= bit_is_on * weight
        elif channel == "b":
            b |= bit_is_on * weight
        else:
            raise ValueError(f"Unknown channel: {channel}")

    boid_to_rgb = np.empty((n, 3), dtype=np.uint16)
    boid_to_rgb[:, 0] = r.astype(np.uint16)
    boid_to_rgb[:, 1] = g.astype(np.uint16)
    boid_to_rgb[:, 2] = b.astype(np.uint16)

    rgb_to_boid = np.empty(n, dtype=np.uint32)
    rgb_index = (r + side * (g + side * b)).astype(np.uint32)
    rgb_to_boid[rgb_index] = boid_ids

    return boid_to_rgb, rgb_to_boid


def _iter_diagonal_rgb_coords(side: int) -> Iterator[Tuple[int, int, int]]:
    """
    Yield every RGB cube coordinate in diagonal-layer order.

    This is the coordinate ordering from the old generateLayeredMapping.py file:
    first all coordinates where R+G+B is small, then the next diagonal layer, and
    so on until the far corner of the cube.
    """
    max_sum = 3 * (side - 1)
    for total in range(max_sum + 1):
        for r in range(side):
            g_min = max(0, total - r - (side - 1))
            g_max = min(side - 1, total - r)

            if g_min > g_max:
                continue

            for g in range(g_min, g_max + 1):
                b = total - r - g
                yield r, g, b


def _iter_ints_by_popcount(n: int, bitlen: int) -> Iterator[int]:
    """
    Yield integers 0..n-1 grouped by popcount.

    This matches the core idea of the old layered mapping: all values with 0 one
    bits first, then all values with 1 one bit, then 2, etc. Within each group,
    values are yielded in increasing integer order.
    """
    for popcount in range(bitlen + 1):
        if popcount == 0:
            if n > 0:
                yield 0
            continue

        # First integer with this many 1 bits.
        value = (1 << popcount) - 1
        limit = 1 << bitlen

        while value < limit:
            if value < n:
                yield value

            # Gosper's hack: next higher integer with the same number of 1 bits.
            smallest = value & -value
            ripple = value + smallest
            value = ripple | (((ripple ^ value) >> 2) // smallest)


def layered_popcount_direct_mapping(side: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    The old generateLayeredMapping.py behavior, merged into this file.

    The mapping has two steps:

      1. Group all boid IDs / 24-bit values by how many 1s they have in binary.
      2. Assign those groups into diagonal RGB layers where R + G + B increases.

    Conceptually, this makes the number of occupied color-dimension slices line up
    with broad layers of the RGB cube.
    """
    n = side ** 3
    bitlen = math.ceil(math.log2(n))

    boid_to_rgb = np.empty((n, 3), dtype=np.uint16)
    rgb_to_boid = np.empty(n, dtype=np.uint32)

    coord_iter = _iter_diagonal_rgb_coords(side)
    count = 0

    for boid_id in _iter_ints_by_popcount(n, bitlen):
        try:
            r, g, b = next(coord_iter)
        except StopIteration as exc:
            raise RuntimeError("Ran out of RGB coordinates before assigning every boid.") from exc

        boid_to_rgb[boid_id] = (r, g, b)
        rgb_to_boid[r + side * (g + side * b)] = boid_id
        count += 1

    if count != n:
        raise RuntimeError(f"Expected to assign {n} boids, but assigned {count}.")

    try:
        extra_coord = next(coord_iter)
    except StopIteration:
        extra_coord = None

    if extra_coord is not None:
        raise RuntimeError("Generated more RGB coordinates than expected.")

    return boid_to_rgb, rgb_to_boid


# --------------------------------------------------------------------------------------
# ---------------------- 3-pass monotone transport (CPU, O(N)) ------------------------
# --------------------------------------------------------------------------------------

@njit(cache=True)
def _stable_counting_sort_by_key(keys_u16, values_i32, key_count):
    n = values_i32.size
    counts = np.zeros(key_count, dtype=np.int64)

    for i in range(n):
        counts[keys_u16[i]] += 1

    offsets = np.empty(key_count, dtype=np.int64)
    running = 0
    for key in range(key_count):
        offsets[key] = running
        running += counts[key]

    out = np.empty_like(values_i32)
    for i in range(n):
        key = keys_u16[i]
        pos = offsets[key]
        out[pos] = values_i32[i]
        offsets[key] = pos + 1

    return out


def three_pass_monotone_transport(
    fx: np.ndarray,
    fy: np.ndarray,
    fz: np.ndarray,
    side: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build an initial bijective RGB assignment from feature values.

    This does quota-based stable sorting in three stages:

      1. Sort all boids by fx and assign R.
      2. Within each R slice, sort by fy and assign G.
      3. Within each R/G column, sort by fz and assign B.

    The result is already a valid bijection. The optional CUDA swaps later just
    refine it locally.
    """
    n = fx.size
    qx = side * side
    qy = side
    boids = np.arange(n, dtype=np.int32)

    perm_x = _stable_counting_sort_by_key(fx.astype(np.uint16), boids, side)
    r_assigned = np.empty(n, dtype=np.uint16)
    g_assigned = np.empty(n, dtype=np.uint16)
    b_assigned = np.empty(n, dtype=np.uint16)
    rgb_to_boid = np.empty(n, dtype=np.int32)
    pos_of_boid = np.empty(n, dtype=np.uint32)

    for r in range(side):
        ids_r = perm_x[r * qx : (r + 1) * qx]
        r_assigned[ids_r] = r
        ids_r_sorted_by_g = _stable_counting_sort_by_key(
            fy[ids_r].astype(np.uint16), ids_r, side
        )

        for g in range(side):
            ids_rg = ids_r_sorted_by_g[g * qy : (g + 1) * qy]
            g_assigned[ids_rg] = g
            ids_rg_sorted_by_b = _stable_counting_sort_by_key(
                fz[ids_rg].astype(np.uint16), ids_rg, side
            )

            for b, boid in enumerate(ids_rg_sorted_by_b):
                b_assigned[boid] = b
                idx = int(r + side * (g + side * b))
                rgb_to_boid[idx] = boid
                pos_of_boid[boid] = np.uint32(
                    (r & 0xFF) | ((g & 0xFF) << 8) | ((b & 0xFF) << 16)
                )

    return r_assigned, g_assigned, b_assigned, rgb_to_boid, pos_of_boid


# --------------------------------------------------------------------------------------
# ----------------------- PyTorch neighbor swap passes (GPU) ---------------------------
# --------------------------------------------------------------------------------------

@torch.no_grad()
def swap_pass_x(boid_of_voxel, pos_of_boid, fx, side, parity):
    s = side
    grid = torch.arange(s**3, device=boid_of_voxel.device).reshape(s, s, s)

    idx_l = grid[parity : s - 1 : 2, :, :].reshape(-1)
    idx_r = grid[parity + 1 : s : 2, :, :].reshape(-1)

    boid_l, boid_r = boid_of_voxel[idx_l], boid_of_voxel[idx_r]
    fi, fj = fx[boid_l], fx[boid_r]
    mask = (fi > fj) | ((fi == fj) & (boid_l > boid_r))

    if mask.any():
        tmp_l, tmp_r = boid_l[mask], boid_r[mask]
        boid_of_voxel[idx_l[mask]], boid_of_voxel[idx_r[mask]] = tmp_r, tmp_l

        x = idx_l[mask] % s
        y = (idx_l[mask] // s) % s
        z = idx_l[mask] // (s * s)
        pos_of_boid[tmp_l] = (x + 1) | (y << 8) | (z << 16)
        pos_of_boid[tmp_r] = x | (y << 8) | (z << 16)

    return mask.sum().item()


@torch.no_grad()
def swap_pass_y(boid_of_voxel, pos_of_boid, fy, side, parity):
    s = side
    grid = torch.arange(s**3, device=boid_of_voxel.device).reshape(s, s, s)

    idx_b = grid[:, parity : s - 1 : 2, :].reshape(-1)
    idx_t = grid[:, parity + 1 : s : 2, :].reshape(-1)

    boid_b, boid_t = boid_of_voxel[idx_b], boid_of_voxel[idx_t]
    fi, fj = fy[boid_b], fy[boid_t]
    mask = (fi > fj) | ((fi == fj) & (boid_b > boid_t))

    if mask.any():
        tmp_b, tmp_t = boid_b[mask], boid_t[mask]
        boid_of_voxel[idx_b[mask]], boid_of_voxel[idx_t[mask]] = tmp_t, tmp_b

        x = idx_b[mask] % s
        y = (idx_b[mask] // s) % s
        z = idx_b[mask] // (s * s)
        pos_of_boid[tmp_b] = x | ((y + 1) << 8) | (z << 16)
        pos_of_boid[tmp_t] = x | (y << 8) | (z << 16)

    return mask.sum().item()


@torch.no_grad()
def swap_pass_z(boid_of_voxel, pos_of_boid, fz, side, parity):
    s = side
    grid = torch.arange(s**3, device=boid_of_voxel.device).reshape(s, s, s)

    idx_f = grid[:, :, parity : s - 1 : 2].reshape(-1)
    idx_b = grid[:, :, parity + 1 : s : 2].reshape(-1)

    boid_f, boid_b = boid_of_voxel[idx_f], boid_of_voxel[idx_b]
    fi, fj = fz[boid_f], fz[boid_b]
    mask = (fi > fj) | ((fi == fj) & (boid_f > boid_b))

    if mask.any():
        tmp_f, tmp_b = boid_f[mask], boid_b[mask]
        boid_of_voxel[idx_f[mask]], boid_of_voxel[idx_b[mask]] = tmp_b, tmp_f

        x = idx_f[mask] % s
        y = (idx_f[mask] // s) % s
        z = idx_f[mask] // (s * s)
        pos_of_boid[tmp_f] = x | (y << 8) | ((z + 1) << 16)
        pos_of_boid[tmp_b] = x | (y << 8) | (z << 16)

    return mask.sum().item()


def run_neighbor_swaps_torch(
    side,
    boid_of_voxel,
    pos_of_boid,
    fx,
    fy,
    fz,
    max_iters=6,
    stop_ratio=0.0,
    verbose=True,
) -> None:
    """
    Optional local refinement for feature / transport modes.

    These passes compare neighboring voxels and swap them when that improves the
    local feature ordering. They only run on CUDA, because this is a bandwidth-heavy
    operation and the CPU version is not worth it here.
    """
    n = side**3

    for iteration in range(max_iters):
        total_swaps = 0
        t0 = time.time()

        for parity in (0, 1):
            total_swaps += swap_pass_x(boid_of_voxel, pos_of_boid, fx, side, parity)
            total_swaps += swap_pass_y(boid_of_voxel, pos_of_boid, fy, side, parity)
            total_swaps += swap_pass_z(boid_of_voxel, pos_of_boid, fz, side, parity)

        t1 = time.time()

        if verbose:
            print(
                f"[swap {iteration + 1}/{max_iters}] "
                f"swaps={total_swaps:,} time={t1 - t0:.3f}s ratio={total_swaps / n:.3e}"
            )

        if stop_ratio > 0 and total_swaps < stop_ratio * n:
            if verbose:
                print("Early stop.")
            break


def transport_mapping(
    side: int,
    mapping_mode: str,
    max_iters: int,
    stop_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a mapping using the feature -> transport -> optional swap pipeline.
    """
    n = side**3
    boid_ids = np.arange(n, dtype=np.uint32)
    fx, fy, fz = compute_features(boid_ids, side, mapping_mode)

    _, _, _, rgb_to_boid, pos_of_boid = three_pass_monotone_transport(fx, fy, fz, side)

    boid_to_rgb = np.empty((n, 3), dtype=np.uint16)
    boid_to_rgb[:, 0] = (pos_of_boid & 0xFF).astype(np.uint16)
    boid_to_rgb[:, 1] = ((pos_of_boid >> 8) & 0xFF).astype(np.uint16)
    boid_to_rgb[:, 2] = ((pos_of_boid >> 16) & 0xFF).astype(np.uint16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    boid_of_voxel = torch.from_numpy(rgb_to_boid.astype(np.int64)).to(device)
    pos_of_boid_t = torch.from_numpy(pos_of_boid.astype(np.int64)).to(device)
    fx_t = torch.from_numpy(fx.astype(np.int64)).to(device)
    fy_t = torch.from_numpy(fy.astype(np.int64)).to(device)
    fz_t = torch.from_numpy(fz.astype(np.int64)).to(device)

    if max_iters > 0 and device.type == "cuda":
        run_neighbor_swaps_torch(
            side,
            boid_of_voxel,
            pos_of_boid_t,
            fx_t,
            fy_t,
            fz_t,
            max_iters=max_iters,
            stop_ratio=stop_ratio,
            verbose=True,
        )

        rgb_to_boid = boid_of_voxel.cpu().numpy().astype(np.uint32)
        pos_of_boid = pos_of_boid_t.cpu().numpy().astype(np.uint32)
        boid_to_rgb[:, 0] = (pos_of_boid & 0xFF).astype(np.uint16)
        boid_to_rgb[:, 1] = ((pos_of_boid >> 8) & 0xFF).astype(np.uint16)
        boid_to_rgb[:, 2] = ((pos_of_boid >> 16) & 0xFF).astype(np.uint16)
    else:
        print("Skipping swap passes (CPU or max-iters=0).")

    return boid_to_rgb, rgb_to_boid.astype(np.uint32)


# --------------------------------------------------------------------------------------
# ------------------------------------- Main -------------------------------------------
# --------------------------------------------------------------------------------------

def build_mapping(
    side: int,
    mapping_mode: str,
    max_iters: int,
    stop_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Dispatch to the correct mapping generator.

    This keeps the important distinction clear:
    - feature / transport modes go through compute_features()
    - direct modes build the mapping themselves
    """
    if mapping_mode == "centered_bgr":
        print("Using direct centered_bgr bit-permutation mapping.")
        description = (
            "Direct bit permutation: C bits 0..7 -> blue low-to-high, "
            "C bits 8..15 -> centered green, C bits 16..23 -> red high-to-low."
        )
        boid_to_rgb, rgb_to_boid = centered_bgr_direct_mapping(side)

    elif mapping_mode == "layered_popcount":
        print("Using direct layered_popcount mapping.")
        description = (
            "Values are grouped by binary popcount, then assigned through diagonal "
            "RGB layers where R+G+B increases."
        )
        boid_to_rgb, rgb_to_boid = layered_popcount_direct_mapping(side)

    elif mapping_mode in TRANSPORT_MAPPING_MODES:
        print(f"Using feature / transport mapping: {mapping_mode}")
        description = (
            f"Feature / transport mapping using {mapping_mode}, followed by optional "
            "CUDA neighbor compare-and-swap refinement."
        )
        boid_to_rgb, rgb_to_boid = transport_mapping(
            side=side,
            mapping_mode=mapping_mode,
            max_iters=max_iters,
            stop_ratio=stop_ratio,
        )

    else:
        raise ValueError(f"Unknown mapping mode: {mapping_mode}")

    return boid_to_rgb, rgb_to_boid, description


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate a bijective color mapping file for the WHCT / XYTC viewer."
    )
    ap.add_argument("--side", type=int, default=256, help="RGB cube side length.")
    ap.add_argument(
        "--mapping-mode",
        "--feature-mode",
        dest="mapping_mode",
        type=str,
        default="rgb_bytes_generalized",
        choices=MAPPING_MODES,
        help=(
            "Mapping strategy to use. --feature-mode is still accepted as an old alias, "
            "but --mapping-mode is the clearer name."
        ),
    )
    ap.add_argument(
        "--max-iters",
        type=int,
        default=6,
        help="Number of CUDA swap-refinement iterations for feature / transport modes.",
    )
    ap.add_argument(
        "--stop-ratio",
        type=float,
        default=0.0,
        help="Early-stop threshold for CUDA swap refinement. 0 disables early stop.",
    )
    ap.add_argument(
        "--preview-csv",
        action="store_true",
        help="Also write a preview CSV. Only allowed when N <= 1,000,000.",
    )
    ap.add_argument("--output", type=str, default="mapping.npz")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    side = int(args.side)
    if side < 2 or side > 256:
        raise ValueError("side must be between 2 and 256. The mapping stores RGB coordinates as byte-sized values.")

    n = side**3
    print(f"Generating mapping for side={side} N={n}")

    boid_to_rgb, rgb_to_boid, description = build_mapping(
        side=side,
        mapping_mode=args.mapping_mode,
        max_iters=args.max_iters,
        stop_ratio=args.stop_ratio,
    )

    save_mapping_npz(
        output=args.output,
        boid_to_rgb=boid_to_rgb,
        rgb_to_boid=rgb_to_boid,
        side=side,
        mapping_mode=args.mapping_mode,
        description=description,
    )

    if args.preview_csv:
        if n > 1_000_000:
            raise ValueError(
                "Refusing to write preview CSV because N > 1,000,000. "
                "For side=256 this would be 16,777,216 rows."
            )
        write_preview_csv(args.output, boid_to_rgb)


if __name__ == "__main__":
    main()
