#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generateMapping.py (PyTorch version)

Builds a bijective mapping from all boids (bitstrings) to an S×S×S RGB lattice
using:
  1) 3-pass monotone transport (stable quota fills along X -> Y -> Z), then
  2) PyTorch CUDA neighbor compare-and-swap sweeps (odd-even on x, y, z).

USAGE (examples):
  python generateMapping.py --side 8 --max-iters 4
  python generateMapping.py --side 256 --max-iters 6 --stop-ratio 1e-7

OUTPUT:
  - mapping.npz
      * boid_to_rgb: (N,3) uint16 -> assigned RGB = (x',y',z')
      * rgb_to_boid: (N,) uint32  -> inverse map; index = x + S*(y + S*z)
      * side:        scalar int
      * meta:        dict-like metadata (as JSON string)
  - mapping_preview.csv (optional if N<=1e6)

Notes:
  - Replace `compute_features(...)` with your own 24-bit feature extractor.
  - Neighbor swap passes are O(N) per iteration and bandwidth-bound.
"""

import argparse, os, sys, time, json
import numpy as np
import torch
from numba import njit  # still used for the CPU counting sort

# --------------------------------------------------------------------------------------
# --------------------------- Feature computation (EDIT ME) ----------------------------
# --------------------------------------------------------------------------------------

import math
import numpy as np

def compute_features(boid_ids: np.ndarray, side: int, mode: str = "SPC"):
    """
    Return (fx, fy, fz) in [0, side-1], automatically adapting the bit-width
    to the cube size. We assume N = side**3 boids, so bitlen = ceil(log2(N)).

    Modes:
      - "SPC":  fx = popcount
                fy = average position of 1s   (0..bitlen-1; LSB=0)
                fz = transitions count        (0..bitlen-1)
      - "rgb_bytes_generalized":
                Split the bitstring into 3 contiguous chunks (LSB→MSB)
                as evenly as possible; use each chunk's integer value.
    """
    assert side >= 2
    N = int(side)**3
    # Number of bits needed to represent boid IDs in [0..N-1]
    bitlen = math.ceil(math.log2(N))
    # Safety: cap at 64 to stay in uint64 math; N never exceeds 256^3 so we’re fine
    assert bitlen <= 64, "bitlen too large for uint64 path"

    x = boid_ids.astype(np.uint64) & ((1 << bitlen) - 1)

    def to_side_scale(byte0_255: np.ndarray) -> np.ndarray:
        # Map [0,255] -> [0, side-1] by floor(val * side / 256)
        return (byte0_255.astype(np.uint32) * side // 256).astype(np.uint16)

    if mode == "rgb_bytes_generalized":
        # Split bitlen as evenly as possible into 3 chunks (sizes s1, s2, s3).
        base = bitlen // 3
        rem  = bitlen % 3
        s1 = base + (1 if rem > 0 else 0)
        s2 = base + (1 if rem > 1 else 0)
        s3 = base
        # Note: LSB chunk first
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

        # Scale each chunk to [0,255] using its own max
        def to_byte(vals, vmax):
            # vmax can be 0 if s* = 0; guard to avoid divide-by-zero
            vmax = int(vmax)
            if vmax <= 0:
                return np.zeros_like(vals, dtype=np.uint16)
            return np.floor(vals.astype(np.float64) * (255.0 / vmax) + 0.5).astype(np.uint16)

        fx_raw = to_byte(c1, c1max)
        fy_raw = to_byte(c2, c2max)
        fz_raw = to_byte(c3, c3max)

        return to_side_scale(fx_raw), to_side_scale(fy_raw), to_side_scale(fz_raw)

    elif mode == "SPC":
        # Build (N, bitlen) bit matrix, LSB-first: bit position 0..bitlen-1
        positions = np.arange(bitlen, dtype=np.uint64)
        bits = ((x[:, None] >> positions[None, :]) & 1).astype(np.uint8)

        # S: popcount (0..bitlen)
        pop = bits.sum(axis=1).astype(np.float64)  # float for later division

        # P: average position of 1s (0..bitlen-1); if pop==0 -> 0
        weighted_sum = (bits * positions.astype(np.uint64)[None, :]).sum(axis=1).astype(np.float64)
        avgpos = np.where(pop > 0.0, weighted_sum / pop, 0.0)  # 0..bitlen-1

        # C: transitions across adjacent bits (there are bitlen-1 boundaries)
        if bitlen > 1:
            trans = (bits[:, 1:] != bits[:, :-1]).sum(axis=1).astype(np.float64)  # 0..bitlen-1
        else:
            trans = np.zeros_like(pop, dtype=np.float64)

        # Scale each raw feature to [0,255] using the true max for that feature
        # pop max = bitlen
        fx_raw = np.floor(pop   * (255.0 / max(1, bitlen))       + 0.5).astype(np.uint16)
        # avgpos max = bitlen-1 (if bitlen==1, max=0 -> always 0)
        fy_raw = np.floor(avgpos * (255.0 / max(1, bitlen - 1))  + 0.5).astype(np.uint16)
        # trans max = bitlen-1 (if bitlen==1, max=0 -> always 0)
        fz_raw = np.floor(trans * (255.0 / max(1, bitlen - 1))   + 0.5).astype(np.uint16)

        return to_side_scale(fx_raw), to_side_scale(fy_raw), to_side_scale(fz_raw)

    else:
        raise ValueError(f"Unknown feature mode: {mode}")



# --------------------------------------------------------------------------------------
# ---------------------- 3-pass monotone transport (CPU, O(N)) ------------------------
# --------------------------------------------------------------------------------------

@njit(cache=True)
def _stable_counting_sort_by_key(keys_u16, values_i32, K):
    n = values_i32.size
    counts = np.zeros(K, dtype=np.int64)
    for i in range(n):
        counts[keys_u16[i]] += 1
    offsets = np.empty(K, dtype=np.int64)
    run = 0
    for k in range(K):
        offsets[k] = run
        run += counts[k]
    out = np.empty_like(values_i32)
    for i in range(n):
        key = keys_u16[i]
        pos = offsets[key]
        out[pos] = values_i32[i]
        offsets[key] = pos + 1
    return out

def three_pass_monotone_transport(fx, fy, fz, side: int):
    N = fx.size
    QX, QY = side * side, side
    boids = np.arange(N, dtype=np.int32)

    perm_x = _stable_counting_sort_by_key(fx.astype(np.uint16), boids, side)
    x_assigned = np.empty(N, dtype=np.uint16)
    y_assigned = np.empty(N, dtype=np.uint16)
    z_assigned = np.empty(N, dtype=np.uint16)
    rgb_to_boid = np.empty(N, dtype=np.int32)
    pos_of_boid = np.empty(N, dtype=np.uint32)

    for xi in range(side):
        ids_x = perm_x[xi*QX:(xi+1)*QX]
        x_assigned[ids_x] = xi
        ids_x_sorted_by_y = _stable_counting_sort_by_key(fy[ids_x].astype(np.uint16), ids_x, side)
        for yi in range(side):
            ids_xy = ids_x_sorted_by_y[yi*QY:(yi+1)*QY]
            y_assigned[ids_xy] = yi
            ids_xy_sorted_by_z = _stable_counting_sort_by_key(fz[ids_xy].astype(np.uint16), ids_xy, side)
            for zi, boid in enumerate(ids_xy_sorted_by_z):
                z_assigned[boid] = zi
                idx = int(xi + side * (yi + side * zi))
                rgb_to_boid[idx] = boid
                pos_of_boid[boid] = np.uint32((xi & 0xFF) | ((yi & 0xFF) << 8) | ((zi & 0xFF) << 16))

    return x_assigned, y_assigned, z_assigned, rgb_to_boid, pos_of_boid


# --------------------------------------------------------------------------------------
# ----------------------- PyTorch neighbor swap passes (GPU) ---------------------------
# --------------------------------------------------------------------------------------

@torch.no_grad()
def swap_pass_x(boid_of_voxel, pos_of_boid, fx, side, parity):
    s = side
    # grid[x,y,z]
    grid = torch.arange(s**3, device=boid_of_voxel.device).reshape(s, s, s)
    # valid x are [parity, parity+2, ..., s-2]  (so x+1 is valid)
    idxL = grid[parity:s-1:2, :, :].reshape(-1)
    idxR = grid[parity+1:s:2, :, :].reshape(-1)

    boidL, boidR = boid_of_voxel[idxL], boid_of_voxel[idxR]
    fi, fj = fx[boidL], fx[boidR]
    mask = (fi > fj) | ((fi == fj) & (boidL > boidR))
    if mask.any():
        tmpL, tmpR = boidL[mask], boidR[mask]
        boid_of_voxel[idxL[mask]], boid_of_voxel[idxR[mask]] = tmpR, tmpL
        # update pos
        x = (idxL[mask] % s)
        y = (idxL[mask] // s) % s
        z = (idxL[mask] // (s*s))
        pos_of_boid[tmpL] = (x+1) | (y<<8) | (z<<16)
        pos_of_boid[tmpR] = (x)   | (y<<8) | (z<<16)
    return mask.sum().item()

@torch.no_grad()
def swap_pass_y(boid_of_voxel, pos_of_boid, fy, side, parity):
    s = side
    grid = torch.arange(s**3, device=boid_of_voxel.device).reshape(s, s, s)
    # valid y = [parity, parity+2, ..., s-2]
    idxB = grid[:, parity:s-1:2, :].reshape(-1)
    idxT = grid[:, parity+1:s:2, :].reshape(-1)

    boidB, boidT = boid_of_voxel[idxB], boid_of_voxel[idxT]
    fi, fj = fy[boidB], fy[boidT]
    mask = (fi > fj) | ((fi == fj) & (boidB > boidT))
    if mask.any():
        tmpB, tmpT = boidB[mask], boidT[mask]
        boid_of_voxel[idxB[mask]], boid_of_voxel[idxT[mask]] = tmpT, tmpB
        x = (idxB[mask] % s)
        y = (idxB[mask] // s) % s
        z = (idxB[mask] // (s*s))
        pos_of_boid[tmpB] = (x) | ((y+1)<<8) | (z<<16)
        pos_of_boid[tmpT] = (x) | (y<<8)     | (z<<16)
    return mask.sum().item()

@torch.no_grad()
def swap_pass_z(boid_of_voxel, pos_of_boid, fz, side, parity):
    s = side
    grid = torch.arange(s**3, device=boid_of_voxel.device).reshape(s, s, s)
    # valid z = [parity, parity+2, ..., s-2]
    idxF = grid[:, :, parity:s-1:2].reshape(-1)
    idxB = grid[:, :, parity+1:s:2].reshape(-1)

    boidF, boidB = boid_of_voxel[idxF], boid_of_voxel[idxB]
    fi, fj = fz[boidF], fz[boidB]
    mask = (fi > fj) | ((fi == fj) & (boidF > boidB))
    if mask.any():
        tmpF, tmpB = boidF[mask], boidB[mask]
        boid_of_voxel[idxF[mask]], boid_of_voxel[idxB[mask]] = tmpB, tmpF
        x = (idxF[mask] % s)
        y = (idxF[mask] // s) % s
        z = (idxF[mask] // (s*s))
        pos_of_boid[tmpF] = (x) | (y<<8) | ((z+1)<<16)
        pos_of_boid[tmpB] = (x) | (y<<8) | (z<<16)
    return mask.sum().item()


def run_neighbor_swaps_torch(side, boid_of_voxel, pos_of_boid, fx, fy, fz,
                             max_iters=6, stop_ratio=0.0, verbose=True):
    N = side**3
    for it in range(max_iters):
        total_swaps = 0
        t0 = time.time()
        for parity in (0,1):
            total_swaps += swap_pass_x(boid_of_voxel, pos_of_boid, fx, side, parity)
            total_swaps += swap_pass_y(boid_of_voxel, pos_of_boid, fy, side, parity)
            total_swaps += swap_pass_z(boid_of_voxel, pos_of_boid, fz, side, parity)
        t1 = time.time()
        if verbose:
            print(f"[swap {it+1}/{max_iters}] swaps={total_swaps:,} time={t1-t0:.3f}s ratio={total_swaps/N:.3e}")
        if stop_ratio>0 and total_swaps < stop_ratio*N:
            if verbose: print("Early stop.")
            break


# --------------------------------------------------------------------------------------
# ------------------------------------- MAIN ------------------------------------------
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", type=int, default=256)
    ap.add_argument("--feature-mode", type=str, default="rgb_bytes_generalized", choices=["rgb_bytes_generalized", "SPC"])
    ap.add_argument("--max-iters", type=int, default=6)
    ap.add_argument("--stop-ratio", type=float, default=0.0)
    ap.add_argument("--preview-csv", action="store_true")
    ap.add_argument("--output", type=str, default="mapping.npz")
    args = ap.parse_args()

    side = args.side
    N = side**3
    print(f"Generating mapping for side={side} N={N}")

    # Features
    boid_ids = np.arange(N, dtype=np.uint32)
    fx, fy, fz = compute_features(boid_ids, side, args.feature_mode)

    # 3-pass monotone transport
    _, _, _, rgb_to_boid, pos_of_boid = three_pass_monotone_transport(fx, fy, fz, side)

    # Build boid_to_rgb
    boid_to_rgb = np.empty((N,3),dtype=np.uint16)
    boid_to_rgb[:,0] = (pos_of_boid & 0xFF).astype(np.uint16)
    boid_to_rgb[:,1] = ((pos_of_boid>>8)&0xFF).astype(np.uint16)
    boid_to_rgb[:,2] = ((pos_of_boid>>16)&0xFF).astype(np.uint16)

    # Move to torch for swaps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:",device)
    boid_of_voxel = torch.from_numpy(rgb_to_boid.astype(np.int64)).to(device)
    pos_of_boid_t = torch.from_numpy(pos_of_boid.astype(np.int64)).to(device)
    fx_t = torch.from_numpy(fx.astype(np.int64)).to(device)
    fy_t = torch.from_numpy(fy.astype(np.int64)).to(device)
    fz_t = torch.from_numpy(fz.astype(np.int64)).to(device)

    if args.max_iters>0 and device.type=="cuda":
        run_neighbor_swaps_torch(side, boid_of_voxel, pos_of_boid_t, fx_t, fy_t, fz_t,
                                 max_iters=args.max_iters, stop_ratio=args.stop_ratio, verbose=True)
        # Copy back
        rgb_to_boid = boid_of_voxel.cpu().numpy().astype(np.uint32)
        pos_of_boid = pos_of_boid_t.cpu().numpy().astype(np.uint32)
        boid_to_rgb[:,0] = (pos_of_boid & 0xFF).astype(np.uint16)
        boid_to_rgb[:,1] = ((pos_of_boid>>8)&0xFF).astype(np.uint16)
        boid_to_rgb[:,2] = ((pos_of_boid>>16)&0xFF).astype(np.uint16)
    else:
        print("Skipping swap passes (CPU or max-iters=0).")

    # Save mapping
    meta = {"side": side, "N": int(N), "feature_mode": args.feature_mode,
            "description":"boid_to_rgb[i]=(x,y,z); rgb_to_boid[linear(x,y,z)]=i"}
    np.savez_compressed(args.output,
        boid_to_rgb=boid_to_rgb,
        rgb_to_boid=rgb_to_boid,
        side=np.array(side,dtype=np.int32),
        meta=np.array(json.dumps(meta),dtype=object)
    )
    print("Saved",args.output)

    if args.preview_csv and N<=1_000_000:
        csv_path = os.path.splitext(args.output)[0]+"_preview.csv"
        with open(csv_path,"w") as f:
            f.write("boid_id,R,G,B\n")
            for i in range(N):
                r,g,b=boid_to_rgb[i]
                f.write(f"{i},{r},{g},{b}\n")
        print("Saved",csv_path)

if __name__=="__main__":
    main()
