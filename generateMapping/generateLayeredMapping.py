import argparse
import math
import numpy as np, json

def save_npz_from_mapping(mapping, side, output="25layermapping.npz"):
    """
    mapping: list/ndarray of shape (N,4) as [boid_id, x, y, z]
             or shape (N,2) as [boid_id, color24 (0xRRGGBB)]
    side:    cube side S, so N must be S**3
    """
    mapping = np.asarray(mapping, dtype=np.uint64)
    N_expected = side**3

    if mapping.ndim != 2 or mapping.shape[0] != N_expected:
        raise ValueError(f"Expected {N_expected} rows, got {mapping.shape}")

    # Parse into boid_ids and (x,y,z)
    if mapping.shape[1] == 4:
        boid_ids = mapping[:,0].astype(np.uint32)
        xyz = mapping[:,1:4].astype(np.uint32)
    elif mapping.shape[1] == 2:
        boid_ids = mapping[:,0].astype(np.uint32)
        color24  = mapping[:,1].astype(np.uint32)
        R = (color24 >> 16) & 0xFF
        G = (color24 >>  8) & 0xFF
        B = (color24 >>  0) & 0xFF
        # Map byte 0..255 -> 0..side-1 by floor(val * side / 256)
        xyz = np.stack([(R * side) // 256,
                        (G * side) // 256,
                        (B * side) // 256], axis=1).astype(np.uint32)
    else:
        raise ValueError("mapping rows must be length 4 ([id,x,y,z]) or 2 ([id,color24])")

    # Basic validations
    if boid_ids.min() < 0 or boid_ids.max() >= N_expected:
        raise ValueError("boid_id out of range 0..N-1")
    if np.unique(boid_ids).size != N_expected:
        raise ValueError("boid_id values must be a permutation of 0..N-1 (no dups/missing)")
    if not ((xyz >= 0).all() and (xyz < side).all()):
        raise ValueError("x,y,z must be in [0, side-1]")

    # Build the two lookup tables
    boid_to_rgb = np.empty((N_expected,3), dtype=np.uint16)
    boid_to_rgb[boid_ids] = xyz.astype(np.uint16)

    rgb_to_boid = np.empty(N_expected, dtype=np.uint32)
    idx = (xyz[:,0] + side * (xyz[:,1] + side * xyz[:,2])).astype(np.uint64)

    # Ensure bijectivity (no two boids map to same voxel)
    if np.unique(idx).size != N_expected:
        # Find collisions (optional, but helpful)
        _, counts = np.unique(idx, return_counts=True)
        raise ValueError("Non-bijective mapping: at least one voxel has multiple boids.")

    rgb_to_boid[idx] = boid_ids

    # Save exactly like your generator
    meta = {
        "side": int(side),
        "N": int(N_expected),
        "feature_mode": "EXTERNAL_IMPORT",
        "description": "boid_to_rgb[i]=(x,y,z); rgb_to_boid[linear(x,y,z)]=i"
    }
    np.savez_compressed(output,
        boid_to_rgb=boid_to_rgb,
        rgb_to_boid=rgb_to_boid,
        side=np.array(side, dtype=np.int32),
        meta=np.array(json.dumps(meta), dtype=object)
    )
    print("Saved", output)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", type=int, default=256)
    args = ap.parse_args()

    side = args.side

    N = side**3
    bitlen = math.ceil(math.log2(N))

    layers = []
    for i in range(bitlen+1):
        layers.append([])

    #sorts all numbers up to N by the number of 1s in its binary representation, this creates the layers
    for i in range(N):  #generally 0-24
        num1s = bin(i).count("1")
        layers[num1s].append(i)

    mapping = []
    for n in range(0, 3*side):
        for x in range(0, side):
            # Compute allowable y range
            y_min = max(0, n - x - (side-1))
            y_max = min(side-1, n - x)

            if y_min > y_max:
                continue  # no valid y for this x

            for y in range(y_min, y_max + 1):
                z = n - x - y  # guaranteed 0 <= z < m
                mapping.append([0, x, y, z])

    mIdx = 0
    for i in range(len(layers)):
        for j in range(len(layers[i])):
            mapping[mIdx][0] = layers[i][j]
            mIdx+=1

    print(mapping[len(mapping)-1])


    save_npz_from_mapping(mapping, side)
        
        

if __name__=="__main__":
    main()