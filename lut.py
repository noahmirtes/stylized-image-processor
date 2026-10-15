import numpy as np
import itertools

def load_cube_lut(path):
    lut_vals = []
    size = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("LUT_3D_SIZE"):
                size = int(line.split()[-1])
                continue

            if line.startswith(("TITLE", "DOMAIN_MIN", "DOMAIN_MAX")):
                continue

            parts = line.split()
            if len(parts) == 3:
                lut_vals.append([float(parts[0]), float(parts[1]), float(parts[2])])

    if size is None:
        raise ValueError("Missing LUT_3D_SIZE in .cube")

    lut = np.array(lut_vals, dtype=np.float32)
    if lut.shape[0] != size ** 3:
        raise ValueError(f"Expected {size**3} LUT rows, got {lut.shape[0]}")

    lut = lut.reshape((size, size, size, 3))
    return lut, size


def apply_lut(img_u8, lut, size, strength=1.0):
    """
    strength: 0..1
      0 = no LUT (original)
      1 = full LUT
    """
    strength = float(np.clip(strength, 0.0, 1.0))

    # Keep original for blending
    orig = img_u8.astype(np.float32) / 255.0
    x = np.clip(orig, 0.0, 1.0)

    pos = x * (size - 1)
    idx0 = np.floor(pos).astype(int)
    idx1 = np.clip(idx0 + 1, 0, size - 1)
    w = pos - idx0

    r0, g0, b0 = idx0[..., 0], idx0[..., 1], idx0[..., 2]
    r1, g1, b1 = idx1[..., 0], idx1[..., 1], idx1[..., 2]

    c000 = lut[r0, g0, b0]
    c001 = lut[r0, g0, b1]
    c010 = lut[r0, g1, b0]
    c011 = lut[r0, g1, b1]
    c100 = lut[r1, g0, b0]
    c101 = lut[r1, g0, b1]
    c110 = lut[r1, g1, b0]
    c111 = lut[r1, g1, b1]

    wx, wy, wz = w[..., 0:1], w[..., 1:2], w[..., 2:3]

    c00 = c000 * (1 - wz) + c001 * wz
    c01 = c010 * (1 - wz) + c011 * wz
    c10 = c100 * (1 - wz) + c101 * wz
    c11 = c110 * (1 - wz) + c111 * wz

    c0 = c00 * (1 - wy) + c01 * wy
    c1 = c10 * (1 - wy) + c11 * wy

    y = c0 * (1 - wx) + c1 * wx
    y = np.clip(y, 0.0, 1.0)

    if strength < 1.0:
        y = x * (1.0 - strength) + y * strength

    return (y * 255.0 + 0.5).astype(np.uint8)


def auto_fix_lut(lut, size, verbose=False):
    primaries = [
        (size - 1, 0, 0),
        (0, size - 1, 0),
        (0, 0, size - 1),
    ]

    best = None
    best_score = -1.0

    for axis_perm in itertools.permutations([0, 1, 2], 3):
        lut_axes = np.transpose(lut, (axis_perm[0], axis_perm[1], axis_perm[2], 3))

        for out_perm in itertools.permutations([0, 1, 2], 3):
            lut_adj = lut_axes[..., out_perm]

            score = 0.0
            for i, (r, g, b) in enumerate(primaries):
                out = lut_adj[r, g, b]
                score += float(out[i] - 0.5 * (out[(i+1) % 3] + out[(i+2) % 3]))

            if score > best_score:
                best_score = score
                best = (axis_perm, out_perm, lut_adj)

    axis_perm, out_perm, lut_fixed = best

    if verbose:
        print("auto_fix_lut axis_perm =", axis_perm, "out_perm =", out_perm)
        print("R->", lut_fixed[size-1, 0, 0], " G->", lut_fixed[0, size-1, 0], " B->", lut_fixed[0, 0, size-1])

    return lut_fixed