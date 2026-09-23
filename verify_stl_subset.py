"""
Verify that a cropped STL (e.g. C3-C4) was taken from an original STL (e.g. C2-C5).

Usage:
    python verify_stl_subset.py "original.stl" "extract_from_original.stl"

Requires: pip install numpy scipy plotly
Output:
    - console report (exact triangle match + distance statistics)
    - <subset>_verify.html : interactive 3D view (open in browser)
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import plotly.graph_objects as go


def read_binary_stl(path):
    """Return triangles as float32 array (n, 3, 3), read bit-for-bit from the file."""
    raw = Path(path).read_bytes()
    n = int(np.frombuffer(raw, "<u4", 1, 80)[0])
    if len(raw) != 84 + 50 * n:
        raise ValueError(f"{path} is not a binary STL (size mismatch)")
    dt = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    return np.frombuffer(raw, dt, n, 84)["v"].copy()


def triangle_keys(tris_a, tris_b):
    """Order-independent integer key per triangle, shared vertex table for both meshes."""
    verts = np.concatenate([tris_a.reshape(-1, 3), tris_b.reshape(-1, 3)])
    vkey = np.ascontiguousarray(verts).view("V12").ravel()   # exact float32 bytes
    _, vid = np.unique(vkey, return_inverse=True)
    vid = np.sort(vid.reshape(-1, 3).astype(np.int64), axis=1)  # vertex order doesn't matter
    n = vid.max() + 1
    if n ** 3 >= 2 ** 63:
        raise ValueError("too many unique vertices for int64 key")
    key = (vid[:, 0] * n + vid[:, 1]) * n + vid[:, 2]
    return key[: len(tris_a)], key[len(tris_a):]


def main(original_path, subset_path):
    A = read_binary_stl(original_path)
    B = read_binary_stl(subset_path)
    print(f"original : {len(A):>9,d} triangles  ({original_path})")
    print(f"subset   : {len(B):>9,d} triangles  ({subset_path})")

    # 1) exact match: is every subset triangle identical (bitwise float32) to an original triangle?
    ka, kb = triangle_keys(A, B)
    exact = np.isin(kb, ka)
    print(f"\n[1] Exact triangle match: {exact.sum():,d} / {len(B):,d} "
          f"({100 * exact.mean():.4f}%)")

    # 2) geometric difference: distance of each subset vertex to nearest original vertex
    Av = np.unique(A.reshape(-1, 3), axis=0)
    Bv = B.reshape(-1, 3)
    dist, _ = cKDTree(Av).query(Bv)
    print(f"[2] Subset vertex -> nearest original vertex distance (mm): "
          f"max={dist.max():.6f}, mean={dist.mean():.6f}, "
          f"#>0 = {(dist > 0).sum():,d}")

    # 3) coverage: how much of the original the subset occupies
    print(f"[3] Subset = {100 * len(B) / len(A):.2f}% of original triangles")
    print(f"    original bounds: {A.reshape(-1,3).min(0).round(2)} -> {A.reshape(-1,3).max(0).round(2)}")
    print(f"    subset   bounds: {Bv.min(0).round(2)} -> {Bv.max(0).round(2)}")

    verdict = "IDENTICAL SUBSET of the original" if exact.all() else "NOT an exact subset"
    print(f"\nVerdict: {verdict}")

    # 4) visualization: original (grey, remaining part) + subset colored by match
    rng = np.random.default_rng(0)
    rest = A[~np.isin(ka, kb)]
    rest = rest[rng.choice(len(rest), min(len(rest), 150_000), replace=False)]
    fig = go.Figure()
    fig.add_trace(mesh_trace(rest, "original (other vertebrae)", "lightgrey", 0.25))
    show = rng.choice(len(B), min(len(B), 250_000), replace=False)
    fig.add_trace(mesh_trace(B[show][exact[show]], "subset: exact match", "seagreen", 1.0))
    if (~exact).any():
        fig.add_trace(mesh_trace(B[~exact], "subset: NOT in original", "red", 1.0))
    fig.update_layout(title=f"{Path(subset_path).name} vs {Path(original_path).name} — {verdict}",
                      scene=dict(aspectmode="data"))
    out = Path(subset_path).with_name(Path(subset_path).stem + "_verify.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"3D view saved: {out}")


def mesh_trace(tris, name, color, opacity):
    v = tris.reshape(-1, 3)
    i = np.arange(0, len(v), 3)
    return go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=i, j=i + 1, k=i + 2,
                     color=color, opacity=opacity, name=name, showlegend=True, flatshading=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
