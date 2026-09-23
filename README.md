# Pig cervical spine STL subset verification

This repository contains a pig cervical spine mesh (`Pig C2-C5.stl`), a cropped C3–C4 mesh (`Pig C3-C4.stl`), and a Python script that checks whether the cropped mesh is an exact subset of the original.

## Files

| File | Description |
| --- | --- |
| `Pig C2-C5.stl` | Original binary STL mesh. |
| `Pig C3-C4.stl` | Cropped binary STL mesh. |
| `verify_stl_subset.py` | Compares triangles and vertex positions, then generates an interactive 3D view. |
| `Pig C3-C4_verify.html` | Generated comparison view for the included meshes. Open it in a browser. |

## Run the verification

Requires Python 3 and NumPy, SciPy, and Plotly:

```bash
python3 -m pip install numpy scipy plotly
python3 verify_stl_subset.py "Pig C2-C5.stl" "Pig C3-C4.stl"
```

The script reports exact triangle matches, nearest original-vertex distances, mesh coverage, and bounding boxes. It writes `Pig C3-C4_verify.html`, showing the rest of the original mesh in gray, matching subset triangles in green, and any unmatched triangles in red. The generated HTML loads Plotly from a CDN, so an internet connection is needed to view it.
