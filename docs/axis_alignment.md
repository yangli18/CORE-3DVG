# ScanNet axis alignment

## What the matrix means

Raw ScanNet reconstruction coordinates have a scene-dependent horizontal
orientation and origin. `axisAlignment` is a row-major 4×4 homogeneous
transformation:

```text
r00 r01 r02 tx
r10 r11 r12 ty
r20 r21 r22 tz
0   0   0   1
```

The upper-left 3×3 part rotates the scene so major room directions align with
the X/Y axes. The last column translates the origin. For a point
`p = [x, y, z, 1]`, the loader computes `p_aligned = M × p`; the NumPy
implementation uses the equivalent row-vector expression `p @ M.T`.

This common coordinate frame is required because ScanRefer target boxes are
axis-aligned boxes computed after the same transform. Spatial relations and
the trained box regressor also expect that convention.

## Scene `.txt` contents

A ScanNet `<scene>.txt` contains:

- `axisAlignment`: scene-to-aligned 4×4 transform;
- color/depth image dimensions;
- camera intrinsics (`fx`, `fy`, `mx`, `my`);
- color-to-depth extrinsics;
- frame and IMU counts;
- scene type.

CORE-3DVG point-cloud inference uses only `axisAlignment`. The camera and frame
fields are relevant to RGB-D projection workflows, not this grounding path.

## Repository JSON versus `.txt`

`data/meta_data/scans_axis_alignment_matrices.json` is a scene-ID-to-matrix
index extracted from the original per-scene `.txt` files. It contains 1,513
scenes.

All 1,513 indexed matrices matched their available original `.txt` files
exactly. In particular, for all 141 ScanRefer validation scenes:

- the `.txt` file was present;
- all 16 JSON and `.txt` values matched exactly;
- maximum absolute difference was 0.

Dataset evaluation continues to use this indexed JSON. Single-sample inference
does not: it requires the matching `.txt` explicitly, which avoids dependence
on a repository-wide scene index.

## Measured effect of omitting alignment

The full 9,508-sample ScanRefer evaluation produced:

| Input | Acc@0.25 | Acc@0.50 |
| --- | ---: | ---: |
| Correct axis alignment | 0.5677324 | 0.4383677 |
| No axis alignment | 0.0012621 | 0.0000000 |

Without alignment, only 12 samples passed IoU 0.25 and none passed IoU 0.50.
This is primarily a coordinate-frame mismatch: predictions are expressed in
raw scene coordinates while GT boxes remain in aligned coordinates.
