# OSGB round-trip validation

## Evaluation protocol

- Dataset: ScanRefer validation, 141 scenes and 9,508 expressions.
- Only model scene inputs are restored from OSGB.
- Language annotations, target boxes, and metric GT stay in the original
  `DATA_ROOT`/`val_v3scans.pkl`.

## Results

| Model scene input | Acc@0.25 | Acc@0.50 |
| --- | ---: | ---: |
| Original baseline | 0.5677324 | 0.4383677 |
| Embedded inference payload | 0.5677324 | 0.4383677 |
| OSGB Geometry | 0.5648927 | 0.4368952 |
| Original points without axis alignment | 0.0012621 | 0.0000000 |

Payload restoration was lossless for all 141 PLY files. Geometry restoration
expanded 22,224,760 mesh vertices to 29,027,178 vertices (+30.61%), changing
the deterministic 50,000-point sample and causing the small metric difference.

Skipping axis alignment leaves model predictions and aligned GT in different
coordinate frames: only 12/9,508 predictions passed IoU 0.25 and none passed
IoU 0.50.

Machine-readable results:

- [evaluation summary](../artifacts/scanrefer_osgb_eval/evaluation_summary.json)
- [conversion summary](../artifacts/scanrefer_osgb_eval/conversion_summary.json)

Reproduced runs write raw logs under
`artifacts/scanrefer_osgb_eval/eval_*/log.txt`.

## Two OSGB modes

`geometry` stores standard renderable geometry and vertex colors. It is the
default for `infer_single.py` and is the more portable choice.

`payload` additionally embeds a checksummed ZIP/base64 payload in root
user-data. The inference profile contains the source PLY and optional scene
`.txt`. It must be selected explicitly during single-sample inference:

```bash
python tools/infer_single.py scene.osgb scene.txt \
  --osgb-mode payload \
  --utterance "the target description"
```

Platforms may remove custom user-data while optimizing OSGB. Verify a
downloaded file before relying on payload mode.

## Reproduction

```bash
python tools/convert_scanrefer_osgb.py --workers 4

python main.py \
  --config configs/scanrefer_cfg.py \
  --checkpoint_path logs/core3dvg_scanrefer.pth \
  --inference_scans_dir artifacts/scanrefer_osgb_eval/geometry_restored \
  --eval_output_dir artifacts/scanrefer_osgb_eval/eval_geometry
```

See [axis alignment](axis_alignment.md) for the matrix definition and `.txt`
requirements.
