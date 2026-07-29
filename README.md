# Exploiting Contextual Objects and Relations for 3D Visual Grounding

Official code for **Exploiting Contextual Objects and Relations for 3D Visual Grounding**.

## Environment

```bash
conda env create -f environment.yml
conda activate core3dvg
sh init.sh
```

## Data


Set `DATA_ROOT` to the directory containing the prepared benchmark data.

Expected dataset layout:

- `DATA_ROOT/scans/`
- `DATA_ROOT/refer_it_3d/`
- `DATA_ROOT/scanrefer/`
- local `roberta-base` files used by `text_encoder_name_or_path`

ScanNet download helpers are provided in `scripts/`.

## Evaluation

Download checkpoints from [CORE-3DVG](https://huggingface.co/younglight/CORE-3DVG).

### ScanRefer

```bash
python main.py --config=configs/scanrefer_cfg.py --checkpoint_path=path/to/scanrefer_weights.pth --use_eval_dir
```


### NR3D

```bash
python main.py --config=configs/nr3d_cfg.py --checkpoint_path=path/to/nr3d_weights.pth --use_eval_dir
```


### SR3D

```bash
python main.py --config=configs/sr3d_cfg.py --checkpoint_path=path/to/sr3d_weights.pth --use_eval_dir
```

### Single-sample inference

The default example uses a standard `.osgb` scene and restores its Geometry.
It does not require a scene ID argument, a scans directory, `val_v3scans.pkl`,
benchmark annotations, or precomputed Group-Free detector outputs. IoU is not
computed because this deployment path has no ground-truth target:

```bash
CUDA_VISIBLE_DEVICES=7 python tools/infer_single.py \
  artifacts/scanrefer_osgb_eval/geometry_osgb/scene0011_00.osgb \
  artifacts/scanrefer_osgb_eval/geometry_osgb/scene0011_00.txt \
  --config configs/scanrefer_cfg.py \
  --checkpoint logs/core3dvg_scanrefer.pth \
  --utterance "there is a dark brown wooden and leather chair" \
  --output artifacts/single_inference/result.json
```

OSGB input uses standard Geometry by default. For an OSGB generated in payload
mode, select exact embedded restoration explicitly with `--osgb-mode payload`.

Coordinates must match the axis-aligned coordinate frame used during training.
Single-sample inference therefore requires the matching scene `.txt` as its
second positional input and reads `axisAlignment` from that file. It does not
read the repository-wide axis-alignment JSON.

## ScanNet / OSGB round trip

### OpenSceneGraph installation

`tools/osgb_bundle.cpp` performs all OSGB reading and writing. OpenSceneGraph
provides the headers, shared libraries, and `osgdb_osg` plugin used by this
helper. The helper is compiled for the CPU of each deployment machine, so an
amd64 helper cannot be copied to an ARM64/910B machine.

With sudo, install the system packages:

```bash
sudo apt-get update
sudo apt-get install -y g++ openscenegraph libopenscenegraph-dev
```

The first OSGB command then compiles `.tools/osg_bundle` automatically.

Without sudo, this repository includes Ubuntu `.deb` packages for both
deployment environments:

| Host | Included OSG |
| --- | --- |
| Ubuntu 20.04, x86_64/amd64 | 3.6.4 |
| Ubuntu 22.04, aarch64/ARM64 (Ascend 910B) | 3.6.5 |

After `git clone`, run:

```bash
python tools/setup_local_osg.py --clean
```

This command checks package hashes, selects the matching architecture, extracts
the packages to `.tools/osg/root` with `dpkg-deb -x`, compiles the native
`.tools/osg_bundle` with the machine's `g++`, and checks missing shared
libraries with `ldd`. It does not modify `/usr` and does not need root access.
The OSGB tools find this local runtime automatically.

Smoke test:

```bash
python tools/osgb_roundtrip.py restore \
  scene.osgb /tmp/osgb-smoke-test \
  --mode geometry
```

Two conversion modes are supported:

- `geometry`: restore XYZ/RGB from ordinary OSGB geometry. This works on
  platforms that preserve only normal OSGB nodes, but OSG may expand shared
  mesh vertices and change the subsequent 50,000-point sample.
- `payload`: attach a checksummed ZIP/base64 payload to a renderable OSGB. With
  `--profile inference` it embeds only `<scene>_vh_clean_2.ply` and, when
  present, `<scene>.txt`; the restored PLY is byte-identical.

Pack, verify, and restore one inference scene with:

```bash
python tools/osgb_roundtrip.py pack \
  DATA_ROOT/scans/scene0011_00 \
  artifacts/osgb_roundtrip/scene0011_00_payload.osgb \
  --mode payload --profile inference

python tools/osgb_roundtrip.py validate \
  artifacts/osgb_roundtrip/scene0011_00_payload.osgb \
  DATA_ROOT/scans/scene0011_00

python tools/osgb_roundtrip.py restore \
  artifacts/osgb_roundtrip/scene0011_00_payload.osgb \
  artifacts/osgb_roundtrip/payload_restored \
  --mode payload

python tools/osgb_roundtrip.py pack \
  DATA_ROOT/scans/scene0011_00 \
  artifacts/osgb_roundtrip/scene0011_00.osgb \
  --mode geometry --profile inference

python tools/osgb_roundtrip.py restore \
  artifacts/osgb_roundtrip/scene0011_00.osgb \
  artifacts/osgb_roundtrip/geometry_restored \
  --mode geometry
```

Convert all 141 ScanRefer validation scenes through both modes:

```bash
python tools/convert_scanrefer_osgb.py --workers 4
```

Converted ScanRefer data is stored under:

- Geometry OSGB/TXT: `artifacts/scanrefer_osgb_eval/geometry_osgb/`
- Geometry-restored PLY: `artifacts/scanrefer_osgb_eval/geometry_restored/`
- Payload OSGB/TXT: `artifacts/scanrefer_osgb_eval/payload_osgb/`
- Payload-restored PLY/TXT: `artifacts/scanrefer_osgb_eval/payload_restored/`
- Conversion and evaluation summaries:
  `artifacts/scanrefer_osgb_eval/conversion_summary.json` and
  `artifacts/scanrefer_osgb_eval/evaluation_summary.json`

Run the standard 9,508-sample evaluation while retaining every annotation,
target box, and other GT asset from the original `DATA_ROOT`:

```bash
CUDA_VISIBLE_DEVICES=7 python main.py \
  --config configs/scanrefer_cfg.py \
  --checkpoint_path logs/core3dvg_scanrefer.pth \
  --inference_scans_dir artifacts/scanrefer_osgb_eval/payload_restored \
  --eval_output_dir artifacts/scanrefer_osgb_eval/eval_payload

CUDA_VISIBLE_DEVICES=7 python main.py \
  --config configs/scanrefer_cfg.py \
  --checkpoint_path logs/core3dvg_scanrefer.pth \
  --inference_scans_dir artifacts/scanrefer_osgb_eval/geometry_restored \
  --eval_output_dir artifacts/scanrefer_osgb_eval/eval_geometry
```

See [OSGB validation](docs/osgb_roundtrip_validation.md) and
[axis-alignment notes](docs/axis_alignment.md) for results and constraints.


## Acknowledgements

This code is built on [BUTD-DETR](https://github.com/nickgkan/butd_detr), with components from Group-Free 3D, MDETR, PointNet++, and ScanRefer.
