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


## Acknowledgements

This code is built on [BUTD-DETR](https://github.com/nickgkan/butd_detr), with components from Group-Free 3D, MDETR, PointNet++, and ScanRefer.
