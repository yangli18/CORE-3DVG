#!/usr/bin/env python3
"""Run CORE-3DVG inference for one PLY or OSGB scene file and one expression."""

import argparse
import json
from pathlib import Path
import random
import re
import sys
import tempfile
import time

import numpy as np
from plyfile import PlyData
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_runtime import load_checkpoint, parse_args
from main import GroundingEvaluator
from osgb_roundtrip import (
    read_payload,
    resolve_helper,
    restore_archive,
    run_helper,
)
from utils.config import Config


MEAN_RGB = np.array([109.8, 97.2, 83.8], dtype=np.float32) / 256.0
SCENE_ID_PATTERN = re.compile(r"(scene\d{4}_\d{2})")


def parse_single_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input scene .ply or .osgb file")
    parser.add_argument(
        "scene_txt",
        help="Matching ScanNet scene .txt file containing axisAlignment",
    )
    parser.add_argument("--utterance", required=True)
    parser.add_argument("--config", default="configs/scanrefer_cfg.py")
    parser.add_argument("--checkpoint", default="logs/core3dvg_scanrefer.pth")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text-encoder", help="Override RoBERTa directory")
    parser.add_argument(
        "--osgb-mode",
        choices=("geometry", "payload"),
        default="geometry",
        help="How to restore an OSGB input (default: geometry)",
    )
    parser.add_argument("--osg-root", help="Locally extracted OpenSceneGraph prefix")
    parser.add_argument("--helper", help="Compiled osgb_bundle helper")
    parser.add_argument("--output", help="Optional JSON output path")
    return parser.parse_args()


def build_model_args(single_args):
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]]
        args = parse_args()
    finally:
        sys.argv = original_argv
    Config(single_args.config).merge_to_args(args)
    args.checkpoint_path = single_args.checkpoint
    args.distributed = False
    args.local_rank = 0
    args.eval = True
    local_encoder = PROJECT_ROOT / "huggingface" / "roberta-base"
    if single_args.text_encoder:
        args.text_encoder_name_or_path = single_args.text_encoder
    elif local_encoder.is_dir():
        args.text_encoder_name_or_path = str(local_encoder)
    return args


def infer_scene_id(path):
    match = SCENE_ID_PATTERN.search(path.name)
    return match.group(1) if match else path.stem


def alignment_from_txt(path):
    if not Path(path).is_file():
        raise FileNotFoundError(str(path))
    with open(path) as stream:
        for line in stream:
            if line.startswith("axisAlignment"):
                values = line.split("=", 1)[1].split()
                if len(values) != 16:
                    raise ValueError("axisAlignment must contain 16 values")
                return np.asarray(values, dtype=np.float64).reshape(4, 4)
    raise ValueError("axisAlignment was not found in " + str(path))


def load_ply_points(path, alignment):
    data = PlyData.read(str(path)).elements[0].data
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
    if alignment is not None:
        homogeneous = np.ones((xyz.shape[0], 4), dtype=xyz.dtype)
        homogeneous[:, :3] = xyz
        xyz = np.dot(homogeneous, alignment.transpose())[:, :3]
    names = set(data.dtype.names)
    if {"red", "green", "blue"}.issubset(names):
        color = (
            np.stack([data["red"], data["green"], data["blue"]], axis=1) / 256.0
        ).astype(np.float32)
    else:
        color = np.zeros((xyz.shape[0], 3), dtype=np.float32)
    np.random.seed(1184)
    choices = np.random.choice(
        xyz.shape[0],
        50000,
        replace=xyz.shape[0] < 50000,
    )
    return xyz[choices].astype(np.float32), color[choices], xyz.shape[0]


def extract_osgb_input(path, single_args, temp_dir, expected_scene_id):
    helper, osg_root = resolve_helper(single_args)
    if single_args.osgb_mode == "geometry":
        ply_path = Path(temp_dir) / (expected_scene_id + "_vh_clean_2.ply")
        run_helper(
            helper,
            osg_root,
            ["extract-geometry", path, ply_path],
        )
        return ply_path, expected_scene_id, "geometry"

    payload_path = Path(temp_dir) / "payload.txt"
    run_helper(
        helper,
        osg_root,
        ["unpack", path, payload_path],
        print_output=False,
    )
    scans_dir = Path(temp_dir) / "payload_scans"
    with read_payload(payload_path) as archive:
        manifest, scene_dir = restore_archive(archive, scans_dir)
    scene_id = manifest["scene_id"]
    if scene_id != expected_scene_id:
        raise ValueError(
            "OSGB payload scene ID {} does not match {}".format(
                scene_id,
                expected_scene_id,
            )
        )
    ply_path = scene_dir / (scene_id + "_vh_clean_2.ply")
    return ply_path, scene_id, "payload"


def build_point_cloud(xyz, color, args):
    features = [xyz]
    if args.use_color:
        features.append(color - MEAN_RGB)
    if args.use_height:
        floor_height = np.percentile(xyz[:, 2], 0.99)
        features.append((xyz[:, 2:3] - floor_height).astype(np.float32))
    if args.use_multiview:
        raise ValueError("Single-sample inference does not support multiview HDF5")
    return np.concatenate(features, axis=1).astype(np.float32)


def final_scores_and_boxes(end_points):
    boxes = end_points["pred_boxes"].clone()
    boxes[..., 3:] = torch.clamp(boxes[..., 3:], min=1e-6)
    scores = end_points["pred_logits"]
    if "pred_logits_s2" in end_points:
        stage_two_scores = end_points["pred_logits_s2"]
        if (
            stage_two_scores.shape[1] == scores.shape[1]
            and "cand_topk_idx" not in end_points
        ):
            scores = stage_two_scores
        else:
            scores = torch.full_like(scores, -1e8)
            scores.scatter_(1, end_points["cand_topk_idx"], stage_two_scores)
    return scores.squeeze(0).squeeze(-1), boxes.squeeze(0)


def cxcyczwhd_to_xyzxyz(box):
    box = np.asarray(box, dtype=np.float64)
    return np.concatenate([box[:3] - box[3:] * 0.5, box[:3] + box[3:] * 0.5])


def main():
    single_args = parse_single_args()
    args = build_model_args(single_args)
    input_path = Path(single_args.input).resolve()
    scene_txt = Path(single_args.scene_txt).resolve()
    if input_path.suffix.lower() not in (".ply", ".osgb"):
        raise ValueError("Input must have a .ply or .osgb extension")
    if not input_path.is_file():
        raise FileNotFoundError(str(input_path))
    if not scene_txt.is_file():
        raise FileNotFoundError(str(scene_txt))
    scene_id = infer_scene_id(scene_txt)
    alignment = alignment_from_txt(scene_txt)
    if single_args.top_k < 1:
        raise ValueError("--top-k must be positive")
    if single_args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run outside the GPU sandbox")
    device = torch.device(single_args.device)

    random.seed(single_args.seed)
    np.random.seed(single_args.seed)
    torch.manual_seed(single_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(single_args.seed)

    with tempfile.TemporaryDirectory(prefix="core3dvg_single_") as temp_dir:
        if input_path.suffix.lower() == ".osgb":
            ply_path, scene_id, osgb_mode = extract_osgb_input(
                input_path,
                single_args,
                temp_dir,
                scene_id,
            )
        else:
            ply_path = input_path
            osgb_mode = None
        xyz, color, source_point_count = load_ply_points(ply_path, alignment)
        point_cloud = build_point_cloud(xyz, color, args)

    inputs = {
        "point_clouds": torch.from_numpy(point_cloud).unsqueeze(0).to(device),
        "text": [single_args.utterance],
        "train": False,
    }
    model = GroundingEvaluator.get_model(args).to(device)
    load_checkpoint(args, model, single_args.checkpoint)
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.time()
    with torch.no_grad():
        end_points = model(inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.time() - start_time) * 1000.0

    scores, boxes = final_scores_and_boxes(end_points)
    top_k = min(single_args.top_k, scores.numel())
    top_scores, top_indices = torch.topk(scores, k=top_k)
    top_boxes = boxes[top_indices]
    predictions = []
    for rank, (query_index, score, box) in enumerate(
        zip(top_indices.tolist(), top_scores.tolist(), top_boxes.cpu().tolist()),
        start=1,
    ):
        predictions.append(
            {
                "rank": rank,
                "query_index": int(query_index),
                "score_logit": float(score),
                "score_sigmoid": float(torch.sigmoid(torch.tensor(score)).item()),
                "bbox_cxcyczwhd": [float(value) for value in box],
                "bbox_xyzxyz": cxcyczwhd_to_xyzxyz(box).tolist(),
            }
        )

    result = {
        "input": single_args.input,
        "input_format": input_path.suffix.lower().lstrip("."),
        "osgb_restore_mode": osgb_mode,
        "inferred_scene_id": scene_id,
        "axis_alignment_source": single_args.scene_txt,
        "utterance": single_args.utterance,
        "checkpoint": single_args.checkpoint,
        "source_point_count": int(source_point_count),
        "sampled_point_count": int(point_cloud.shape[0]),
        "point_feature_dim": int(point_cloud.shape[1]),
        "inference_ms": elapsed_ms,
        "prediction": predictions[0],
        "top_k_predictions": predictions,
    }
    output_text = json.dumps(result, indent=2) + "\n"
    if single_args.output:
        output_path = Path(single_args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")
    print(output_text, end="")


if __name__ == "__main__":
    main()
