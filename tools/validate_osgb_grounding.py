#!/usr/bin/env python3
"""Compare CORE-3DVG inference before and after a ScanNet OSGB round trip."""

import argparse
import copy
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_runtime import load_checkpoint, parse_args
from main import GroundingEvaluator
from models.box_ops import _batch_iou3d_par, box_cxcyczwhd_to_xyzxyz
from src.grounding_dataset import GroundingDataset
from src.visual_data_handlers import Scan
from utils.config import Config
from utils.misc import collate_fn_w_span_indices


def parse_validation_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scanrefer_cfg.py")
    parser.add_argument("--checkpoint", default="logs/core3dvg_scanrefer.pth")
    parser.add_argument("--data-root", default="DATA_ROOT/")
    parser.add_argument(
        "--restored-scans-dir",
        default="artifacts/osgb_roundtrip/restored_scans",
    )
    parser.add_argument("--scene-id", default="scene0011_00")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        default="artifacts/osgb_roundtrip/grounding_comparison.json",
    )
    return parser.parse_args()


def build_model_args(validation_args):
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]]
        args = parse_args()
    finally:
        sys.argv = original_argv
    Config(validation_args.config).merge_to_args(args)
    args.data_root = validation_args.data_root
    args.checkpoint_path = validation_args.checkpoint
    args.text_encoder_name_or_path = str(
        (PROJECT_ROOT / "huggingface" / "roberta-base").resolve()
    )
    args.distributed = False
    args.local_rank = 0
    args.eval = True
    return args


def compare_scans(source_scan, restored_scan):
    if len(source_scan.three_d_objects) != len(restored_scan.three_d_objects):
        raise AssertionError("Object count changed after OSGB round trip")
    bbox_errors = []
    object_points_equal = True
    object_ids_equal = True
    object_labels_equal = True
    for index, (source_object, restored_object) in enumerate(
        zip(source_scan.three_d_objects, restored_scan.three_d_objects)
    ):
        object_points_equal &= np.array_equal(
            source_object["points"], restored_object["points"]
        )
        object_ids_equal &= (
            source_object["object_id"] == restored_object["object_id"]
        )
        object_labels_equal &= (
            source_object["instance_label"] == restored_object["instance_label"]
        )
        bbox_errors.append(
            np.max(
                np.abs(
                    source_scan.get_object_bbox(index)
                    - restored_scan.get_object_bbox(index)
                )
            )
        )
    return {
        "point_count": int(source_scan.pc.shape[0]),
        "object_count": len(source_scan.three_d_objects),
        "xyz_exact": bool(np.array_equal(source_scan.pc, restored_scan.pc)),
        "xyz_max_abs_error": float(
            np.max(np.abs(source_scan.pc - restored_scan.pc))
        ),
        "color_exact": bool(np.array_equal(source_scan.color, restored_scan.color)),
        "color_max_abs_error": float(
            np.max(np.abs(source_scan.color - restored_scan.color))
        ),
        "sampling_choices_exact": bool(
            np.array_equal(source_scan.choices, restored_scan.choices)
        ),
        "object_ids_exact": bool(object_ids_equal),
        "object_labels_exact": bool(object_labels_equal),
        "object_point_indices_exact": bool(object_points_equal),
        "bbox_max_abs_error": float(max(bbox_errors, default=0.0)),
    }


def make_sample(dataset, index, scan, seed):
    scene_id = dataset.annos[index]["scan_id"]
    saved_scan = dataset.scans[scene_id]
    saved_annotation = copy.deepcopy(dataset.annos[index])
    try:
        dataset.scans[scene_id] = scan
        dataset.annos[index] = copy.deepcopy(saved_annotation)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return dataset[index]
    finally:
        dataset.scans[scene_id] = saved_scan
        dataset.annos[index] = saved_annotation


def compare_sample_dicts(source_samples, restored_samples):
    max_abs_error = {}
    unequal_keys = set()
    for source, restored in zip(source_samples, restored_samples):
        for key in source:
            source_value = source[key]
            restored_value = restored[key]
            if isinstance(source_value, np.ndarray):
                if not np.array_equal(source_value, restored_value):
                    unequal_keys.add(key)
                if np.issubdtype(source_value.dtype, np.number):
                    error = float(
                        np.max(
                            np.abs(
                                source_value.astype(np.float64)
                                - restored_value.astype(np.float64)
                            )
                        )
                    )
                    max_abs_error[key] = max(max_abs_error.get(key, 0.0), error)
            elif source_value != restored_value:
                unequal_keys.add(key)
    return {
        "all_sample_fields_exact": not unequal_keys,
        "unequal_keys": sorted(unequal_keys),
        "max_abs_error_by_numeric_key": max_abs_error,
    }


def to_device(batch, device):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


def select_predictions(end_points):
    boxes = end_points["pred_boxes"].clone()
    boxes[..., 3:] = torch.clamp(boxes[..., 3:], min=1e-6)
    logits = end_points["pred_logits"]
    if "pred_logits_s2" in end_points:
        stage_two_logits = end_points["pred_logits_s2"]
        if (
            stage_two_logits.shape[1] == logits.shape[1]
            and "cand_topk_idx" not in end_points
        ):
            logits = stage_two_logits
        else:
            selected_candidates = end_points["cand_topk_idx"]
            logits = torch.full_like(logits, -1e8)
            logits.scatter_(1, selected_candidates, stage_two_logits)
    selected_indices = logits.argmax(dim=1)
    if selected_indices.ndim == 2 and selected_indices.shape[-1] == 1:
        selected_indices = selected_indices.squeeze(-1)
    selected_boxes = boxes.gather(
        1,
        selected_indices.reshape(-1, 1, 1).expand(-1, 1, boxes.shape[-1]),
    ).squeeze(1)
    return selected_indices, selected_boxes, logits


def infer(model, evaluator, batch, device):
    batch = to_device(batch, device)
    inputs = evaluator._get_inputs(batch)
    inputs["train"] = False
    with torch.no_grad():
        end_points = model(inputs)
    selected_indices, selected_boxes, logits = select_predictions(end_points)
    gt_boxes = torch.cat([batch["center_label"], batch["size_gts"]], dim=-1)[:, 0]
    ious, _ = _batch_iou3d_par(
        box_cxcyczwhd_to_xyzxyz(selected_boxes.unsqueeze(1)),
        box_cxcyczwhd_to_xyzxyz(gt_boxes.unsqueeze(1)),
    )
    return {
        "indices": selected_indices.detach().cpu(),
        "boxes": selected_boxes.detach().cpu(),
        "logits": logits.detach().cpu(),
        "ious": ious.reshape(-1).detach().cpu(),
    }


def main():
    validation_args = parse_validation_args()
    args = build_model_args(validation_args)
    if validation_args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run outside the GPU sandbox")
    device = torch.device(validation_args.device)

    source_scans_dir = Path(validation_args.data_root) / "scans"
    restored_scans_dir = Path(validation_args.restored_scans_dir)
    source_scan = Scan(validation_args.scene_id, str(source_scans_dir), True)
    restored_scan = Scan(validation_args.scene_id, str(restored_scans_dir), True)
    scan_comparison = compare_scans(source_scan, restored_scan)

    dataset = GroundingDataset(
        dataset_dict={args.test_dataset: 1},
        test_dataset=args.test_dataset,
        split="val",
        use_color=args.use_color,
        use_height=args.use_height,
        data_path=args.data_root,
        detect_intermediate=args.detect_intermediate,
        use_multiview=args.use_multiview,
        extra_dataset_cfg=args.extra_dataset_cfg,
        text_encoder_name_or_path=args.text_encoder_name_or_path,
    )
    sample_indices = [
        index
        for index, annotation in enumerate(dataset.annos)
        if annotation["scan_id"] == validation_args.scene_id
    ][: validation_args.samples]
    if not sample_indices:
        raise RuntimeError(
            "{} has no {} validation annotations".format(
                validation_args.scene_id, args.test_dataset
            )
        )

    source_samples = []
    restored_samples = []
    for order, index in enumerate(sample_indices):
        seed = 1000 + order
        source_samples.append(make_sample(dataset, index, source_scan, seed))
        restored_samples.append(make_sample(dataset, index, restored_scan, seed))
    sample_comparison = compare_sample_dicts(source_samples, restored_samples)

    source_batch = collate_fn_w_span_indices(source_samples)
    restored_batch = collate_fn_w_span_indices(restored_samples)
    model = GroundingEvaluator.get_model(args).to(device)
    load_checkpoint(args, model, validation_args.checkpoint)
    model.eval()
    evaluator = GroundingEvaluator

    torch.manual_seed(0)
    source_predictions = infer(model, evaluator, source_batch, device)
    torch.manual_seed(0)
    restored_predictions = infer(model, evaluator, restored_batch, device)
    prediction_comparison = {
        "selected_query_indices_exact": bool(
            torch.equal(
                source_predictions["indices"], restored_predictions["indices"]
            )
        ),
        "predicted_boxes_max_abs_error": float(
            torch.max(
                torch.abs(
                    source_predictions["boxes"] - restored_predictions["boxes"]
                )
            ).item()
        ),
        "prediction_logits_max_abs_error": float(
            torch.max(
                torch.abs(
                    source_predictions["logits"] - restored_predictions["logits"]
                )
            ).item()
        ),
        "iou_max_abs_error": float(
            torch.max(
                torch.abs(
                    source_predictions["ious"] - restored_predictions["ious"]
                )
            ).item()
        ),
        "source_ious": source_predictions["ious"].tolist(),
        "restored_ious": restored_predictions["ious"].tolist(),
        "source_selected_query_indices": source_predictions["indices"].tolist(),
        "restored_selected_query_indices": restored_predictions["indices"].tolist(),
    }
    pass_status = (
        scan_comparison["xyz_exact"]
        and scan_comparison["color_exact"]
        and scan_comparison["object_point_indices_exact"]
        and scan_comparison["bbox_max_abs_error"] == 0.0
        and sample_comparison["all_sample_fields_exact"]
        and prediction_comparison["selected_query_indices_exact"]
        and prediction_comparison["predicted_boxes_max_abs_error"] == 0.0
        and prediction_comparison["prediction_logits_max_abs_error"] == 0.0
        and prediction_comparison["iou_max_abs_error"] == 0.0
    )
    report = {
        "status": "PASS" if pass_status else "FAIL",
        "dataset": args.test_dataset,
        "scene_id": validation_args.scene_id,
        "sample_count": len(sample_indices),
        "sample_indices": sample_indices,
        "scan_comparison": scan_comparison,
        "model_input_comparison": sample_comparison,
        "prediction_comparison": prediction_comparison,
    }
    output = Path(validation_args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not pass_status:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
