#!/usr/bin/env python3
"""Round-trip all ScanRefer validation scene inputs through two OSGB modes."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import tempfile
import time

from osgb_roundtrip import (
    make_payload,
    read_payload,
    resolve_helper,
    restore_archive,
    run_helper,
    sha256_file,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="DATA_ROOT")
    parser.add_argument(
        "--scene-list",
        default="DATA_ROOT/scanrefer/ScanRefer_filtered_val.txt",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/scanrefer_osgb_eval",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("payload", "geometry"),
        default=("payload", "geometry"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--osg-root")
    parser.add_argument("--helper")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate outputs that already pass basic completeness checks",
    )
    return parser.parse_args()


def ply_vertex_count(path):
    with open(path, "rb") as stream:
        header = bytearray()
        while b"end_header\n" not in header:
            block = stream.read(4096)
            if not block:
                raise RuntimeError("PLY end_header was not found: " + str(path))
            header.extend(block)
            if len(header) > 1024 * 1024:
                raise RuntimeError("PLY header is unexpectedly large: " + str(path))
    for line in header.splitlines():
        if line.startswith(b"element vertex "):
            return int(line.split()[2])
    raise RuntimeError("PLY vertex count was not found: " + str(path))


def output_paths(output_root, scene_id, mode):
    osgb = output_root / (mode + "_osgb") / (scene_id + ".osgb")
    restored = (
        output_root
        / (mode + "_restored")
        / scene_id
        / (scene_id + "_vh_clean_2.ply")
    )
    return osgb, restored


def copy_scene_txt_next_to_osgb(scene_dir, osgb):
    scene_id = scene_dir.name
    source = scene_dir / (scene_id + ".txt")
    if not source.is_file():
        raise FileNotFoundError(str(source))
    osgb.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(osgb.parent / source.name))


def payload_roundtrip(scene_dir, output_root, helper, osg_root, overwrite):
    scene_id = scene_dir.name
    source_ply = scene_dir / (scene_id + "_vh_clean_2.ply")
    osgb, restored_ply = output_paths(output_root, scene_id, "payload")
    copy_scene_txt_next_to_osgb(scene_dir, osgb)
    source_hash = sha256_file(source_ply)
    if (
        not overwrite
        and osgb.is_file()
        and restored_ply.is_file()
        and sha256_file(restored_ply) == source_hash
    ):
        return {
            "mode": "payload",
            "status": "reused",
            "osgb_bytes": osgb.stat().st_size,
            "source_vertices": ply_vertex_count(source_ply),
            "restored_vertices": ply_vertex_count(restored_ply),
            "byte_identical_ply": True,
        }

    osgb.parent.mkdir(parents=True, exist_ok=True)
    restored_ply.parent.mkdir(parents=True, exist_ok=True)
    _, payload = make_payload(scene_dir, profile="inference")
    with tempfile.TemporaryDirectory(prefix="core3dvg_payload_") as temp_dir:
        temp_dir = Path(temp_dir)
        payload_path = temp_dir / "payload.txt"
        temp_osgb = temp_dir / (scene_id + ".osgb")
        extracted_payload = temp_dir / "extracted_payload.txt"
        payload_path.write_bytes(payload)
        run_helper(
            helper,
            osg_root,
            ["pack", source_ply, payload_path, temp_osgb],
            print_output=False,
        )
        run_helper(
            helper,
            osg_root,
            ["unpack", temp_osgb, extracted_payload],
            print_output=False,
        )
        with read_payload(extracted_payload) as archive:
            _, restored_dir = restore_archive(
                archive,
                output_root / "payload_restored",
            )
        restored_ply = restored_dir / (scene_id + "_vh_clean_2.ply")
        shutil.copy2(str(temp_osgb), str(osgb))

    byte_identical = sha256_file(restored_ply) == source_hash
    if not byte_identical:
        raise RuntimeError(scene_id + ": payload-restored PLY is not byte-identical")
    return {
        "mode": "payload",
        "status": "generated",
        "osgb_bytes": osgb.stat().st_size,
        "source_vertices": ply_vertex_count(source_ply),
        "restored_vertices": ply_vertex_count(restored_ply),
        "byte_identical_ply": True,
    }


def geometry_roundtrip(scene_dir, output_root, helper, osg_root, overwrite):
    scene_id = scene_dir.name
    source_ply = scene_dir / (scene_id + "_vh_clean_2.ply")
    osgb, restored_ply = output_paths(output_root, scene_id, "geometry")
    copy_scene_txt_next_to_osgb(scene_dir, osgb)
    if not overwrite and osgb.is_file() and restored_ply.is_file():
        return {
            "mode": "geometry",
            "status": "reused",
            "osgb_bytes": osgb.stat().st_size,
            "source_vertices": ply_vertex_count(source_ply),
            "restored_vertices": ply_vertex_count(restored_ply),
            "byte_identical_ply": sha256_file(restored_ply) == sha256_file(source_ply),
        }

    osgb.parent.mkdir(parents=True, exist_ok=True)
    restored_ply.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="core3dvg_geometry_") as temp_dir:
        temp_osgb = Path(temp_dir) / (scene_id + ".osgb")
        run_helper(
            helper,
            osg_root,
            ["convert-geometry", source_ply, temp_osgb],
            print_output=False,
        )
        run_helper(
            helper,
            osg_root,
            ["extract-geometry", temp_osgb, restored_ply],
            print_output=False,
        )
        shutil.copy2(str(temp_osgb), str(osgb))

    return {
        "mode": "geometry",
        "status": "generated",
        "osgb_bytes": osgb.stat().st_size,
        "source_vertices": ply_vertex_count(source_ply),
        "restored_vertices": ply_vertex_count(restored_ply),
        "byte_identical_ply": sha256_file(restored_ply) == sha256_file(source_ply),
    }


def convert_scene(scene_id, scans_dir, output_root, modes, helper, osg_root, overwrite):
    started = time.time()
    scene_dir = scans_dir / scene_id
    if not scene_dir.is_dir():
        raise FileNotFoundError(str(scene_dir))
    results = {}
    if "payload" in modes:
        results["payload"] = payload_roundtrip(
            scene_dir, output_root, helper, osg_root, overwrite
        )
    if "geometry" in modes:
        results["geometry"] = geometry_roundtrip(
            scene_dir, output_root, helper, osg_root, overwrite
        )
    return {
        "scene_id": scene_id,
        "elapsed_seconds": time.time() - started,
        "modes": results,
    }


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    data_root = Path(args.data_root).resolve()
    scans_dir = data_root / "scans"
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    scene_ids = [
        line.strip()
        for line in Path(args.scene_list).read_text().splitlines()
        if line.strip()
    ]
    if len(scene_ids) != len(set(scene_ids)):
        raise RuntimeError("Scene list contains duplicate IDs")
    if args.limit is not None:
        scene_ids = scene_ids[: args.limit]

    helper, osg_root = resolve_helper(args)
    started = time.time()
    scene_results = {}
    failures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                convert_scene,
                scene_id,
                scans_dir,
                output_root,
                set(args.modes),
                helper,
                osg_root,
                args.overwrite,
            ): scene_id
            for scene_id in scene_ids
        }
        for index, future in enumerate(as_completed(futures), start=1):
            scene_id = futures[future]
            try:
                result = future.result()
                scene_results[scene_id] = result
                mode_status = ", ".join(
                    mode + ":" + value["status"]
                    for mode, value in result["modes"].items()
                )
                print(
                    "[{}/{}] {} ({}, {:.1f}s)".format(
                        index,
                        len(scene_ids),
                        scene_id,
                        mode_status,
                        result["elapsed_seconds"],
                    ),
                    flush=True,
                )
            except Exception as error:
                failures[scene_id] = "{}: {}".format(
                    type(error).__name__,
                    error,
                )
                print(
                    "[{}/{}] {} FAILED: {}".format(
                        index,
                        len(scene_ids),
                        scene_id,
                        failures[scene_id],
                    ),
                    flush=True,
                )

    summary = {
        "scene_count": len(scene_ids),
        "success_count": len(scene_results),
        "failure_count": len(failures),
        "modes": list(args.modes),
        "elapsed_seconds": time.time() - started,
        "data_root": args.data_root,
        "output_root": args.output_root,
        "scenes": {key: scene_results[key] for key in sorted(scene_results)},
        "failures": failures,
    }
    summary_path = output_root / "conversion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in summary if key != "scenes"}, indent=2))
    print("summary=" + str(summary_path))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
