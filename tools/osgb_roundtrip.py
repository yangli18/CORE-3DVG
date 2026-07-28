#!/usr/bin/env python3
"""Create and restore self-contained OSGB bundles for ScanNet scenes."""

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile


SCHEMA = "core3dvg-scannet-osgb-v1"
REQUIRED_SUFFIXES = (
    "_vh_clean_2.ply",
    "_vh_clean_2.labels.ply",
    "_vh_clean_2.0.010000.segs.json",
    ".aggregation.json",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_osg_root(explicit=None):
    if explicit:
        return Path(explicit).resolve()
    local = Path(__file__).resolve().parents[1] / ".tools" / "osg" / "root"
    return local if local.exists() else None


def osg_environment(osg_root):
    env = os.environ.copy()
    if osg_root is None:
        return env
    library_dir = osg_root / "usr" / "lib" / "x86_64-linux-gnu"
    plugin_dir = library_dir / "osgPlugins-3.6.4"
    env["LD_LIBRARY_PATH"] = str(library_dir) + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    env["OSG_LIBRARY_PATH"] = str(plugin_dir)
    return env


def compile_helper(osg_root, helper_path):
    source = Path(__file__).with_name("osgb_bundle.cpp")
    include_dir = osg_root / "usr" / "include" if osg_root else Path("/usr/include")
    library_dir = (
        osg_root / "usr" / "lib" / "x86_64-linux-gnu"
        if osg_root
        else Path("/usr/lib/x86_64-linux-gnu")
    )
    helper_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "g++",
        "-std=c++11",
        "-O2",
        "-I",
        str(include_dir),
        str(source),
        "-L",
        str(library_dir),
        "-Wl,-rpath," + str(library_dir),
        "-losgDB",
        "-losg",
        "-lOpenThreads",
        "-o",
        str(helper_path),
    ]
    subprocess.run(command, check=True)


def resolve_helper(args):
    osg_root = find_osg_root(args.osg_root)
    if args.helper:
        helper = Path(args.helper).resolve()
    else:
        helper = Path(__file__).resolve().parents[1] / ".tools" / "osg_bundle"
    source = Path(__file__).with_name("osgb_bundle.cpp")
    if not helper.exists() or helper.stat().st_mtime < source.stat().st_mtime:
        if osg_root is None and not Path("/usr/include/osg/Group").exists():
            raise RuntimeError(
                "OpenSceneGraph headers were not found. Install "
                "'openscenegraph' and 'libopenscenegraph-dev', or pass --osg-root."
            )
        compile_helper(osg_root, helper)
    return helper, osg_root


def scene_files(scene_dir, profile="full"):
    scene_dir = Path(scene_dir).resolve()
    scene_id = scene_dir.name
    if profile == "inference":
        required = [scene_dir / (scene_id + "_vh_clean_2.ply")]
    elif profile == "full":
        required = [scene_dir / (scene_id + suffix) for suffix in REQUIRED_SUFFIXES]
    else:
        raise ValueError("Unsupported bundle profile: " + profile)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required ScanNet files: " + ", ".join(missing))
    optional = scene_dir / (scene_id + ".txt")
    return scene_id, required + ([optional] if optional.is_file() else [])


def make_payload(scene_dir, profile="full"):
    scene_id, files = scene_files(scene_dir, profile=profile)
    file_entries = [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    manifest = {
        "schema": SCHEMA,
        "profile": profile,
        "scene_id": scene_id,
        "files": file_entries,
    }
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(
        archive_buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
        )
        for path in files:
            archive.write(path, "files/" + path.name)
    return manifest, base64.b64encode(archive_buffer.getvalue())


def read_payload(payload_path):
    try:
        archive_bytes = base64.b64decode(Path(payload_path).read_bytes(), validate=True)
    except Exception as error:
        raise RuntimeError("Invalid base64 bundle payload") from error
    return zipfile.ZipFile(io.BytesIO(archive_bytes), "r")


def validate_archive_paths(archive):
    for name in archive.namelist():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError("Unsafe path in OSGB payload: " + name)


def restore_archive(archive, output_scans_dir):
    validate_archive_paths(archive)
    manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise RuntimeError("Unsupported payload schema: " + str(manifest.get("schema")))
    scene_dir = Path(output_scans_dir).resolve() / manifest["scene_id"]
    scene_dir.mkdir(parents=True, exist_ok=True)
    for entry in manifest["files"]:
        file_name = entry["name"]
        if (
            not file_name
            or Path(file_name).name != file_name
            or file_name in (".", "..")
        ):
            raise RuntimeError("Unsafe file name in OSGB manifest: " + file_name)
        archive_name = "files/" + file_name
        data = archive.read(archive_name)
        output = scene_dir / file_name
        output.write_bytes(data)
        if output.stat().st_size != entry["size"] or sha256_file(output) != entry["sha256"]:
            raise RuntimeError("Restored file failed verification: " + str(output))
    return manifest, scene_dir


def run_helper(helper, osg_root, arguments, print_output=True):
    result = subprocess.run(
        [str(helper)] + [str(value) for value in arguments],
        check=True,
        env=osg_environment(osg_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if print_output and result.stderr:
        print(result.stderr, end="")
    if print_output and result.stdout:
        print(result.stdout, end="")
    return result


def unlink_if_exists(path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def command_pack(args):
    helper, osg_root = resolve_helper(args)
    scene_dir = Path(args.scene_dir).resolve()
    scene_id, _ = scene_files(scene_dir, profile=args.profile)
    mesh_path = scene_dir / (scene_id + "_vh_clean_2.ply")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "geometry":
        run_helper(helper, osg_root, ["convert-geometry", mesh_path, output])
        print(
            json.dumps(
                {
                    "output": str(output),
                    "mode": "geometry",
                    "scene_id": scene_id,
                },
                indent=2,
            )
        )
        return

    manifest, payload = make_payload(scene_dir, profile=args.profile)
    with tempfile.NamedTemporaryFile(suffix=".payload", delete=False) as stream:
        payload_path = Path(stream.name)
        stream.write(payload)
    try:
        run_helper(helper, osg_root, ["pack", mesh_path, payload_path, output])
        run_helper(helper, osg_root, ["inspect", output])
    finally:
        unlink_if_exists(payload_path)
    print(json.dumps({"output": str(output), "manifest": manifest}, indent=2))


def command_restore(args):
    helper, osg_root = resolve_helper(args)
    input_path = Path(args.input).resolve()
    if args.mode == "geometry":
        scene_id = input_path.stem
        scene_dir = Path(args.output_scans_dir).resolve() / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        output = scene_dir / (scene_id + "_vh_clean_2.ply")
        run_helper(helper, osg_root, ["extract-geometry", input_path, output])
        print(
            json.dumps(
                {
                    "scene_dir": str(scene_dir),
                    "scene_id": scene_id,
                    "mode": "geometry",
                    "point_file": str(output),
                },
                indent=2,
            )
        )
        return

    with tempfile.NamedTemporaryFile(suffix=".payload", delete=False) as stream:
        payload_path = Path(stream.name)
    try:
        run_helper(helper, osg_root, ["unpack", input_path, payload_path])
        with read_payload(payload_path) as archive:
            manifest, scene_dir = restore_archive(archive, args.output_scans_dir)
    finally:
        unlink_if_exists(payload_path)
    print(
        json.dumps(
            {
                "scene_dir": str(scene_dir),
                "verified_files": len(manifest["files"]),
                "scene_id": manifest["scene_id"],
            },
            indent=2,
        )
    )


def command_validate(args):
    helper, osg_root = resolve_helper(args)
    input_path = Path(args.input).resolve()
    source_dir = Path(args.source_scene_dir).resolve()
    with tempfile.TemporaryDirectory(prefix="core3dvg_osgb_validate_") as temp_dir:
        payload_path = Path(temp_dir) / "payload.txt"
        scans_dir = Path(temp_dir) / "scans"
        run_helper(helper, osg_root, ["inspect", input_path])
        run_helper(helper, osg_root, ["unpack", input_path, payload_path])
        with read_payload(payload_path) as archive:
            manifest, restored_dir = restore_archive(archive, scans_dir)
        mismatches = []
        for entry in manifest["files"]:
            source = source_dir / entry["name"]
            restored = restored_dir / entry["name"]
            if (
                not source.is_file()
                or sha256_file(source) != entry["sha256"]
                or sha256_file(restored) != entry["sha256"]
            ):
                mismatches.append(entry["name"])
        if mismatches:
            raise RuntimeError("Round-trip mismatch: " + ", ".join(mismatches))
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "scene_id": manifest["scene_id"],
                    "byte_identical_files": len(manifest["files"]),
                    "osgb_bytes": input_path.stat().st_size,
                },
                indent=2,
            )
        )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osg-root", help="Locally extracted OpenSceneGraph prefix")
    parser.add_argument("--helper", help="Compiled osgb_bundle helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack_parser = subparsers.add_parser("pack", help="Pack one ScanNet scene")
    pack_parser.add_argument("scene_dir")
    pack_parser.add_argument("output")
    pack_parser.add_argument(
        "--mode",
        choices=("payload", "geometry"),
        default="payload",
        help="Self-contained payload bundle or geometry-only OSGB",
    )
    pack_parser.add_argument(
        "--profile",
        choices=("full", "inference"),
        default="full",
        help="Files embedded by payload mode",
    )
    pack_parser.set_defaults(func=command_pack)

    restore_parser = subparsers.add_parser("restore", help="Restore a scene from OSGB")
    restore_parser.add_argument("input")
    restore_parser.add_argument("output_scans_dir")
    restore_parser.add_argument(
        "--mode",
        choices=("payload", "geometry"),
        default="payload",
    )
    restore_parser.set_defaults(func=command_restore)

    validate_parser = subparsers.add_parser(
        "validate", help="Check a bundle against its source scene"
    )
    validate_parser.add_argument("input")
    validate_parser.add_argument("source_scene_dir")
    validate_parser.set_defaults(func=command_validate)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
