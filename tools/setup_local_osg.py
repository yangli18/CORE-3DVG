#!/usr/bin/env python3
"""Extract the vendored OSG packages and build the native OSGB helper."""

import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "third_party" / "openscenegraph"
SUPPORTED = {
    ("20.04", "x86_64"): ("ubuntu20.04-amd64", "amd64"),
    ("22.04", "aarch64"): ("ubuntu22.04-arm64", "arm64"),
}


def normalized_machine():
    machine = platform.machine().lower()
    return {
        "amd64": "x86_64",
        "x86_64": "x86_64",
        "arm64": "aarch64",
        "aarch64": "aarch64",
    }.get(machine, machine)


def ubuntu_version():
    values = {}
    with open("/etc/os-release", encoding="utf-8") as stream:
        for line in stream:
            key, separator, value = line.strip().partition("=")
            if separator:
                values[key] = value.strip("\"'")
    return values.get("VERSION_ID", "")


def package_set():
    host = (ubuntu_version(), normalized_machine())
    try:
        directory, deb_arch = SUPPORTED[host]
    except KeyError as error:
        supported = ", ".join(
            f"Ubuntu {version} {machine}" for version, machine in SUPPORTED
        )
        raise RuntimeError(
            f"Unsupported host Ubuntu {host[0] or '?'} {host[1]}; "
            f"vendored packages support: {supported}"
        ) from error
    return PACKAGE_ROOT / directory, deb_arch


def package_field(package, field):
    return subprocess.check_output(
        ["dpkg-deb", "-f", str(package), field], text=True
    ).strip()


def extract_packages(package_dir, expected_arch, output_root):
    packages = sorted(package_dir.glob("*.deb"))
    if not packages:
        raise RuntimeError("No vendored .deb files found in " + str(package_dir))
    output_root.mkdir(parents=True, exist_ok=True)
    for package in packages:
        architecture = package_field(package, "Architecture")
        if architecture not in (expected_arch, "all"):
            raise RuntimeError(
                f"{package.name} is {architecture}, expected {expected_arch}"
            )
        print("Extracting", package.name)
        subprocess.run(
            ["dpkg-deb", "-x", str(package), str(output_root)], check=True
        )


def verify_checksums(package_dir):
    checksum_file = package_dir / "SHA256SUMS"
    subprocess.run(
        ["sha256sum", "--check", checksum_file.name],
        cwd=package_dir,
        check=True,
    )


def check_native_dependencies(helper, osg_root):
    environment = os.environ.copy()
    library_dir = osgb_roundtrip.osg_library_dir(osg_root)
    environment["LD_LIBRARY_PATH"] = str(library_dir) + (
        ":" + environment["LD_LIBRARY_PATH"]
        if environment.get("LD_LIBRARY_PATH")
        else ""
    )
    result = subprocess.run(
        ["ldd", str(helper)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    missing = [
        line.strip() for line in result.stdout.splitlines() if "not found" in line
    ]
    if missing:
        raise RuntimeError(
            "The host is missing runtime libraries required by OSG:\n  "
            + "\n  ".join(missing)
        )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a no-sudo OpenSceneGraph runtime for this host"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / ".tools" / "osg" / "root",
    )
    parser.add_argument(
        "--helper",
        type=Path,
        default=REPO_ROOT / ".tools" / "osg_bundle",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="replace an existing output root before extracting packages",
    )
    args = parser.parse_args()

    package_dir, expected_arch = package_set()
    output_root = args.output_root.resolve()
    helper = args.helper.resolve()
    if args.clean and output_root.exists():
        safe_parents = ((REPO_ROOT / ".tools").resolve(), Path("/tmp"))
        if not any(
            output_root != parent and parent in output_root.parents
            for parent in safe_parents
        ):
            raise RuntimeError("--clean is restricted to .tools/ or /tmp")
        shutil.rmtree(output_root)

    verify_checksums(package_dir)
    extract_packages(package_dir, expected_arch, output_root)
    osgb_roundtrip.compile_helper(output_root, helper)
    check_native_dependencies(helper, output_root)
    plugin = osgb_roundtrip.osg_plugin_dir(
        osgb_roundtrip.osg_library_dir(output_root)
    )
    print("OSG root:", output_root)
    print("OSG plugin:", plugin)
    print("OSGB helper:", helper)
    print("Local OpenSceneGraph setup completed.")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import osgb_roundtrip

    main()
