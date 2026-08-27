#!/usr/bin/env python3
"""Audit an M3DGR Mocap TUM file before FAST-LIVO2 alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from align_m3dgr_mocap_gt import load_tum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--max-quaternion-norm-error", type=float, default=1.0e-6)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    path = args.input.expanduser().resolve()
    if args.max_quaternion_norm_error <= 0:
        raise SystemExit("--max-quaternion-norm-error must be positive.")
    _, _, _, stats = load_tum(path)
    actual_hash = sha256_file(path)
    checksum_passed = (
        args.expected_sha256 is None
        or actual_hash == args.expected_sha256.strip().lower()
    )
    quaternion_passed = (
        stats["quaternion_norm_error_max"] < args.max_quaternion_norm_error
    )
    passed = bool(
        checksum_passed
        and stats["non_monotonic_count"] == 0
        and quaternion_passed
    )
    payload = {
        "schema_version": 1,
        "input": str(path),
        "sha256": actual_hash,
        "expected_sha256": args.expected_sha256,
        "checksum_passed": checksum_passed,
        **stats,
        "quaternion_norm_error_limit_strictly_lt": args.max_quaternion_norm_error,
        "passed": passed,
    }
    output = args.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("Mocap audit failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
