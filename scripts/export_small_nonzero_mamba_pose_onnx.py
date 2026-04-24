#!/usr/bin/env python3
"""
Export a minimal non-zero ONNX model for FAST-LIVO2 pose compensation safety tests.

Dependencies:
  - python3
  - torch
  - onnx

Model contract:
  - single input named "input" by default
  - single output named "output" by default
  - input shape: [T, 18], with T as a dynamic sequence length
  - output shape: [6]
  - output layout: [d_roll, d_pitch, d_yaw, d_tx, d_ty, d_tz]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def require_dependency(module_name: str, install_hint: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency '{module_name}'. Install it first, for example:\n"
            f"  {install_hint}"
        ) from exc


torch = require_dependency("torch", "python3 -m pip install torch")
onnx = require_dependency("onnx", "python3 -m pip install onnx")


DEFAULT_NORMAL = [0.01, 0.0, 0.0, 0.01, 0.0, 0.0]
DEFAULT_OVERSIZED = [0.30, 0.0, 0.0, 0.50, 0.0, 0.0]


class FixedCorrectionModel(torch.nn.Module):
    def __init__(self, correction):
        super().__init__()
        self.register_buffer(
            "correction",
            torch.tensor(correction, dtype=torch.float32),
            persistent=True,
        )

    def forward(self, x):
        # Keep the graph connected to the declared input tensor so the exported
        # model preserves the expected [T, 18] input contract.
        anchor = x.sum() * 0.0
        return anchor + self.correction


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_output = repo_root / "Log" / "models" / "small_nonzero_mamba_pose.onnx"

    parser = argparse.ArgumentParser(
        description="Export a minimal non-zero ONNX model for FAST-LIVO2 safety-layer tests."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Output ONNX path. Default: {default_output}",
    )
    parser.add_argument(
        "--input-name",
        default="input",
        help="ONNX input tensor name. Default: input",
    )
    parser.add_argument(
        "--output-name",
        default="output",
        help="ONNX output tensor name. Default: output",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=18,
        help="Feature dimension F for input shape [T, F]. Default: 18",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=8,
        help="Dummy sequence length used only during export. Default: 8",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version. Default: 17",
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "oversized"],
        default="normal",
        help="normal exports a small safe correction, oversized exports a deliberately large correction.",
    )
    parser.add_argument(
        "--correction",
        nargs=6,
        type=float,
        metavar=("D_ROLL", "D_PITCH", "D_YAW", "D_TX", "D_TY", "D_TZ"),
        help="Optional manual correction override with 6 values.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.feature_dim <= 0:
        raise SystemExit("--feature-dim must be positive.")
    if args.sequence_length <= 0:
        raise SystemExit("--sequence-length must be positive.")

    if args.correction is not None:
        correction = list(args.correction)
    elif args.mode == "oversized":
        correction = list(DEFAULT_OVERSIZED)
    else:
        correction = list(DEFAULT_NORMAL)

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = FixedCorrectionModel(correction).eval()
    dummy_input = torch.zeros(
        (args.sequence_length, args.feature_dim), dtype=torch.float32
    )

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=[args.input_name],
            output_names=[args.output_name],
            dynamic_axes={
                args.input_name: {0: "sequence_length"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
        )

    exported_model = onnx.load(str(output_path))
    onnx.checker.check_model(exported_model)

    print("Exported non-zero ONNX model successfully.")
    print(f"  path: {output_path}")
    print(f"  mode: {args.mode}")
    print(f"  input_name: {args.input_name}")
    print(f"  output_name: {args.output_name}")
    print(f"  input_shape: [T, {args.feature_dim}]")
    print("  output_shape: [6]")
    print(f"  fixed_correction: {correction}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
