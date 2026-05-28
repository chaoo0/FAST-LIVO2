#!/usr/bin/env python3
"""
Export a runtime-ready raw-input ONNX model with normalization embedded inside.

This script is intentionally limited to offline ONNX export and verification.
It does not modify FAST-LIVO2 runtime code, PoseCompensator inference logic,
ONNX backend behavior inside FAST-LIVO2, the safety layer, launch files, or
YAML configuration.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def require_dependency(module_name: str, install_hint: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency '{module_name}'. Install it first, for example:\n"
            f"  {install_hint}"
        ) from exc


np = require_dependency("numpy", "python3 -m pip install numpy")
torch = require_dependency("torch", "python3 -m pip install torch")
onnx = require_dependency("onnx", "python3 -m pip install onnx")
onnxruntime = require_dependency(
    "onnxruntime", "python3 -m pip install onnxruntime"
)

from train_mamba_pose_static_zero import (
    EXPECTED_FEATURE_NAMES,
    EXPECTED_LABEL_NAMES,
    FlattenMLP,
    decode_string_array,
    ensure_input_exists,
    extract_scalar,
    format_scalar,
    load_npz,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "Log" / "models" / "static_zero_mamba_pose.pt"
DEFAULT_NORM = REPO_ROOT / "Log" / "mamba_pose_feature_norm_T10_interpolated.npz"
DEFAULT_NORMALIZED_ONNX = REPO_ROOT / "Log" / "models" / "static_zero_mamba_pose.onnx"
DEFAULT_RAW_SEQUENCES = REPO_ROOT / "Log" / "mamba_pose_sequences_T10_interpolated.npz"
DEFAULT_NORMALIZED_SEQUENCES = (
    REPO_ROOT / "Log" / "mamba_pose_sequences_T10_interpolated_normalized.npz"
)
DEFAULT_OUTPUT = REPO_ROOT / "Log" / "models" / "static_zero_mamba_pose_raw_input.onnx"
DEFAULT_REPORT = (
    REPO_ROOT / "Log" / "models" / "static_zero_mamba_pose_raw_input_onnx_report.txt"
)


class RawInputWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, mean: np.ndarray, std_safe: np.ndarray):
        super().__init__()
        self.model = model
        self.register_buffer(
            "mean", torch.from_numpy(mean.astype(np.float32, copy=False))
        )
        self.register_buffer(
            "std_safe", torch.from_numpy(std_safe.astype(np.float32, copy=False))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise RuntimeError(
                f"Expected raw ONNX input shape [T, 18], got tensor with ndim={x.ndim}"
            )
        x_normalized = (x - self.mean) / self.std_safe
        output = self.model(x_normalized.unsqueeze(0))
        return output.squeeze(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a runtime-ready raw-input ONNX model by embedding feature "
            "normalization ahead of the trained static-zero baseline MLP."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Checkpoint path. Default: {DEFAULT_CHECKPOINT}",
    )
    parser.add_argument(
        "--norm",
        type=Path,
        default=DEFAULT_NORM,
        help=f"Feature norm NPZ path. Default: {DEFAULT_NORM}",
    )
    parser.add_argument(
        "--normalized-onnx",
        type=Path,
        default=DEFAULT_NORMALIZED_ONNX,
        help=f"Normalized-input ONNX path. Default: {DEFAULT_NORMALIZED_ONNX}",
    )
    parser.add_argument(
        "--raw-sequences",
        type=Path,
        default=DEFAULT_RAW_SEQUENCES,
        help=f"Raw sequence NPZ path. Default: {DEFAULT_RAW_SEQUENCES}",
    )
    parser.add_argument(
        "--normalized-sequences",
        type=Path,
        default=DEFAULT_NORMALIZED_SEQUENCES,
        help=f"Normalized sequence NPZ path. Default: {DEFAULT_NORMALIZED_SEQUENCES}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Raw-input ONNX output path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"TXT report output path. Default: {DEFAULT_REPORT}",
    )
    parser.add_argument("--input-name", default="input", help="Default: input")
    parser.add_argument("--output-name", default="output", help="Default: output")
    parser.add_argument("--onnx-opset", type=int, default=17, help="Default: 17")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.onnx_opset <= 0:
        raise SystemExit("--onnx-opset must be positive.")


def validate_checkpoint(checkpoint: dict[str, object]) -> dict[str, object]:
    required_fields = [
        "model_state_dict",
        "model_config",
        "feature_names",
        "label_names",
        "norm_path",
        "seq_len",
        "feature_dim",
        "label_dim",
        "label_type",
    ]
    missing = [field for field in required_fields if field not in checkpoint]
    if missing:
        raise SystemExit(
            "Checkpoint is missing required fields:\n"
            + "\n".join(f"  - {field}" for field in missing)
        )

    feature_names = checkpoint["feature_names"]
    label_names = checkpoint["label_names"]
    if feature_names != EXPECTED_FEATURE_NAMES:
        raise SystemExit(
            "Checkpoint feature_names do not match expected basic18 order.\n"
            f"Expected: {', '.join(EXPECTED_FEATURE_NAMES)}\n"
            f"Got: {', '.join(feature_names)}"
        )
    if label_names != EXPECTED_LABEL_NAMES:
        raise SystemExit(
            "Checkpoint label_names do not match expected 6D correction order.\n"
            f"Expected: {', '.join(EXPECTED_LABEL_NAMES)}\n"
            f"Got: {', '.join(label_names)}"
        )
    if checkpoint["label_type"] != "static_zero":
        raise SystemExit(
            f"Checkpoint label_type must be static_zero. Got: {checkpoint['label_type']}"
        )

    model_config = checkpoint["model_config"]
    if model_config.get("model_type") != "flatten_mlp":
        raise SystemExit(
            f"Unsupported model_type in checkpoint: {model_config.get('model_type')}"
        )

    return {
        "feature_names": feature_names,
        "label_names": label_names,
        "model_config": model_config,
        "norm_path": checkpoint["norm_path"],
        "seq_len": int(checkpoint["seq_len"]),
        "feature_dim": int(checkpoint["feature_dim"]),
        "label_dim": int(checkpoint["label_dim"]),
    }


def validate_norm_npz(norm_npz) -> dict[str, np.ndarray]:
    required_fields = ["feature_names", "mean", "std_safe"]
    missing = [field for field in required_fields if field not in norm_npz]
    if missing:
        raise SystemExit(
            "Norm NPZ is missing required fields:\n"
            + "\n".join(f"  - {field}" for field in missing)
        )

    feature_names = decode_string_array(norm_npz["feature_names"])
    if feature_names != EXPECTED_FEATURE_NAMES:
        raise SystemExit(
            "Norm feature_names do not match expected basic18 order.\n"
            f"Expected: {', '.join(EXPECTED_FEATURE_NAMES)}\n"
            f"Got: {', '.join(feature_names)}"
        )

    mean = np.asarray(norm_npz["mean"], dtype=np.float32)
    std_safe = np.asarray(norm_npz["std_safe"], dtype=np.float32)
    if mean.shape != (18,) or std_safe.shape != (18,):
        raise SystemExit(
            f"Norm mean/std_safe must have shape (18,). Got mean={mean.shape}, std_safe={std_safe.shape}"
        )
    if np.isnan(mean).any() or np.isinf(mean).any():
        raise SystemExit("Norm mean contains NaN/Inf.")
    if np.isnan(std_safe).any() or np.isinf(std_safe).any():
        raise SystemExit("Norm std_safe contains NaN/Inf.")
    if np.any(std_safe <= 0):
        raise SystemExit("Norm std_safe must be strictly positive.")
    return {"mean": mean, "std_safe": std_safe}


def validate_sequence_pair(raw_npz, normalized_npz) -> dict[str, object]:
    required_raw = ["X", "timestamps", "source_indices", "feature_names", "seq_len"]
    required_normalized = [
        "X_normalized",
        "timestamps",
        "source_indices",
        "feature_names",
        "seq_len",
        "norm_path",
    ]
    missing_raw = [field for field in required_raw if field not in raw_npz]
    missing_norm = [field for field in required_normalized if field not in normalized_npz]
    if missing_raw:
        raise SystemExit(
            "Raw sequence NPZ is missing required fields:\n"
            + "\n".join(f"  - {field}" for field in missing_raw)
        )
    if missing_norm:
        raise SystemExit(
            "Normalized sequence NPZ is missing required fields:\n"
            + "\n".join(f"  - {field}" for field in missing_norm)
        )

    raw_x = np.asarray(raw_npz["X"], dtype=np.float32)
    normalized_x = np.asarray(normalized_npz["X_normalized"], dtype=np.float32)
    raw_feature_names = decode_string_array(raw_npz["feature_names"])
    normalized_feature_names = decode_string_array(normalized_npz["feature_names"])

    if raw_feature_names != EXPECTED_FEATURE_NAMES:
        raise SystemExit("Raw sequence feature_names do not match expected basic18 order.")
    if normalized_feature_names != EXPECTED_FEATURE_NAMES:
        raise SystemExit(
            "Normalized sequence feature_names do not match expected basic18 order."
        )
    if raw_x.shape != normalized_x.shape:
        raise SystemExit(
            f"Raw and normalized sequence X shapes do not match: {raw_x.shape} vs {normalized_x.shape}"
        )
    if raw_x.ndim != 3 or raw_x.shape[2] != 18:
        raise SystemExit(f"Raw X must have shape [N, T, 18]. Got {raw_x.shape}.")
    if np.isnan(raw_x).any() or np.isinf(raw_x).any():
        raise SystemExit("Raw X contains NaN/Inf.")
    if np.isnan(normalized_x).any() or np.isinf(normalized_x).any():
        raise SystemExit("Normalized X contains NaN/Inf.")

    raw_timestamps = np.asarray(raw_npz["timestamps"])
    normalized_timestamps = np.asarray(normalized_npz["timestamps"])
    raw_source_indices = np.asarray(raw_npz["source_indices"])
    normalized_source_indices = np.asarray(normalized_npz["source_indices"])
    if not np.allclose(raw_timestamps, normalized_timestamps):
        raise SystemExit("Raw and normalized timestamps do not align.")
    if not np.array_equal(raw_source_indices, normalized_source_indices):
        raise SystemExit("Raw and normalized source_indices do not align.")
    return {
        "raw_x": raw_x,
        "normalized_x": normalized_x,
        "seq_len": int(extract_scalar(raw_npz["seq_len"], "seq_len")),
    }


def build_model_from_checkpoint(checkpoint: dict[str, object]) -> torch.nn.Module:
    model_config = checkpoint["model_config"]
    model = FlattenMLP(
        seq_len=int(model_config["seq_len"]),
        feature_dim=int(model_config["feature_dim"]),
        label_dim=int(model_config["label_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        num_layers=int(model_config["num_layers"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def export_raw_input_onnx(
    wrapper: RawInputWrapper,
    output_path: Path,
    seq_len: int,
    feature_dim: int,
    input_name: str,
    output_name: str,
    onnx_opset: int,
) -> None:
    example_input = torch.zeros((seq_len, feature_dim), dtype=torch.float32)
    torch.onnx.export(
        wrapper.cpu().eval(),
        example_input,
        output_path,
        export_params=True,
        opset_version=onnx_opset,
        do_constant_folding=True,
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes=None,
    )


def run_onnx_verification(
    args: argparse.Namespace,
    raw_input_onnx_path: Path,
    normalized_input_onnx_path: Path,
    raw_x: np.ndarray,
    normalized_x: np.ndarray,
) -> dict[str, object]:
    checker_status = "passed"
    smoke_status = "passed"
    smoke_message = "OK"

    try:
        raw_model = onnx.load(str(raw_input_onnx_path))
        onnx.checker.check_model(raw_model)
    except Exception as exc:
        checker_status = f"failed: {type(exc).__name__}: {exc}"
        smoke_status = "not_run"
        smoke_message = "Checker failed before runtime verification."
        return {
            "onnx_checker_status": checker_status,
            "onnxruntime_smoke_test_status": smoke_status,
            "onnxruntime_smoke_test_message": smoke_message,
        }

    try:
        raw_session = onnxruntime.InferenceSession(
            str(raw_input_onnx_path), providers=["CPUExecutionProvider"]
        )
        normalized_session = onnxruntime.InferenceSession(
            str(normalized_input_onnx_path), providers=["CPUExecutionProvider"]
        )

        sample_index = 0
        raw_sample = raw_x[sample_index].astype(np.float32, copy=False)
        normalized_sample = normalized_x[sample_index].astype(np.float32, copy=False)

        raw_output = np.asarray(
            raw_session.run([args.output_name], {args.input_name: raw_sample})[0],
            dtype=np.float32,
        )
        normalized_output = np.asarray(
            normalized_session.run([args.output_name], {args.input_name: normalized_sample})[0],
            dtype=np.float32,
        )

        if list(raw_output.shape) != [6]:
            raise RuntimeError(f"Unexpected raw-input ONNX output shape: {list(raw_output.shape)}")
        if list(normalized_output.shape) != [6]:
            raise RuntimeError(
                f"Unexpected normalized-input ONNX output shape: {list(normalized_output.shape)}"
            )

        diff = raw_output - normalized_output
        return {
            "onnx_checker_status": checker_status,
            "onnxruntime_smoke_test_status": smoke_status,
            "onnxruntime_smoke_test_message": smoke_message,
            "raw_onnx_output_abs_max": float(np.max(np.abs(raw_output))),
            "raw_onnx_output_abs_mean": float(np.mean(np.abs(raw_output))),
            "normalized_onnx_output_abs_max": float(np.max(np.abs(normalized_output))),
            "normalized_onnx_output_abs_mean": float(np.mean(np.abs(normalized_output))),
            "output_diff_abs_max_between_raw_and_normalized_onnx": float(np.max(np.abs(diff))),
            "output_diff_abs_mean_between_raw_and_normalized_onnx": float(np.mean(np.abs(diff))),
        }
    except Exception as exc:
        return {
            "onnx_checker_status": checker_status,
            "onnxruntime_smoke_test_status": "failed",
            "onnxruntime_smoke_test_message": f"{type(exc).__name__}: {exc}",
        }


def build_report_lines(
    args: argparse.Namespace,
    checkpoint_path: Path,
    norm_path: Path,
    normalized_onnx_path: Path,
    raw_sequences_path: Path,
    normalized_sequences_path: Path,
    output_path: Path,
    feature_names: list[str],
    label_names: list[str],
    embedded_status: str,
    verify_result: dict[str, object],
) -> list[str]:
    lines = [
        "Static Zero Raw-Input ONNX Export Report",
        "=" * 40,
        "",
        "Paths",
        f"- checkpoint_path: {checkpoint_path}",
        f"- norm_path: {norm_path}",
        f"- normalized_input_onnx_path: {normalized_onnx_path}",
        f"- raw_input_onnx_output_path: {output_path}",
        f"- raw_sequence_path: {raw_sequences_path}",
        f"- normalized_sequence_path: {normalized_sequences_path}",
        "",
        "ONNX Interface",
        "- input_name: input",
        "- output_name: output",
        "- input_shape: [10, 18]",
        "- output_shape: [6]",
        f"- feature_names: {', '.join(feature_names)}",
        f"- label_names: {', '.join(label_names)}",
        f"- mean_std_safe_embedded: {embedded_status}",
        f"- onnx_checker_status: {verify_result.get('onnx_checker_status', 'unknown')}",
        (
            f"- onnxruntime_smoke_test_status: "
            f"{verify_result.get('onnxruntime_smoke_test_status', 'unknown')}"
        ),
        (
            f"- onnxruntime_smoke_test_message: "
            f"{verify_result.get('onnxruntime_smoke_test_message', 'none')}"
        ),
    ]

    if "raw_onnx_output_abs_max" in verify_result:
        lines.extend(
            [
                "",
                "Output Comparison",
                f"- raw_onnx_output_abs_max: {format_scalar(verify_result['raw_onnx_output_abs_max'])}",
                f"- raw_onnx_output_abs_mean: {format_scalar(verify_result['raw_onnx_output_abs_mean'])}",
                (
                    f"- normalized_onnx_output_abs_max: "
                    f"{format_scalar(verify_result['normalized_onnx_output_abs_max'])}"
                ),
                (
                    f"- normalized_onnx_output_abs_mean: "
                    f"{format_scalar(verify_result['normalized_onnx_output_abs_mean'])}"
                ),
                (
                    "- output_diff_abs_max_between_raw_and_normalized_onnx: "
                    f"{format_scalar(verify_result['output_diff_abs_max_between_raw_and_normalized_onnx'])}"
                ),
                (
                    "- output_diff_abs_mean_between_raw_and_normalized_onnx: "
                    f"{format_scalar(verify_result['output_diff_abs_mean_between_raw_and_normalized_onnx'])}"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "Notes",
            "- static_zero_mamba_pose_raw_input.onnx is the recommended model to connect back into the current FAST-LIVO2 C++ backend, because FAST-LIVO2 sends raw basic18 features at runtime.",
            "- static_zero_mamba_pose.onnx is the normalized-input version and should not be connected directly to the current C++ backend.",
            "- The current model is still only a static-zero no-op baseline. It validates the export and runtime interface loop, not dynamic pose correction ability.",
        ]
    )
    return lines


def main() -> None:
    args = parse_args()
    validate_args(args)
    ensure_input_exists(args.checkpoint, "Checkpoint")
    ensure_input_exists(args.norm, "Norm NPZ")
    ensure_input_exists(args.normalized_onnx, "Normalized-input ONNX")
    ensure_input_exists(args.raw_sequences, "Raw sequence NPZ")
    ensure_input_exists(args.normalized_sequences, "Normalized sequence NPZ")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_info = validate_checkpoint(checkpoint)

    norm_npz = load_npz(args.norm)
    norm_info = validate_norm_npz(norm_npz)

    raw_npz = load_npz(args.raw_sequences)
    normalized_npz = load_npz(args.normalized_sequences)
    sequence_info = validate_sequence_pair(raw_npz, normalized_npz)

    if sequence_info["seq_len"] != checkpoint_info["seq_len"]:
        raise SystemExit(
            f"Sequence seq_len {sequence_info['seq_len']} does not match checkpoint seq_len {checkpoint_info['seq_len']}."
        )

    model = build_model_from_checkpoint(checkpoint)
    wrapper = RawInputWrapper(
        model=model,
        mean=norm_info["mean"],
        std_safe=norm_info["std_safe"],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    export_raw_input_onnx(
        wrapper=wrapper,
        output_path=args.output,
        seq_len=checkpoint_info["seq_len"],
        feature_dim=checkpoint_info["feature_dim"],
        input_name=args.input_name,
        output_name=args.output_name,
        onnx_opset=args.onnx_opset,
    )

    verify_result = run_onnx_verification(
        args=args,
        raw_input_onnx_path=args.output,
        normalized_input_onnx_path=args.normalized_onnx,
        raw_x=sequence_info["raw_x"],
        normalized_x=sequence_info["normalized_x"],
    )
    report_lines = build_report_lines(
        args=args,
        checkpoint_path=args.checkpoint,
        norm_path=args.norm,
        normalized_onnx_path=args.normalized_onnx,
        raw_sequences_path=args.raw_sequences,
        normalized_sequences_path=args.normalized_sequences,
        output_path=args.output,
        feature_names=checkpoint_info["feature_names"],
        label_names=checkpoint_info["label_names"],
        embedded_status="true",
        verify_result=verify_result,
    )
    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
