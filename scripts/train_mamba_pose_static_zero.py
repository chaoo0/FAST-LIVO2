#!/usr/bin/env python3
"""
Train a static zero-label no-op baseline model from the offline normalized dataset.

This script is intentionally limited to offline training and ONNX export. It
does not modify FAST-LIVO2 runtime code, PoseCompensator inference logic, ONNX
backend behavior inside FAST-LIVO2, the safety layer, launch files, or YAML
configuration.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
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
torch = require_dependency(
    "torch",
    "python3 -m pip install torch",
)

from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "Log" / "mamba_pose_static_zero_label_dataset_T10.npz"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Log" / "models"

EXPECTED_FEATURE_NAMES = [
    "pos_x",
    "pos_y",
    "pos_z",
    "rot_x",
    "rot_y",
    "rot_z",
    "rot_w",
    "vel_x",
    "vel_y",
    "vel_z",
    "bias_g_x",
    "bias_g_y",
    "bias_g_z",
    "bias_a_x",
    "bias_a_y",
    "bias_a_z",
    "effective_feature_num",
    "avg_residual",
]

EXPECTED_LABEL_NAMES = [
    "d_roll",
    "d_pitch",
    "d_yaw",
    "d_tx",
    "d_ty",
    "d_tz",
]


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value: {value}. Use true/false."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a static zero-label no-op baseline and optionally export a "
            "FAST-LIVO2-compatible ONNX model."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Input dataset NPZ path. Default: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output model directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Default: 30")
    parser.add_argument("--batch-size", type=int, default=128, help="Default: 128")
    parser.add_argument("--lr", type=float, default=1e-3, help="Default: 1e-3")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Default: 1e-5",
    )
    parser.add_argument("--hidden-dim", type=int, default=64, help="Default: 64")
    parser.add_argument("--num-layers", type=int, default=2, help="Default: 2")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Default: 0.2")
    parser.add_argument("--seed", type=int, default=42, help="Default: 42")
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device: auto/cpu/cuda. Default: auto",
    )
    parser.add_argument(
        "--export-onnx",
        type=parse_bool,
        default=True,
        help="Whether to export ONNX. Default: true",
    )
    parser.add_argument("--onnx-opset", type=int, default=17, help="Default: 17")
    parser.add_argument("--input-name", default="input", help="Default: input")
    parser.add_argument("--output-name", default="output", help="Default: output")
    return parser.parse_args()


def ensure_input_exists(input_path: Path, description: str) -> None:
    if not input_path.exists():
        raise SystemExit(f"{description} not found: {input_path}")
    if not input_path.is_file():
        raise SystemExit(f"{description} is not a file: {input_path}")


def format_scalar(value) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return value
    if value is None:
        return "none"
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.6f}"
    return str(value)


def extract_scalar(array_like, field_name: str):
    array_value = np.asarray(array_like)
    if array_value.size != 1:
        raise SystemExit(
            f"Field '{field_name}' must be a scalar in the NPZ. "
            f"Got shape {array_value.shape}."
        )
    return array_value.reshape(()).item()


def decode_string_array(array_like) -> list[str]:
    values = np.asarray(array_like).tolist()
    result: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            result.append(value.decode("utf-8"))
        else:
            result.append(str(value))
    return result


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.lr <= 0:
        raise SystemExit("--lr must be positive.")
    if args.weight_decay < 0:
        raise SystemExit("--weight-decay must be non-negative.")
    if args.hidden_dim <= 0:
        raise SystemExit("--hidden-dim must be positive.")
    if args.num_layers <= 0:
        raise SystemExit("--num-layers must be positive.")
    if not (0.0 < args.val_ratio < 1.0):
        raise SystemExit("--val-ratio must satisfy 0 < val_ratio < 1.")
    if args.onnx_opset <= 0:
        raise SystemExit("--onnx-opset must be positive.")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    normalized = device_arg.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized in {"cpu", "cuda"}:
        if normalized == "cuda" and not torch.cuda.is_available():
            raise SystemExit("Requested --device cuda but CUDA is not available.")
        return torch.device(normalized)
    raise SystemExit("Unsupported --device value. Use auto/cpu/cuda.")


def load_npz(input_path: Path):
    try:
        return np.load(input_path, allow_pickle=True)
    except Exception as exc:
        raise SystemExit(f"Failed to load NPZ '{input_path}': {exc}") from exc


class StaticZeroDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32, copy=False))
        self.y = torch.from_numpy(y.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


class FlattenMLP(nn.Module):
    def __init__(
        self,
        seq_len: int,
        feature_dim: int,
        label_dim: int,
        hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        input_dim = seq_len * feature_dim
        layers: list[nn.Module] = [nn.Flatten(start_dim=1)]
        prev_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, label_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise RuntimeError(
                f"Expected ONNX export input shape [T, 18], got tensor with ndim={x.ndim}"
            )
        output = self.model(x.unsqueeze(0))
        return output.squeeze(0)


def validate_dataset_npz(npz_data) -> dict[str, object]:
    required_fields = [
        "X_normalized",
        "y",
        "feature_names",
        "label_names",
        "seq_len",
        "label_type",
        "norm_path",
    ]
    missing_fields = [field for field in required_fields if field not in npz_data]
    if missing_fields:
        raise SystemExit(
            "Dataset NPZ is missing required fields:\n"
            + "\n".join(f"  - {field}" for field in missing_fields)
        )

    x = np.asarray(npz_data["X_normalized"], dtype=np.float32)
    y = np.asarray(npz_data["y"], dtype=np.float32)
    timestamps = np.asarray(npz_data["timestamps"])
    source_indices = np.asarray(npz_data["source_indices"])
    feature_names = decode_string_array(npz_data["feature_names"])
    label_names = decode_string_array(npz_data["label_names"])
    seq_len = int(extract_scalar(npz_data["seq_len"], "seq_len"))
    stride = int(extract_scalar(npz_data["stride"], "stride"))
    max_time_gap = float(extract_scalar(npz_data["max_time_gap"], "max_time_gap"))
    label_type = str(extract_scalar(npz_data["label_type"], "label_type"))
    norm_path = str(extract_scalar(npz_data["norm_path"], "norm_path"))

    if x.ndim != 3:
        raise SystemExit(f"X_normalized must be 3D. Got shape {x.shape}.")
    if y.ndim != 2:
        raise SystemExit(f"y must be 2D. Got shape {y.shape}.")
    if x.shape[0] != y.shape[0]:
        raise SystemExit(
            "X_normalized and y must have the same sample count. "
            f"Got {x.shape[0]} vs {y.shape[0]}."
        )
    if x.shape[2] != 18:
        raise SystemExit(f"X_normalized feature_dim must be 18. Got {x.shape[2]}.")
    if y.shape[1] != 6:
        raise SystemExit(f"y label_dim must be 6. Got {y.shape[1]}.")
    if x.shape[1] != seq_len:
        raise SystemExit(
            f"X_normalized T dimension {x.shape[1]} does not match seq_len={seq_len}."
        )
    if timestamps.shape != x.shape[:2]:
        raise SystemExit(
            f"timestamps shape must match X_normalized [N, T]. Got {timestamps.shape} vs {x.shape[:2]}."
        )
    if source_indices.shape != x.shape[:2]:
        raise SystemExit(
            f"source_indices shape must match X_normalized [N, T]. Got {source_indices.shape} vs {x.shape[:2]}."
        )
    if feature_names != EXPECTED_FEATURE_NAMES:
        raise SystemExit(
            "feature_names do not match required basic18 order.\n"
            f"Expected: {', '.join(EXPECTED_FEATURE_NAMES)}\n"
            f"Got: {', '.join(feature_names)}"
        )
    if label_names != EXPECTED_LABEL_NAMES:
        raise SystemExit(
            "label_names do not match required 6D correction order.\n"
            f"Expected: {', '.join(EXPECTED_LABEL_NAMES)}\n"
            f"Got: {', '.join(label_names)}"
        )
    if label_type != "static_zero":
        raise SystemExit(
            f"label_type must be static_zero for this training script. Got: {label_type}"
        )

    x_nan_count = int(np.isnan(x).sum())
    x_inf_count = int(np.isinf(x).sum())
    y_nan_count = int(np.isnan(y).sum())
    y_inf_count = int(np.isinf(y).sum())
    if x_nan_count > 0 or x_inf_count > 0:
        raise SystemExit(
            f"X_normalized contains NaN/Inf. NaN={x_nan_count}, Inf={x_inf_count}"
        )
    if y_nan_count > 0 or y_inf_count > 0:
        raise SystemExit(f"y contains NaN/Inf. NaN={y_nan_count}, Inf={y_inf_count}")
    if not np.all(y == 0.0):
        raise SystemExit("This dataset is not all-zero labels, so it is not static_zero.")

    return {
        "x": x,
        "y": y,
        "timestamps": timestamps,
        "source_indices": source_indices,
        "feature_names": feature_names,
        "label_names": label_names,
        "seq_len": seq_len,
        "stride": stride,
        "max_time_gap": max_time_gap,
        "label_type": label_type,
        "norm_path": norm_path,
        "x_nan_count": x_nan_count,
        "x_inf_count": x_inf_count,
        "y_nan_count": y_nan_count,
        "y_inf_count": y_inf_count,
    }


def read_norm_path(norm_path: Path) -> None:
    ensure_input_exists(norm_path, "Norm NPZ")
    norm_npz = load_npz(norm_path)
    required_fields = ["feature_names", "mean", "std_safe"]
    missing = [field for field in required_fields if field not in norm_npz]
    if missing:
        raise SystemExit(
            f"Norm NPZ is missing required fields:\n" + "\n".join(f"  - {field}" for field in missing)
        )


def split_dataset(
    dataset: Dataset,
    val_ratio: float,
    seed: int,
) -> tuple[Dataset, Dataset, int, int]:
    total_size = len(dataset)
    val_size = max(1, int(round(total_size * val_ratio)))
    train_size = total_size - val_size
    if train_size <= 0:
        raise SystemExit(
            f"Validation ratio {val_ratio} leaves no training samples. "
            f"total_size={total_size}, val_size={val_size}"
        )
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )
    return train_dataset, val_dataset, train_size, val_size


def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = int(batch_x.shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def write_train_log_csv(log_rows: list[dict[str, float]], csv_path: Path) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "val_loss", "learning_rate"],
        )
        writer.writeheader()
        writer.writerows(log_rows)


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    model_config: dict[str, object],
    feature_names: list[str],
    label_names: list[str],
    norm_path: str,
    seq_len: int,
    feature_dim: int,
    label_dim: int,
    best_val_loss: float,
    label_type: str,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "feature_names": feature_names,
            "label_names": label_names,
            "norm_path": norm_path,
            "seq_len": seq_len,
            "feature_dim": feature_dim,
            "label_dim": label_dim,
            "best_val_loss": best_val_loss,
            "label_type": label_type,
        },
        checkpoint_path,
    )


def run_sanity_inference(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    sample_count: int = 5,
) -> dict[str, object]:
    model.eval()
    sample_count = min(sample_count, x.shape[0])
    sample_x = torch.from_numpy(x[:sample_count]).to(device)
    with torch.no_grad():
        prediction = model(sample_x).detach().cpu().numpy().astype(np.float32, copy=False)
    return {
        "prediction_samples": prediction,
        "pred_abs_max": float(np.max(np.abs(prediction))),
        "pred_abs_mean": float(np.mean(np.abs(prediction))),
    }


def try_export_onnx(
    args: argparse.Namespace,
    model: nn.Module,
    seq_len: int,
    feature_dim: int,
    onnx_path: Path,
) -> dict[str, object]:
    if not args.export_onnx:
        return {
            "onnx_export_status": "disabled",
            "onnx_smoke_test_status": "not_run",
            "onnx_smoke_test_message": "ONNX export disabled by --export-onnx false.",
        }

    export_model = ExportWrapper(model.cpu()).eval()
    example_input = torch.zeros((seq_len, feature_dim), dtype=torch.float32)
    try:
        torch.onnx.export(
            export_model,
            example_input,
            onnx_path,
            export_params=True,
            opset_version=args.onnx_opset,
            do_constant_folding=True,
            input_names=[args.input_name],
            output_names=[args.output_name],
            dynamic_axes=None,
        )
    except Exception as exc:
        return {
            "onnx_export_status": f"failed: {type(exc).__name__}: {exc}",
            "onnx_smoke_test_status": "not_run",
            "onnx_smoke_test_message": "ONNX export failed before smoke test.",
        }

    smoke_status = "skipped"
    smoke_message = "onnx or onnxruntime not available."
    try:
        onnx = __import__("onnx")
        onnxruntime = __import__("onnxruntime")
    except Exception:
        return {
            "onnx_export_status": "exported",
            "onnx_smoke_test_status": smoke_status,
            "onnx_smoke_test_message": smoke_message,
        }

    try:
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        zero_input = np.zeros((seq_len, feature_dim), dtype=np.float32)
        outputs = session.run([args.output_name], {args.input_name: zero_input})
        output = np.asarray(outputs[0], dtype=np.float32)
        if list(output.shape) != [len(EXPECTED_LABEL_NAMES)]:
            smoke_status = "failed"
            smoke_message = f"Unexpected ONNX output shape: {list(output.shape)}"
        else:
            abs_max = float(np.max(np.abs(output)))
            abs_mean = float(np.mean(np.abs(output)))
            smoke_status = "passed"
            smoke_message = (
                f"shape=[6], zero_input_output_abs_max={abs_max:.6f}, "
                f"zero_input_output_abs_mean={abs_mean:.6f}"
            )
    except Exception as exc:
        smoke_status = "failed"
        smoke_message = f"{type(exc).__name__}: {exc}"

    return {
        "onnx_export_status": "exported",
        "onnx_smoke_test_status": smoke_status,
        "onnx_smoke_test_message": smoke_message,
    }


def build_report_lines(
    args: argparse.Namespace,
    dataset_path: Path,
    checkpoint_path: Path,
    onnx_path: Path,
    log_csv_path: Path,
    report_path: Path,
    dataset_info: dict[str, object],
    train_size: int,
    val_size: int,
    model_config: dict[str, object],
    log_rows: list[dict[str, float]],
    best_train_loss: float,
    best_val_loss: float,
    final_train_loss: float,
    final_val_loss: float,
    sanity_result: dict[str, object],
    onnx_result: dict[str, object],
) -> list[str]:
    sample_predictions = sanity_result["prediction_samples"]
    lines = [
        "Mamba Pose Static Zero Baseline Training Report",
        "=" * 47,
        "",
        "Paths",
        f"- dataset_path: {dataset_path}",
        f"- output_checkpoint_path: {checkpoint_path}",
        f"- output_onnx_path: {onnx_path}",
        f"- train_log_csv_path: {log_csv_path}",
        f"- output_report_path: {report_path}",
        "",
        "Dataset Summary",
        f"- X shape: {list(dataset_info['x'].shape)}",
        f"- y shape: {list(dataset_info['y'].shape)}",
        f"- train_size: {train_size}",
        f"- val_size: {val_size}",
        f"- feature_names: {', '.join(dataset_info['feature_names'])}",
        f"- label_names: {', '.join(dataset_info['label_names'])}",
        f"- label_type: {dataset_info['label_type']}",
        "",
        "Model",
        "- model_type: flatten_mlp",
        f"- model_config: {model_config}",
        f"- epochs: {args.epochs}",
        f"- batch_size: {args.batch_size}",
        f"- lr: {format_scalar(args.lr)}",
        f"- weight_decay: {format_scalar(args.weight_decay)}",
        f"- device: {args.device}",
        "",
        "Training Result",
        f"- best_train_loss: {format_scalar(best_train_loss)}",
        f"- best_val_loss: {format_scalar(best_val_loss)}",
        f"- final_train_loss: {format_scalar(final_train_loss)}",
        f"- final_val_loss: {format_scalar(final_val_loss)}",
        f"- pred_abs_max: {format_scalar(sanity_result['pred_abs_max'])}",
        f"- pred_abs_mean: {format_scalar(sanity_result['pred_abs_mean'])}",
        "",
        "Sanity Prediction Samples",
    ]

    for index, prediction in enumerate(sample_predictions.tolist()):
        lines.append(f"- sample_{index}: {prediction}")

    lines.extend(
        [
            "",
            "ONNX",
            f"- onnx_export_status: {onnx_result['onnx_export_status']}",
            f"- onnx_smoke_test_status: {onnx_result['onnx_smoke_test_status']}",
            f"- onnx_smoke_test_message: {onnx_result['onnx_smoke_test_message']}",
            f"- onnx_input_name: {args.input_name}",
            f"- onnx_output_name: {args.output_name}",
            f"- onnx_input_shape: [{dataset_info['seq_len']}, {dataset_info['x'].shape[2]}]",
            f"- onnx_output_shape: [{dataset_info['y'].shape[1]}]",
            "",
            "Warnings",
            "- Current model is a static-zero no-op baseline.",
            "- It is only meant to validate training, checkpointing, ONNX export, and later deployment loop wiring.",
            "- It cannot prove real dynamic pose correction ability because the current target y is always zero on a stationary rosbag.",
        ]
    )
    return lines


def main() -> None:
    args = parse_args()
    validate_args(args)
    ensure_input_exists(args.dataset, "Dataset NPZ")
    set_global_seed(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "static_zero_mamba_pose.pt"
    onnx_path = output_dir / "static_zero_mamba_pose.onnx"
    log_csv_path = output_dir / "static_zero_mamba_pose_train_log.csv"
    report_path = output_dir / "static_zero_mamba_pose_train_report.txt"

    dataset_npz = load_npz(args.dataset)
    dataset_info = validate_dataset_npz(dataset_npz)
    read_norm_path(Path(dataset_info["norm_path"]))
    device = resolve_device(args.device)

    dataset = StaticZeroDataset(dataset_info["x"], dataset_info["y"])
    train_dataset, val_dataset, train_size, val_size = split_dataset(
        dataset, args.val_ratio, args.seed
    )
    train_loader = create_dataloader(train_dataset, args.batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, args.batch_size, shuffle=False)

    model = FlattenMLP(
        seq_len=dataset_info["seq_len"],
        feature_dim=dataset_info["x"].shape[2],
        label_dim=dataset_info["y"].shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    model_config = {
        "model_type": "flatten_mlp",
        "seq_len": int(dataset_info["seq_len"]),
        "feature_dim": int(dataset_info["x"].shape[2]),
        "label_dim": int(dataset_info["y"].shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
    }

    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_state_dict = None
    log_rows: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device, optimizer=None)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": learning_rate,
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state_dict is None:
        raise SystemExit("Training did not produce a best checkpoint state.")

    model.load_state_dict(best_state_dict)
    final_train_loss = float(log_rows[-1]["train_loss"])
    final_val_loss = float(log_rows[-1]["val_loss"])

    write_train_log_csv(log_rows, log_csv_path)
    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model.cpu(),
        model_config=model_config,
        feature_names=dataset_info["feature_names"],
        label_names=dataset_info["label_names"],
        norm_path=dataset_info["norm_path"],
        seq_len=dataset_info["seq_len"],
        feature_dim=dataset_info["x"].shape[2],
        label_dim=dataset_info["y"].shape[1],
        best_val_loss=best_val_loss,
        label_type=dataset_info["label_type"],
    )

    model = model.to(device)
    sanity_result = run_sanity_inference(model, dataset_info["x"], device=device)
    onnx_result = try_export_onnx(
        args=args,
        model=model,
        seq_len=dataset_info["seq_len"],
        feature_dim=dataset_info["x"].shape[2],
        onnx_path=onnx_path,
    )

    report_lines = build_report_lines(
        args=args,
        dataset_path=args.dataset,
        checkpoint_path=checkpoint_path,
        onnx_path=onnx_path,
        log_csv_path=log_csv_path,
        report_path=report_path,
        dataset_info=dataset_info,
        train_size=train_size,
        val_size=val_size,
        model_config=model_config,
        log_rows=log_rows,
        best_train_loss=best_train_loss,
        best_val_loss=best_val_loss,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        sanity_result=sanity_result,
        onnx_result=onnx_result,
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
