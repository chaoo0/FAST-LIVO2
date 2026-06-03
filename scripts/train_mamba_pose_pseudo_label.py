#!/usr/bin/env python3
"""
Train a pseudo-label MambaPose baseline model and export runtime-ready ONNX.

This script is intentionally limited to offline training, checkpoint export,
normalized-input ONNX export, raw-input ONNX export, and verification. It does
not modify FAST-LIVO2 runtime code, PoseCompensator inference logic, ONNX
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
torch = require_dependency("torch", "python3 -m pip install torch")

from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "Log" / "mamba_pose_pseudo_label_dataset_T10_selected.npz"
DEFAULT_NORM = REPO_ROOT / "Log" / "mamba_pose_feature_norm_T10_interpolated.npz"
DEFAULT_RAW_SEQUENCES = REPO_ROOT / "Log" / "mamba_pose_sequences_T10_interpolated.npz"
DEFAULT_NORMALIZED_SEQUENCES = (
    REPO_ROOT / "Log" / "mamba_pose_sequences_T10_interpolated_normalized.npz"
)
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
            "Train a pseudo_smooth_reference temporal baseline and export both "
            "normalized-input and raw-input ONNX models."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Pseudo-label dataset NPZ path. Default: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--norm",
        type=Path,
        default=DEFAULT_NORM,
        help=f"Feature norm NPZ path. Default: {DEFAULT_NORM}",
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
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output model directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Default: 50")
    parser.add_argument("--batch-size", type=int, default=128, help="Default: 128")
    parser.add_argument("--lr", type=float, default=1e-3, help="Default: 1e-3")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Default: 1e-5",
    )
    parser.add_argument("--hidden-dim", type=int, default=128, help="Default: 128")
    parser.add_argument("--num-layers", type=int, default=3, help="Default: 3")
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
        help="Whether to export ONNX models. Default: true",
    )
    parser.add_argument("--onnx-opset", type=int, default=17, help="Default: 17")
    parser.add_argument("--input-name", default="input", help="Default: input")
    parser.add_argument("--output-name", default="output", help="Default: output")
    parser.add_argument(
        "--model-type",
        default="flatten_mlp",
        help="Only flatten_mlp is supported right now. Default: flatten_mlp",
    )
    return parser.parse_args()


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
    if args.model_type != "flatten_mlp":
        raise SystemExit(
            f"Unsupported --model-type: {args.model_type}. Only flatten_mlp is supported."
        )


def ensure_input_exists(input_path: Path, description: str) -> None:
    if not input_path.exists():
        raise SystemExit(f"{description} not found: {input_path}")
    if not input_path.is_file():
        raise SystemExit(f"{description} is not a file: {input_path}")


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


def extract_scalar(array_like, field_name: str):
    array_value = np.asarray(array_like)
    if array_value.size != 1:
        raise SystemExit(
            f"Field '{field_name}' must be scalar-like. Got shape {array_value.shape}."
        )
    return array_value.reshape(()).item()


def decode_string_array(array_like) -> list[str]:
    values = np.asarray(array_like).tolist()
    decoded: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


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


class SequenceRegressionDataset(Dataset):
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
                f"Expected ONNX input shape [T, 18], got tensor with ndim={x.ndim}"
            )
        return self.model(x.unsqueeze(0)).squeeze(0)


class RawInputWrapper(nn.Module):
    def __init__(self, model: nn.Module, mean: np.ndarray, std_safe: np.ndarray):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32, copy=False)))
        self.register_buffer(
            "std_safe", torch.from_numpy(std_safe.astype(np.float32, copy=False))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise RuntimeError(
                f"Expected raw ONNX input shape [T, 18], got tensor with ndim={x.ndim}"
            )
        x_normalized = (x - self.mean) / self.std_safe
        return self.model(x_normalized.unsqueeze(0)).squeeze(0)


def validate_dataset_npz(npz_data) -> dict[str, object]:
    required_fields = [
        "X_normalized",
        "y",
        "timestamps",
        "source_indices",
        "feature_names",
        "label_names",
        "label_type",
        "seq_len",
        "stride",
        "max_time_gap",
        "norm_path",
        "smooth_window_sec",
        "max_rot_label_rad",
        "max_trans_label_m",
    ]
    missing_fields = [field for field in required_fields if field not in npz_data]
    if missing_fields:
        raise SystemExit(
            "Dataset NPZ is missing required fields:\n"
            + "\n".join(f"  - {field}" for field in missing_fields)
        )

    x = np.asarray(npz_data["X_normalized"], dtype=np.float32)
    y = np.asarray(npz_data["y"], dtype=np.float32)
    timestamps = np.asarray(npz_data["timestamps"], dtype=np.float64)
    source_indices = np.asarray(npz_data["source_indices"], dtype=np.int64)
    feature_names = decode_string_array(npz_data["feature_names"])
    label_names = decode_string_array(npz_data["label_names"])
    label_type = str(extract_scalar(npz_data["label_type"], "label_type"))
    seq_len = int(extract_scalar(npz_data["seq_len"], "seq_len"))
    stride = int(extract_scalar(npz_data["stride"], "stride"))
    max_time_gap = float(extract_scalar(npz_data["max_time_gap"], "max_time_gap"))
    norm_path = str(extract_scalar(npz_data["norm_path"], "norm_path"))
    smooth_window_sec = float(
        extract_scalar(npz_data["smooth_window_sec"], "smooth_window_sec")
    )
    max_rot_label_rad = float(
        extract_scalar(npz_data["max_rot_label_rad"], "max_rot_label_rad")
    )
    max_trans_label_m = float(
        extract_scalar(npz_data["max_trans_label_m"], "max_trans_label_m")
    )

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
    if label_type != "pseudo_smooth_reference":
        raise SystemExit(
            "label_type must be pseudo_smooth_reference for this script. "
            f"Got: {label_type}"
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
    if np.all(y == 0.0):
        raise SystemExit(
            "Pseudo-label dataset y is all zero, which is invalid for this stage."
        )

    return {
        "x": x,
        "y": y,
        "timestamps": timestamps,
        "source_indices": source_indices,
        "feature_names": feature_names,
        "label_names": label_names,
        "label_type": label_type,
        "seq_len": seq_len,
        "stride": stride,
        "max_time_gap": max_time_gap,
        "norm_path": norm_path,
        "smooth_window_sec": smooth_window_sec,
        "max_rot_label_rad": max_rot_label_rad,
        "max_trans_label_m": max_trans_label_m,
        "x_nan_count": x_nan_count,
        "x_inf_count": x_inf_count,
        "y_nan_count": y_nan_count,
        "y_inf_count": y_inf_count,
    }


def validate_norm_npz(npz_data, expected_path: Path, feature_names: list[str]) -> dict[str, np.ndarray]:
    required_fields = ["feature_names", "mean", "std_safe"]
    missing_fields = [field for field in required_fields if field not in npz_data]
    if missing_fields:
        raise SystemExit(
            "Norm NPZ is missing required fields:\n"
            + "\n".join(f"  - {field}" for field in missing_fields)
        )
    loaded_feature_names = decode_string_array(npz_data["feature_names"])
    if loaded_feature_names != feature_names:
        raise SystemExit(
            "Norm feature_names do not match dataset feature_names.\n"
            f"Norm: {', '.join(loaded_feature_names)}\n"
            f"Dataset: {', '.join(feature_names)}"
        )
    mean = np.asarray(npz_data["mean"], dtype=np.float32)
    std_safe = np.asarray(npz_data["std_safe"], dtype=np.float32)
    if mean.shape != (18,) or std_safe.shape != (18,):
        raise SystemExit(
            f"Norm mean/std_safe must have shape (18,). Got mean={mean.shape}, std_safe={std_safe.shape}."
        )
    if np.isnan(mean).any() or np.isinf(mean).any():
        raise SystemExit("Norm mean contains NaN/Inf.")
    if np.isnan(std_safe).any() or np.isinf(std_safe).any():
        raise SystemExit("Norm std_safe contains NaN/Inf.")
    if np.any(std_safe <= 0):
        raise SystemExit("Norm std_safe must be strictly positive.")
    return {"mean": mean, "std_safe": std_safe}


def validate_full_sequence_npz(npz_data, x_field: str, feature_names: list[str]) -> dict[str, np.ndarray]:
    required_fields = [
        x_field,
        "timestamps",
        "source_indices",
        "feature_names",
        "seq_len",
        "stride",
        "max_time_gap",
    ]
    missing_fields = [field for field in required_fields if field not in npz_data]
    if missing_fields:
        raise SystemExit(
            f"Sequence NPZ missing required fields for '{x_field}':\n"
            + "\n".join(f"  - {field}" for field in missing_fields)
        )
    loaded_feature_names = decode_string_array(npz_data["feature_names"])
    if loaded_feature_names != feature_names:
        raise SystemExit(
            f"Sequence NPZ feature_names for '{x_field}' do not match expected basic18 order."
        )
    x = np.asarray(npz_data[x_field], dtype=np.float32)
    timestamps = np.asarray(npz_data["timestamps"], dtype=np.float64)
    source_indices = np.asarray(npz_data["source_indices"], dtype=np.int64)
    seq_len = int(extract_scalar(npz_data["seq_len"], "seq_len"))
    stride = int(extract_scalar(npz_data["stride"], "stride"))
    max_time_gap = float(extract_scalar(npz_data["max_time_gap"], "max_time_gap"))
    if x.ndim != 3 or x.shape[2] != 18:
        raise SystemExit(f"Sequence field '{x_field}' must have shape [N, T, 18]. Got {x.shape}.")
    if timestamps.shape != x.shape[:2]:
        raise SystemExit(
            f"timestamps shape for '{x_field}' must match [N, T]. Got {timestamps.shape} vs {x.shape[:2]}."
        )
    if source_indices.shape != x.shape[:2]:
        raise SystemExit(
            f"source_indices shape for '{x_field}' must match [N, T]. Got {source_indices.shape} vs {x.shape[:2]}."
        )
    if x.shape[1] != seq_len:
        raise SystemExit(f"Sequence field '{x_field}' T dimension does not match seq_len={seq_len}.")
    if np.isnan(x).any() or np.isinf(x).any():
        raise SystemExit(f"Sequence field '{x_field}' contains NaN/Inf.")
    return {
        "x": x,
        "timestamps": timestamps,
        "source_indices": source_indices,
        "seq_len": seq_len,
        "stride": stride,
        "max_time_gap": max_time_gap,
    }


def validate_sequence_alignment(
    raw_info: dict[str, np.ndarray], normalized_info: dict[str, np.ndarray]
) -> None:
    if raw_info["x"].shape != normalized_info["x"].shape:
        raise SystemExit(
            "Raw and normalized sequence X shapes do not match. "
            f"Got {raw_info['x'].shape} vs {normalized_info['x'].shape}."
        )
    if not np.allclose(raw_info["timestamps"], normalized_info["timestamps"]):
        raise SystemExit("Raw and normalized sequence timestamps do not align.")
    if not np.array_equal(raw_info["source_indices"], normalized_info["source_indices"]):
        raise SystemExit("Raw and normalized sequence source_indices do not align.")


def build_sequence_index_map(
    source_indices: np.ndarray,
    timestamps: np.ndarray,
) -> dict[tuple[int, ...], list[int]]:
    mapping: dict[tuple[int, ...], list[int]] = {}
    for idx in range(source_indices.shape[0]):
        key = tuple(source_indices[idx].tolist())
        mapping.setdefault(key, []).append(idx)
    return mapping


def find_matching_full_sequence_index(
    dataset_source_indices_row: np.ndarray,
    dataset_timestamps_row: np.ndarray,
    full_source_indices: np.ndarray,
    full_timestamps: np.ndarray,
    sequence_map: dict[tuple[int, ...], list[int]],
) -> int:
    key = tuple(dataset_source_indices_row.tolist())
    candidates = sequence_map.get(key, [])
    for idx in candidates:
        if np.allclose(full_timestamps[idx], dataset_timestamps_row, atol=1e-6, rtol=0.0):
            return idx
    raise SystemExit(
        "Failed to match a dataset sequence back to the full raw/normalized sequence NPZ "
        "using source_indices and timestamps."
    )


def split_dataset(
    dataset: Dataset,
    val_ratio: float,
    seed: int,
) -> tuple[Subset, Subset, int, int]:
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
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


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
    dataset_info: dict[str, object],
    dataset_path: Path,
    best_val_loss: float,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "feature_names": dataset_info["feature_names"],
            "label_names": dataset_info["label_names"],
            "norm_path": dataset_info["norm_path"],
            "seq_len": dataset_info["seq_len"],
            "feature_dim": dataset_info["x"].shape[2],
            "label_dim": dataset_info["y"].shape[1],
            "best_val_loss": best_val_loss,
            "label_type": dataset_info["label_type"],
            "dataset_path": str(dataset_path),
            "smooth_window_sec": dataset_info["smooth_window_sec"],
            "max_rot_label_rad": dataset_info["max_rot_label_rad"],
            "max_trans_label_m": dataset_info["max_trans_label_m"],
        },
        checkpoint_path,
    )


def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            prediction = model(batch_x).detach().cpu().numpy().astype(np.float32, copy=False)
            predictions.append(prediction)
            targets.append(batch_y.numpy().astype(np.float32, copy=False))
    if not predictions:
        return (
            np.empty((0, len(EXPECTED_LABEL_NAMES)), dtype=np.float32),
            np.empty((0, len(EXPECTED_LABEL_NAMES)), dtype=np.float32),
        )
    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
    )


def compute_abs_percentiles(y: np.ndarray) -> dict[str, np.ndarray]:
    abs_y = np.abs(y.astype(np.float64, copy=False))
    return {
        "abs_max": np.max(abs_y, axis=0),
        "abs_mean": np.mean(abs_y, axis=0),
        "abs_p90": np.percentile(abs_y, 90, axis=0),
        "abs_p95": np.percentile(abs_y, 95, axis=0),
        "abs_p99": np.percentile(abs_y, 99, axis=0),
    }


def build_first_five_samples(predictions: np.ndarray, targets: np.ndarray) -> list[dict[str, object]]:
    sample_count = min(5, predictions.shape[0], targets.shape[0])
    rows: list[dict[str, object]] = []
    for idx in range(sample_count):
        rows.append(
            {
                "index": idx,
                "pred": predictions[idx].tolist(),
                "target": targets[idx].tolist(),
            }
        )
    return rows


def export_normalized_input_onnx(
    model: nn.Module,
    onnx_path: Path,
    seq_len: int,
    feature_dim: int,
    input_name: str,
    output_name: str,
    onnx_opset: int,
) -> str:
    wrapper = ExportWrapper(model.cpu().eval())
    example_input = torch.zeros((seq_len, feature_dim), dtype=torch.float32)
    try:
        torch.onnx.export(
            wrapper,
            example_input,
            onnx_path,
            export_params=True,
            opset_version=onnx_opset,
            do_constant_folding=True,
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes=None,
        )
        return "exported"
    except Exception as exc:
        return f"failed: {type(exc).__name__}: {exc}"


def export_raw_input_onnx(
    model: nn.Module,
    mean: np.ndarray,
    std_safe: np.ndarray,
    onnx_path: Path,
    seq_len: int,
    feature_dim: int,
    input_name: str,
    output_name: str,
    onnx_opset: int,
) -> str:
    wrapper = RawInputWrapper(model.cpu().eval(), mean=mean, std_safe=std_safe)
    example_input = torch.zeros((seq_len, feature_dim), dtype=torch.float32)
    try:
        torch.onnx.export(
            wrapper,
            example_input,
            onnx_path,
            export_params=True,
            opset_version=onnx_opset,
            do_constant_folding=True,
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes=None,
        )
        return "exported"
    except Exception as exc:
        return f"failed: {type(exc).__name__}: {exc}"


def try_import_onnx_tools():
    try:
        onnx = __import__("onnx")
        onnxruntime = __import__("onnxruntime")
        return onnx, onnxruntime
    except Exception:
        return None, None


def verify_single_onnx(
    onnx_module,
    ort_module,
    onnx_path: Path,
    input_name: str,
    output_name: str,
    sample_input: np.ndarray,
) -> dict[str, object]:
    result = {
        "checker_status": "not_run",
        "onnxruntime_status": "not_run",
        "output_shape": None,
        "output_abs_max": None,
        "output_abs_mean": None,
        "output_array": None,
        "message": "not_run",
    }
    try:
        onnx_model = onnx_module.load(str(onnx_path))
        onnx_module.checker.check_model(onnx_model)
        result["checker_status"] = "passed"
    except Exception as exc:
        result["checker_status"] = f"failed: {type(exc).__name__}: {exc}"
        result["message"] = "checker_failed"
        return result

    try:
        session = ort_module.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        output = session.run(
            [output_name],
            {input_name: sample_input.astype(np.float32, copy=False)},
        )[0]
        output = np.asarray(output, dtype=np.float32)
        result["output_shape"] = list(output.shape)
        result["output_abs_max"] = float(np.max(np.abs(output)))
        result["output_abs_mean"] = float(np.mean(np.abs(output)))
        result["output_array"] = output
        if list(output.shape) != [len(EXPECTED_LABEL_NAMES)]:
            result["onnxruntime_status"] = (
                f"failed: unexpected_output_shape={list(output.shape)}"
            )
            result["message"] = "shape_mismatch"
        else:
            result["onnxruntime_status"] = "passed"
            result["message"] = "ok"
    except Exception as exc:
        result["onnxruntime_status"] = f"failed: {type(exc).__name__}: {exc}"
        result["message"] = "runtime_failed"
    return result


def write_raw_input_onnx_report(
    report_path: Path,
    dataset_path: Path,
    norm_path: Path,
    raw_sequences_path: Path,
    normalized_sequences_path: Path,
    normalized_onnx_path: Path,
    raw_onnx_path: Path,
    sample_dataset_index: int,
    normalized_result: dict[str, object],
    raw_result: dict[str, object],
    diff_abs_max: float | None,
    diff_abs_mean: float | None,
    input_name: str,
    output_name: str,
) -> None:
    lines = [
        "Pseudo Smooth Raw-Input ONNX Verification Report",
        "=" * 47,
        "",
        "Paths",
        f"- dataset_path: {dataset_path}",
        f"- norm_path: {norm_path}",
        f"- raw_sequence_path: {raw_sequences_path}",
        f"- normalized_sequence_path: {normalized_sequences_path}",
        f"- normalized_onnx_path: {normalized_onnx_path}",
        f"- raw_input_onnx_path: {raw_onnx_path}",
        "",
        "Verification Setup",
        f"- sample_dataset_index: {sample_dataset_index}",
        f"- input_name: {input_name}",
        f"- output_name: {output_name}",
        "",
        "Normalized-Input ONNX",
        f"- checker_status: {normalized_result['checker_status']}",
        f"- onnxruntime_status: {normalized_result['onnxruntime_status']}",
        f"- output_shape: {normalized_result['output_shape']}",
        f"- output_abs_max: {format_scalar(normalized_result['output_abs_max'])}",
        f"- output_abs_mean: {format_scalar(normalized_result['output_abs_mean'])}",
        "",
        "Raw-Input ONNX",
        f"- checker_status: {raw_result['checker_status']}",
        f"- onnxruntime_status: {raw_result['onnxruntime_status']}",
        f"- output_shape: {raw_result['output_shape']}",
        f"- output_abs_max: {format_scalar(raw_result['output_abs_max'])}",
        f"- output_abs_mean: {format_scalar(raw_result['output_abs_mean'])}",
        "",
        "Cross-Check",
        f"- output_diff_abs_max_between_raw_and_normalized_onnx: {format_scalar(diff_abs_max)}",
        f"- output_diff_abs_mean_between_raw_and_normalized_onnx: {format_scalar(diff_abs_mean)}",
        "",
        "Warnings",
        "- The current model is trained on pseudo_smooth_reference labels, not ground truth.",
        "- The raw-input ONNX model is the recommended FAST-LIVO2 runtime model because current C++ sends raw basic18 features.",
        "- Runtime use must stay in observe-only mode with apply_correction_en=false first.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_training_report_lines(
    args: argparse.Namespace,
    dataset_path: Path,
    norm_path: Path,
    raw_sequences_path: Path,
    normalized_sequences_path: Path,
    checkpoint_path: Path,
    normalized_onnx_path: Path,
    raw_onnx_path: Path,
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
    y_abs_stats: dict[str, np.ndarray],
    sanity_stats: dict[str, float],
    sample_rows: list[dict[str, object]],
    normalized_export_status: str,
    raw_export_status: str,
    normalized_verify: dict[str, object],
    raw_verify: dict[str, object],
    diff_abs_max: float | None,
    diff_abs_mean: float | None,
) -> list[str]:
    lines = [
        "Mamba Pose Pseudo-Label Baseline Training Report",
        "=" * 48,
        "",
        "Input / Output Paths",
        f"- dataset_path: {dataset_path}",
        f"- norm_path: {norm_path}",
        f"- raw_sequence_path: {raw_sequences_path}",
        f"- normalized_sequence_path: {normalized_sequences_path}",
        f"- checkpoint_path: {checkpoint_path}",
        f"- normalized_onnx_path: {normalized_onnx_path}",
        f"- raw_input_onnx_path: {raw_onnx_path}",
        f"- train_log_csv_path: {log_csv_path}",
        f"- report_path: {report_path}",
        "",
        "Dataset Summary",
        f"- X shape: {list(dataset_info['x'].shape)}",
        f"- y shape: {list(dataset_info['y'].shape)}",
        f"- train_size: {train_size}",
        f"- val_size: {val_size}",
        f"- label_type: {dataset_info['label_type']}",
        f"- smooth_window_sec: {format_scalar(dataset_info['smooth_window_sec'])}",
        f"- feature_names: {', '.join(dataset_info['feature_names'])}",
        f"- label_names: {', '.join(dataset_info['label_names'])}",
        "",
        "y Absolute Statistics",
    ]

    for dim_idx, label_name in enumerate(dataset_info["label_names"]):
        lines.append(
            f"- {label_name}: abs_max={format_scalar(y_abs_stats['abs_max'][dim_idx])}, "
            f"abs_mean={format_scalar(y_abs_stats['abs_mean'][dim_idx])}, "
            f"abs_p90={format_scalar(y_abs_stats['abs_p90'][dim_idx])}, "
            f"abs_p95={format_scalar(y_abs_stats['abs_p95'][dim_idx])}, "
            f"abs_p99={format_scalar(y_abs_stats['abs_p99'][dim_idx])}"
        )

    lines.extend(
        [
            "",
            "Model / Training",
            f"- model_type: {args.model_type}",
            f"- model_config: {model_config}",
            f"- epochs: {args.epochs}",
            f"- batch_size: {args.batch_size}",
            f"- lr: {format_scalar(args.lr)}",
            f"- weight_decay: {format_scalar(args.weight_decay)}",
            f"- device: {args.device}",
            f"- best_train_loss: {format_scalar(best_train_loss)}",
            f"- best_val_loss: {format_scalar(best_val_loss)}",
            f"- final_train_loss: {format_scalar(final_train_loss)}",
            f"- final_val_loss: {format_scalar(final_val_loss)}",
            "",
            "Sanity Inference",
            f"- pred_abs_max: {format_scalar(sanity_stats['pred_abs_max'])}",
            f"- pred_abs_mean: {format_scalar(sanity_stats['pred_abs_mean'])}",
            f"- target_abs_max: {format_scalar(sanity_stats['target_abs_max'])}",
            f"- target_abs_mean: {format_scalar(sanity_stats['target_abs_mean'])}",
            f"- error_abs_max: {format_scalar(sanity_stats['error_abs_max'])}",
            f"- error_abs_mean: {format_scalar(sanity_stats['error_abs_mean'])}",
            "",
            "Sanity Prediction Samples",
        ]
    )

    for row in sample_rows:
        lines.append(f"- sample_{row['index']}_pred: {row['pred']}")
        lines.append(f"  sample_{row['index']}_target: {row['target']}")

    lines.extend(
        [
            "",
            "ONNX",
            f"- normalized_onnx_export_status: {normalized_export_status}",
            f"- raw_input_onnx_export_status: {raw_export_status}",
            f"- normalized_onnx_checker_status: {normalized_verify['checker_status']}",
            f"- normalized_onnxruntime_status: {normalized_verify['onnxruntime_status']}",
            f"- raw_input_onnx_checker_status: {raw_verify['checker_status']}",
            f"- raw_input_onnxruntime_status: {raw_verify['onnxruntime_status']}",
            f"- normalized_onnx_output_abs_max: {format_scalar(normalized_verify['output_abs_max'])}",
            f"- normalized_onnx_output_abs_mean: {format_scalar(normalized_verify['output_abs_mean'])}",
            f"- raw_input_onnx_output_abs_max: {format_scalar(raw_verify['output_abs_max'])}",
            f"- raw_input_onnx_output_abs_mean: {format_scalar(raw_verify['output_abs_mean'])}",
            (
                "- output_diff_abs_max_between_raw_and_normalized_onnx: "
                f"{format_scalar(diff_abs_max)}"
            ),
            (
                "- output_diff_abs_mean_between_raw_and_normalized_onnx: "
                f"{format_scalar(diff_abs_mean)}"
            ),
            f"- onnx_input_name: {args.input_name}",
            f"- onnx_output_name: {args.output_name}",
            f"- normalized_onnx_input_shape: [{dataset_info['seq_len']}, {dataset_info['x'].shape[2]}]",
            f"- normalized_onnx_output_shape: [{dataset_info['y'].shape[1]}]",
            f"- raw_input_onnx_input_shape: [{dataset_info['seq_len']}, {dataset_info['x'].shape[2]}]",
            f"- raw_input_onnx_output_shape: [{dataset_info['y'].shape[1]}]",
            "",
            "Warnings",
            "- Current model is trained on pseudo_smooth_reference pseudo labels, not ground truth.",
            "- Current model cannot directly prove true dynamic pose correction ability.",
            "- FAST-LIVO2 runtime validation must keep apply_correction_en=false for observe-only behavior first.",
            "- Only after observe-only output stays stable should very cautious closed-loop discussion even start.",
        ]
    )
    return lines


def main() -> int:
    args = parse_args()
    validate_args(args)
    set_global_seed(args.seed)

    dataset_path = args.dataset.expanduser().resolve()
    norm_path = args.norm.expanduser().resolve()
    raw_sequences_path = args.raw_sequences.expanduser().resolve()
    normalized_sequences_path = args.normalized_sequences.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    ensure_input_exists(dataset_path, "Dataset NPZ")
    ensure_input_exists(norm_path, "Norm NPZ")
    ensure_input_exists(raw_sequences_path, "Raw sequence NPZ")
    ensure_input_exists(normalized_sequences_path, "Normalized sequence NPZ")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "pseudo_smooth_mamba_pose.pt"
    normalized_onnx_path = output_dir / "pseudo_smooth_mamba_pose.onnx"
    raw_onnx_path = output_dir / "pseudo_smooth_mamba_pose_raw_input.onnx"
    log_csv_path = output_dir / "pseudo_smooth_mamba_pose_train_log.csv"
    report_path = output_dir / "pseudo_smooth_mamba_pose_train_report.txt"
    raw_input_onnx_report_path = (
        output_dir / "pseudo_smooth_mamba_pose_raw_input_onnx_report.txt"
    )

    dataset_npz = load_npz(dataset_path)
    dataset_info = validate_dataset_npz(dataset_npz)
    if str(norm_path) != dataset_info["norm_path"]:
        raise SystemExit(
            "Dataset norm_path does not match --norm.\n"
            f"Dataset: {dataset_info['norm_path']}\n"
            f"Arg: {norm_path}"
        )

    norm_npz = load_npz(norm_path)
    norm_stats = validate_norm_npz(
        norm_npz, expected_path=norm_path, feature_names=dataset_info["feature_names"]
    )

    raw_sequence_npz = load_npz(raw_sequences_path)
    normalized_sequence_npz = load_npz(normalized_sequences_path)
    raw_sequence_info = validate_full_sequence_npz(
        raw_sequence_npz, x_field="X", feature_names=dataset_info["feature_names"]
    )
    normalized_sequence_info = validate_full_sequence_npz(
        normalized_sequence_npz,
        x_field="X_normalized",
        feature_names=dataset_info["feature_names"],
    )
    validate_sequence_alignment(raw_sequence_info, normalized_sequence_info)

    if dataset_info["seq_len"] != raw_sequence_info["seq_len"]:
        raise SystemExit(
            "Dataset seq_len does not match raw sequence NPZ seq_len. "
            f"Got {dataset_info['seq_len']} vs {raw_sequence_info['seq_len']}."
        )

    device = resolve_device(args.device)
    dataset = SequenceRegressionDataset(dataset_info["x"], dataset_info["y"])
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
        "model_type": args.model_type,
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
    model = model.to(device)
    final_train_loss = float(log_rows[-1]["train_loss"])
    final_val_loss = float(log_rows[-1]["val_loss"])

    write_train_log_csv(log_rows, log_csv_path)
    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model.cpu(),
        model_config=model_config,
        dataset_info=dataset_info,
        dataset_path=dataset_path,
        best_val_loss=best_val_loss,
    )

    model = model.to(device).eval()
    val_predictions, val_targets = collect_predictions(model, val_loader, device)
    if val_predictions.shape[0] == 0:
        raise SystemExit("Validation predictions are empty.")
    absolute_error = np.abs(val_predictions - val_targets)
    sanity_stats = {
        "pred_abs_max": float(np.max(np.abs(val_predictions))),
        "pred_abs_mean": float(np.mean(np.abs(val_predictions))),
        "target_abs_max": float(np.max(np.abs(val_targets))),
        "target_abs_mean": float(np.mean(np.abs(val_targets))),
        "error_abs_max": float(np.max(absolute_error)),
        "error_abs_mean": float(np.mean(absolute_error)),
    }
    sample_rows = build_first_five_samples(val_predictions, val_targets)
    y_abs_stats = compute_abs_percentiles(dataset_info["y"])

    normalized_export_status = "disabled"
    raw_export_status = "disabled"
    normalized_verify = {
        "checker_status": "not_run",
        "onnxruntime_status": "not_run",
        "output_abs_max": None,
        "output_abs_mean": None,
        "output_shape": None,
        "output_array": None,
    }
    raw_verify = {
        "checker_status": "not_run",
        "onnxruntime_status": "not_run",
        "output_abs_max": None,
        "output_abs_mean": None,
        "output_shape": None,
        "output_array": None,
    }
    diff_abs_max: float | None = None
    diff_abs_mean: float | None = None

    sample_dataset_index = 0
    if args.export_onnx:
        normalized_export_status = export_normalized_input_onnx(
            model=model.cpu(),
            onnx_path=normalized_onnx_path,
            seq_len=dataset_info["seq_len"],
            feature_dim=dataset_info["x"].shape[2],
            input_name=args.input_name,
            output_name=args.output_name,
            onnx_opset=args.onnx_opset,
        )
        raw_export_status = export_raw_input_onnx(
            model=model.cpu(),
            mean=norm_stats["mean"],
            std_safe=norm_stats["std_safe"],
            onnx_path=raw_onnx_path,
            seq_len=dataset_info["seq_len"],
            feature_dim=dataset_info["x"].shape[2],
            input_name=args.input_name,
            output_name=args.output_name,
            onnx_opset=args.onnx_opset,
        )

        onnx_module, ort_module = try_import_onnx_tools()
        if onnx_module is not None and ort_module is not None:
            sequence_map = build_sequence_index_map(
                source_indices=normalized_sequence_info["source_indices"],
                timestamps=normalized_sequence_info["timestamps"],
            )
            full_index = find_matching_full_sequence_index(
                dataset_source_indices_row=dataset_info["source_indices"][sample_dataset_index],
                dataset_timestamps_row=dataset_info["timestamps"][sample_dataset_index],
                full_source_indices=normalized_sequence_info["source_indices"],
                full_timestamps=normalized_sequence_info["timestamps"],
                sequence_map=sequence_map,
            )
            normalized_sample = normalized_sequence_info["x"][full_index]
            raw_sample = raw_sequence_info["x"][full_index]
            normalized_verify = verify_single_onnx(
                onnx_module=onnx_module,
                ort_module=ort_module,
                onnx_path=normalized_onnx_path,
                input_name=args.input_name,
                output_name=args.output_name,
                sample_input=normalized_sample,
            )
            raw_verify = verify_single_onnx(
                onnx_module=onnx_module,
                ort_module=ort_module,
                onnx_path=raw_onnx_path,
                input_name=args.input_name,
                output_name=args.output_name,
                sample_input=raw_sample,
            )
            if (
                normalized_verify["output_array"] is not None
                and raw_verify["output_array"] is not None
            ):
                diff = np.abs(
                    np.asarray(raw_verify["output_array"], dtype=np.float32)
                    - np.asarray(normalized_verify["output_array"], dtype=np.float32)
                )
                diff_abs_max = float(np.max(diff))
                diff_abs_mean = float(np.mean(diff))
        else:
            normalized_verify["checker_status"] = "skipped"
            normalized_verify["onnxruntime_status"] = "skipped"
            raw_verify["checker_status"] = "skipped"
            raw_verify["onnxruntime_status"] = "skipped"

    training_report_lines = build_training_report_lines(
        args=args,
        dataset_path=dataset_path,
        norm_path=norm_path,
        raw_sequences_path=raw_sequences_path,
        normalized_sequences_path=normalized_sequences_path,
        checkpoint_path=checkpoint_path,
        normalized_onnx_path=normalized_onnx_path,
        raw_onnx_path=raw_onnx_path,
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
        y_abs_stats=y_abs_stats,
        sanity_stats=sanity_stats,
        sample_rows=sample_rows,
        normalized_export_status=normalized_export_status,
        raw_export_status=raw_export_status,
        normalized_verify=normalized_verify,
        raw_verify=raw_verify,
        diff_abs_max=diff_abs_max,
        diff_abs_mean=diff_abs_mean,
    )
    report_path.write_text("\n".join(training_report_lines) + "\n", encoding="utf-8")

    write_raw_input_onnx_report(
        report_path=raw_input_onnx_report_path,
        dataset_path=dataset_path,
        norm_path=norm_path,
        raw_sequences_path=raw_sequences_path,
        normalized_sequences_path=normalized_sequences_path,
        normalized_onnx_path=normalized_onnx_path,
        raw_onnx_path=raw_onnx_path,
        sample_dataset_index=sample_dataset_index,
        normalized_result=normalized_verify,
        raw_result=raw_verify,
        diff_abs_max=diff_abs_max,
        diff_abs_mean=diff_abs_mean,
        input_name=args.input_name,
        output_name=args.output_name,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
