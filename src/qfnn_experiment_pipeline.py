#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QFNN-only image-classification experiment pipeline.

This is a standalone reference implementation for the QFNN architecture in
the accompanying paper.  It intentionally has no dependency on ModelB or on
any local project module.

What the full ``--stage all`` workflow does
-------------------------------------------
1. For each of MNIST, FashionMNIST, and KMNIST separately, create a fixed
   stratified train/validation split from the official training partition.
   The official test partition is not loaded by the search.
2. Resume or create one Optuna study per dataset.  Sixty trials (total, not
   sixty new trials after every restart) search the declared space.  Every
   trial is trained for 30 epochs under two independent search seeds.
   QFNN depth in {1,2} and Chebyshev degree in {1,2,3} are architectural
   hyperparameters.  Training shots are fixed by --train-shots and are never
   tuned.
3. Re-train the selected per-dataset recipe for 100 epochs under six seeds.
   Every seed selects its checkpoint using validation data only.
4. Only after *all* checkpoints for a dataset have been selected, load its
   official test partition.  Report exact-statevector test results and
   repeated finite-shot test results.
5. Write JSON/CSV/Markdown summaries and mean +/- sample-standard-deviation
   train/validation accuracy plots.  Test accuracy is deliberately not drawn
   as an epoch-by-epoch curve because test data is never monitored in training.

The default stage is ``search`` so that launching the script does not
accidentally begin the much longer 3-dataset x 6-seed x 100-epoch final run.
Inspect ``selected_recipes.md`` and then invoke ``--stage final``.  Use
``--stage all`` only when that entire workload is intended.

The sixty-trial, two-seed search is a substantial validation search, but it is
still not an exhaustive architecture or optimizer comparison.  It covers each
requested depth-degree pair once before TPE spends the remaining budget.

The default architecture is real-valued (RY reference states) and unphased.
Pass --use-phase to add one trainable RZ phase to each QFNN layer.  The
terminal scalar-beta plus class-bias affine head is always present; only its
parameters are trained, not its presence/absence.

Important experimental rule
---------------------------
Optuna sees validation data, never test data.  Once a recipe is selected,
the final seed runs again select checkpoints on validation data.  Test data
is evaluated only after selection and is never used to choose a recipe,
epoch, degree, depth, or random seed.

Dependencies
------------
    pip install torch torchvision numpy optuna matplotlib

Examples
--------
    # No downloads: check both circuit depths, Chebyshev identities and grads
    python qfnn_experiment_pipeline.py --stage self-test

    # Small end-to-end debug run (downloads MNIST if necessary)
    python qfnn_experiment_pipeline.py --stage all --quick --device cpu

    # Recommended staged experiment: search now, inspect, then run final
    python qfnn_experiment_pipeline.py --stage search --device cuda
    python qfnn_experiment_pipeline.py --stage final --device cuda

    # Explicitly request both stages without stopping
    python qfnn_experiment_pipeline.py --stage all --device cuda

    # Opt-in CIFAR-10 (classes 0,...,7, converted to grayscale)
    python qfnn_experiment_pipeline.py --datasets MNIST FashionMNIST KMNIST CIFAR10

    # Compact validation-only 2 x 3 depth/degree diagnostic on FashionMNIST
    python qfnn_experiment_pipeline.py --stage ablation --device cuda

Output layout
-------------
    qfnn_experiments/
      run_config.json
      selected_recipes.json
      selected_recipes.md
      studies/<dataset>_<search-signature>.db
      search/<dataset>/<search-signature>/trial_XXXX/seed_XXXX/...
      final/<dataset>/seed_XXXX/<run-signature>/...
      reports/final_results.csv
      reports/final_results.json
      reports/final_results.md
      reports/<dataset>_accuracy.png
      reports/<dataset>_accuracy.pdf

Notes on the simulator
----------------------
* Inputs are grayscale 8 x 8 images, flattened and L2-normalized into a
  64-amplitude (six-work-qubit) state.
* A layer uses a uniform three-qubit address register (eight references) and
  one interference ancilla.  The direct readout is the exact marginal of the
  final address+ancilla register.  Equivalently, hardware may measure all
  registers and discard the work-register bits; measuring the work register
  is not mathematically required to obtain this marginal.
* For depth two, the second local reference projector acts on A1+c1, while
  the current-state reflection acts on the entire retained W+A1+c1 state.
  The first-layer registers are retained coherently; they are not measured or
  classically re-encoded between layers.
* Finite-shot training uses one multinomial draw over the complete final
  address+ancilla distribution and a straight-through estimator.  This is a
  simulator-side stochastic training model, not a claim about a particular
  hardware gradient estimator.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    import optuna
except Exception:  # Imported lazily for self-test/evaluation-only usability.
    optuna = None

try:
    from torchvision.datasets import CIFAR10, FashionMNIST, KMNIST, MNIST
except Exception:
    CIFAR10 = FashionMNIST = KMNIST = MNIST = None


CLASSES = tuple(range(8))
SUPPORTED_DATASETS = ("MNIST", "FashionMNIST", "KMNIST", "CIFAR10")
DEFAULT_DATASETS = ("MNIST", "FashionMNIST", "KMNIST")
EXPERIMENT_VERSION = "qfnn-pipeline-2026-08-13-v3"
SAMPLER_NAME = "Optuna TPESampler"
SAMPLER_MULTIVARIATE = False


# =============================================================================
# Reproducibility, serialization, and small reporting helpers
# =============================================================================


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = not bool(deterministic)
        torch.backends.cudnn.deterministic = bool(deterministic)


def source_sha256() -> str:
    """Hash the exact source file whose code is producing the artifacts."""
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def core_provenance(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    """Code/runtime and fixed training choices shared by search and final runs."""
    try:
        torchvision_version = importlib.metadata.version("torchvision")
    except importlib.metadata.PackageNotFoundError:
        torchvision_version = None
    if device.type == "cuda" and torch.cuda.is_available():
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        accelerator_name = torch.cuda.get_device_name(device_index)
    else:
        accelerator_name = None
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "source_sha256": source_sha256(),
        "python_version": sys.version,
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "torchvision_version": torchvision_version,
        "device": str(device),
        "accelerator_name": accelerator_name,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "allow_nondeterminism": bool(args.allow_nondeterminism),
        "gradient_clip": float(args.gradient_clip),
        "train_shots": int(args.train_shots),
        "reference_ansatz_depth": int(args.reference_ansatz_depth),
        "use_phase": bool(args.use_phase),
        "beta_init": float(args.beta_init),
        "workers": int(args.workers),
        "eval_batch_size": int(args.eval_batch_size),
    }


def training_run_provenance(
    *,
    args: argparse.Namespace,
    device: torch.device,
    dataset: str,
    recipe: "Recipe",
    seed: int,
    epochs: int,
    train_per_class: int,
) -> Dict[str, Any]:
    payload = core_provenance(args, device)
    payload.update(
        {
            "dataset": str(dataset),
            "recipe": asdict(recipe),
            "seed": int(seed),
            "epochs": int(epochs),
            "split_seed": int(args.split_seed),
            "val_fraction": float(args.val_fraction),
            "val_per_class": int(args.val_per_class),
            "train_per_class": int(train_per_class),
            "data_root": str(Path(args.data_root).resolve()),
        }
    )
    payload["training_fingerprint_sha256"] = stable_hash(payload, length=64)
    return payload


def safe_suffix_token(raw: str) -> str:
    """Filesystem-safe study suffix with a collision-resistant raw-suffix hash."""
    raw = str(raw)
    if not raw:
        return ""
    readable = "".join(character if character.isalnum() or character in "-_" else "_" for character in raw)
    readable = readable.strip("_-")[:32] or "suffix"
    return f"{readable}-{stable_hash(raw, 8)}"


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: jsonable(row.get(k)) for k in keys})
    os.replace(tmp, path)


def append_csv(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow({k: jsonable(v) for k, v in row.items()})


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    out = ["| " + " | ".join(map(cell, headers)) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    out.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in rows)
    return "\n".join(out) + "\n"


def sample_mean_std(values: Sequence[float]) -> Dict[str, float]:
    x = np.asarray([float(v) for v in values], dtype=np.float64)
    return {
        "mean": float(x.mean()) if x.size else float("nan"),
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "min": float(x.min()) if x.size else float("nan"),
        "max": float(x.max()) if x.size else float("nan"),
        "n": int(x.size),
    }


def stable_hash(payload: Any, length: int = 12) -> str:
    blob = json.dumps(jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:length]


def safe_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[device] CUDA was requested but is unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def dataset_label(name: str) -> str:
    return "CIFAR8" if name == "CIFAR10" else f"{name}8"


# =============================================================================
# Data: official training split -> fixed stratified train/validation split
# =============================================================================


@dataclass
class DataBundle:
    dataset: str
    train: TensorDataset
    val: TensorDataset
    metadata: Dict[str, Any]


def _require_torchvision() -> None:
    if MNIST is None:
        raise RuntimeError(
            "torchvision could not be imported. Install a torchvision build "
            "compatible with the installed PyTorch, then retry."
        )


def _load_raw_partition(dataset: str, root: Path, train: bool, download: bool) -> Tuple[np.ndarray, np.ndarray]:
    _require_torchvision()
    if dataset == "MNIST":
        obj = MNIST(root=str(root), train=train, download=download)
        return obj.data.numpy(), obj.targets.numpy()
    if dataset == "FashionMNIST":
        obj = FashionMNIST(root=str(root), train=train, download=download)
        return obj.data.numpy(), obj.targets.numpy()
    if dataset == "KMNIST":
        obj = KMNIST(root=str(root), train=train, download=download)
        return obj.data.numpy(), obj.targets.numpy()
    if dataset == "CIFAR10":
        obj = CIFAR10(root=str(root), train=train, download=download)
        return np.asarray(obj.data), np.asarray(obj.targets, dtype=np.int64)
    raise ValueError(f"Unsupported dataset: {dataset}")


def _filter_and_remap(raw_x: np.ndarray, raw_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    raw_y = np.asarray(raw_y, dtype=np.int64).reshape(-1)
    mask = np.isin(raw_y, np.asarray(CLASSES, dtype=np.int64))
    # CLASSES is 0,...,7, so remapping is the identity.  Keep this explicit in
    # case a future experiment changes the selected original labels.
    mapper = {old: new for new, old in enumerate(CLASSES)}
    y = np.asarray([mapper[int(v)] for v in raw_y[mask]], dtype=np.int64)
    return np.asarray(raw_x)[mask], y


def _grayscale(raw: np.ndarray) -> np.ndarray:
    x = np.asarray(raw)
    if x.ndim == 4 and x.shape[-1] == 3:
        x = x.astype(np.float32) / 255.0
        return (0.2989 * x[..., 0] + 0.5870 * x[..., 1] + 0.1140 * x[..., 2]).astype(np.float32)
    if x.ndim == 3:
        x = x.astype(np.float32)
        if x.size and x.max() > 2.0:
            x /= 255.0
        return x
    raise ValueError(f"Expected [N,H,W] or [N,H,W,3], got {x.shape}")


def amplitude_preprocess(raw: np.ndarray) -> torch.Tensor:
    """Grayscale -> bilinear 8x8 -> 64 amplitudes -> per-example L2 norm."""
    x = torch.from_numpy(_grayscale(raw)).to(torch.float32).unsqueeze(1)
    x = F.interpolate(x, size=(8, 8), mode="bilinear", align_corners=False).squeeze(1)
    x = x.reshape(x.size(0), 64)
    norms = torch.linalg.vector_norm(x, dim=1, keepdim=True)
    bad = (~torch.isfinite(norms)) | (norms <= 1e-12)
    if bool(bad.any()):
        count = int(bad.sum().item())
        raise ValueError(
            f"Amplitude preprocessing found {count} non-finite or zero-norm image(s); "
            "amplitude encoding is undefined for those inputs."
        )
    if not bool(torch.isfinite(x).all()):
        raise ValueError("Amplitude preprocessing received non-finite pixel values.")
    return x / norms


def _stratified_indices(
    labels: np.ndarray,
    *,
    split_seed: int,
    val_fraction: float,
    val_per_class: int,
    train_per_class: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(split_seed))
    train_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []
    for class_id in range(len(CLASSES)):
        ids = np.flatnonzero(labels == class_id).copy()
        rng.shuffle(ids)
        nominal_val = max(1, int(round(float(val_fraction) * ids.size)))
        n_val = min(nominal_val, int(val_per_class), max(1, ids.size - 1))
        val_ids = ids[:n_val]
        remaining = ids[n_val:]
        n_train = remaining.size if train_per_class <= 0 else min(int(train_per_class), remaining.size)
        train_ids = remaining[:n_train]
        if train_ids.size == 0:
            raise ValueError(f"No training examples remain for class {class_id}.")
        train_parts.append(train_ids)
        val_parts.append(val_ids)
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def prepare_train_val(
    dataset: str,
    *,
    root: Path,
    download: bool,
    split_seed: int,
    val_fraction: float,
    val_per_class: int,
    train_per_class: int,
) -> DataBundle:
    raw_x, raw_y = _load_raw_partition(dataset, root, train=True, download=download)
    raw_x, y = _filter_and_remap(raw_x, raw_y)
    train_idx, val_idx = _stratified_indices(
        y,
        split_seed=split_seed,
        val_fraction=val_fraction,
        val_per_class=val_per_class,
        train_per_class=train_per_class,
    )
    x_train = amplitude_preprocess(raw_x[train_idx])
    x_val = amplitude_preprocess(raw_x[val_idx])
    y_train = torch.from_numpy(y[train_idx]).long()
    y_val = torch.from_numpy(y[val_idx]).long()
    metadata = {
        "dataset": dataset,
        "classes_original": list(CLASSES),
        "classes_model": list(range(8)),
        "official_partition_used": "train only",
        "split": "fixed stratified split within official training partition",
        "split_seed": int(split_seed),
        "val_fraction_before_cap": float(val_fraction),
        "val_per_class_cap": int(val_per_class),
        "train_per_class_cap": int(train_per_class),
        "n_train": int(y_train.numel()),
        "n_val": int(y_val.numel()),
        "preprocessing": "grayscale -> bilinear 8x8 -> flatten64 -> L2 normalize",
    }
    return DataBundle(
        dataset=dataset,
        train=TensorDataset(x_train, y_train),
        val=TensorDataset(x_val, y_val),
        metadata=metadata,
    )


def prepare_official_test(
    dataset: str,
    *,
    root: Path,
    download: bool,
    max_per_class: int,
    selection_seed: int,
) -> Tuple[TensorDataset, Dict[str, Any]]:
    """Load test only when called; callers invoke this after checkpoint selection."""
    raw_x, raw_y = _load_raw_partition(dataset, root, train=False, download=download)
    raw_x, y = _filter_and_remap(raw_x, raw_y)
    if max_per_class > 0:
        rng = np.random.default_rng(int(selection_seed))
        pieces = []
        for class_id in range(8):
            ids = np.flatnonzero(y == class_id).copy()
            rng.shuffle(ids)
            pieces.append(ids[: min(int(max_per_class), ids.size)])
        selected = np.concatenate(pieces)
        rng.shuffle(selected)
        raw_x, y = raw_x[selected], y[selected]
    x = amplitude_preprocess(raw_x)
    labels = torch.from_numpy(y).long()
    meta = {
        "dataset": dataset,
        "official_partition_used": "test",
        "n_test": int(labels.numel()),
        "test_per_class_cap": int(max_per_class),
    }
    return TensorDataset(x, labels), meta


def make_loader(
    dataset: TensorDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(workers),
        pin_memory=device.type == "cuda",
        generator=generator,
        persistent_workers=bool(workers > 0),
    )


# =============================================================================
# Standalone statevector primitives
# =============================================================================


def complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if real_dtype == torch.float64 else torch.complex64


def hadamard(*, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=dtype, device=device) / math.sqrt(2.0)


def ry(angle: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    angle = angle.to(device=device)
    c = torch.cos(angle / 2).to(dtype)
    s = torch.sin(angle / 2).to(dtype)
    return torch.stack((torch.stack((c, -s)), torch.stack((s, c))))


def rz(angle: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    angle = angle.to(device=device)
    zero = torch.zeros((), dtype=dtype, device=device)
    e0 = torch.exp((-0.5j * angle).to(dtype))
    e1 = torch.exp((0.5j * angle).to(dtype))
    return torch.stack((torch.stack((e0, zero)), torch.stack((zero, e1))))


def apply_one_qubit_gate(state: torch.Tensor, n_qubits: int, qubit: int, gate: torch.Tensor) -> torch.Tensor:
    """Apply a 2x2 gate; qubit 0 is the most-significant tensor axis."""
    if state.ndim != 2 or state.size(1) != 2 ** int(n_qubits):
        raise ValueError("state must have shape [batch, 2**n_qubits]")
    left = 2 ** int(qubit)
    right = 2 ** (int(n_qubits) - int(qubit) - 1)
    view = state.view(state.size(0), left, 2, right)
    out = torch.einsum("ij,bljr->blir", gate.to(state.dtype), view)
    return out.reshape_as(state)


_CNOT_PERMUTATION_CACHE: Dict[Tuple[int, int, int, str], torch.Tensor] = {}


def cnot_permutation(n_qubits: int, control: int, target: int, device: torch.device) -> torch.Tensor:
    cache_key = (int(n_qubits), int(control), int(target), str(device))
    cached = _CNOT_PERMUTATION_CACHE.get(cache_key)
    if cached is not None:
        return cached
    values = []
    for output_index in range(2 ** int(n_qubits)):
        bits = [(output_index >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        if bits[control] == 1:
            bits[target] ^= 1
        input_index = 0
        for bit in bits:
            input_index = (input_index << 1) | bit
        values.append(input_index)
    permutation = torch.tensor(values, dtype=torch.long, device=device)
    _CNOT_PERMUTATION_CACHE[cache_key] = permutation
    return permutation


def apply_cnot(state: torch.Tensor, n_qubits: int, control: int, target: int) -> torch.Tensor:
    # CNOT is self-inverse, so output[j] = input[CNOT(j)].
    return state.index_select(1, cnot_permutation(n_qubits, control, target, state.device))


def bits_of_int(value: int, width: int) -> List[int]:
    return [(int(value) >> (int(width) - 1 - i)) & 1 for i in range(int(width))]


def _reflect_target_block(block: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """(I-2|r><r|) on the target axis of [batch,target,rest]."""
    if reference.ndim == 1:
        phi = reference.to(block.device, block.dtype).unsqueeze(0).expand(block.size(0), -1)
    elif reference.ndim == 2 and reference.size(0) == block.size(0):
        phi = reference.to(block.device, block.dtype)
    else:
        raise ValueError("reference must be [target] or [batch,target]")
    phi = phi / torch.linalg.vector_norm(phi, dim=1, keepdim=True).clamp_min(1e-12)
    coefficient = torch.einsum("bt,btr->br", torch.conj(phi), block)
    return block - 2.0 * phi.unsqueeze(-1) * coefficient.unsqueeze(1)


def conditioned_reflection(
    state: torch.Tensor,
    *,
    n_qubits: int,
    targets: Sequence[int],
    reference: torch.Tensor,
    controls: Sequence[int],
    control_bits: Sequence[int],
) -> torch.Tensor:
    """Apply a target reflection on exactly one computational control word."""
    targets = list(map(int, targets))
    controls = list(map(int, controls))
    control_bits = list(map(int, control_bits))
    if len(controls) != len(control_bits):
        raise ValueError("controls and control_bits must have equal length")
    if set(targets).intersection(controls):
        raise ValueError("target and control qubits must be disjoint")
    others = [q for q in range(int(n_qubits)) if q not in targets and q not in controls]
    ordered = controls + targets + others
    axes = [0] + [1 + q for q in ordered]
    tensor = state.view(state.size(0), *([2] * int(n_qubits))).permute(*axes).contiguous()
    nc, nt, no = len(controls), len(targets), len(others)
    tensor = tensor.view(state.size(0), 2**nc, 2**nt, 2**no)
    condition_index = 0
    for bit in control_bits:
        condition_index = (condition_index << 1) | bit
    selected = tensor[:, condition_index, :, :]
    reflected = _reflect_target_block(selected, reference)
    mask = F.one_hot(
        torch.tensor(condition_index, device=state.device), num_classes=2**nc
    ).to(state.dtype).view(1, 2**nc, 1, 1)
    tensor = tensor * (1.0 - mask) + reflected.unsqueeze(1) * mask
    tensor = tensor.view(state.size(0), *([2] * int(n_qubits)))
    inverse = [0] * (int(n_qubits) + 1)
    for new_axis, old_axis in enumerate(axes):
        inverse[old_axis] = new_axis
    return tensor.permute(*inverse).contiguous().view_as(state)


def marginal_probabilities(state: torch.Tensor, n_qubits: int, keep: Sequence[int]) -> torch.Tensor:
    """Born marginal in the listed qubit order."""
    keep = list(map(int, keep))
    rest = [q for q in range(int(n_qubits)) if q not in keep]
    probabilities = state.abs().square().view(state.size(0), *([2] * int(n_qubits)))
    axes = [0] + [1 + q for q in keep + rest]
    probabilities = probabilities.permute(*axes).contiguous()
    probabilities = probabilities.view(state.size(0), 2 ** len(keep), 2 ** len(rest)).sum(dim=2)
    return probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)


def multinomial_probabilities(probabilities: torch.Tensor, shots: int) -> torch.Tensor:
    """One joint multinomial count vector per batch item."""
    if int(shots) <= 0:
        return probabilities
    p = probabilities.detach().to(torch.float32).clamp_min(0.0)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)
    counts = torch.distributions.Multinomial(total_count=int(shots), probs=p).sample()
    return counts / float(shots)


def straight_through(noisy: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    return clean + (noisy.to(clean.dtype) - clean).detach()


def chebyshev_t(x: torch.Tensor, degree: int) -> torch.Tensor:
    if int(degree) == 0:
        return torch.ones_like(x)
    if int(degree) == 1:
        return x
    t0, t1 = torch.ones_like(x), x
    for _ in range(2, int(degree) + 1):
        t0, t1 = t1, 2.0 * x * t1 - t0
    return t1


# =============================================================================
# QFNN model: coherent one- or two-layer register construction
# =============================================================================


class RingRYReferences(nn.Module):
    """Eight independently trainable real reference states."""

    def __init__(self, count: int, n_qubits: int, ansatz_depth: int, init_scale: float = 0.08):
        super().__init__()
        self.count = int(count)
        self.n_qubits = int(n_qubits)
        self.ansatz_depth = int(ansatz_depth)
        shape = (self.count, self.ansatz_depth + 1, self.n_qubits)
        self.theta = nn.Parameter(torch.randn(shape, dtype=torch.float32) * float(init_scale))

    def state(self, index: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        state = torch.zeros(1, 2**self.n_qubits, dtype=dtype, device=device)
        state[0, 0] = 1.0
        angles = self.theta[int(index)].to(device=device)
        for qubit in range(self.n_qubits):
            state = apply_one_qubit_gate(
                state, self.n_qubits, qubit, ry(angles[0, qubit], dtype=dtype, device=device)
            )
        for layer in range(self.ansatz_depth):
            for qubit in range(self.n_qubits):
                state = apply_cnot(state, self.n_qubits, qubit, (qubit + 1) % self.n_qubits)
            for qubit in range(self.n_qubits):
                state = apply_one_qubit_gate(
                    state,
                    self.n_qubits,
                    qubit,
                    ry(angles[layer + 1, qubit], dtype=dtype, device=device),
                )
        state = state.squeeze(0)
        return state / torch.linalg.vector_norm(state).clamp_min(1e-12)

    def all_states(self, *, dtype: torch.dtype, device: torch.device) -> List[torch.Tensor]:
        return [self.state(j, dtype=dtype, device=device) for j in range(self.count)]


@dataclass(frozen=True)
class ModelSpec:
    depth: int
    degree: int
    ansatz_depth: int = 4
    use_phase: bool = False
    n_work: int = 6
    n_classes: int = 8


class CoherentQFNN(nn.Module):
    """Literal statevector implementation of the paper's depth-1/2 QFNN."""

    def __init__(self, spec: ModelSpec):
        super().__init__()
        if spec.depth not in (1, 2):
            raise ValueError("Only depth 1 or 2 is implemented.")
        if spec.degree not in (1, 2, 3):
            raise ValueError("This experiment supports degrees 1, 2, and 3.")
        if spec.n_classes != 8 or spec.n_work != 6:
            raise ValueError("This pipeline fixes eight classes and six work qubits.")
        self.spec = spec
        self.n_address = 3
        self.W = list(range(6))
        self.A1 = list(range(6, 9))
        self.c1 = 9
        if spec.depth == 1:
            self.n_total = 10
            self.A2: List[int] = []
            self.c2: Optional[int] = None
        else:
            self.A2 = list(range(10, 13))
            self.c2 = 13
            self.n_total = 14

        self.layer1_references = RingRYReferences(8, 6, spec.ansatz_depth)
        if spec.depth == 2:
            # Local second-layer projectors act on A1+c1 (four qubits).
            self.layer2_references: Optional[RingRYReferences] = RingRYReferences(
                8, 4, spec.ansatz_depth
            )
        else:
            self.layer2_references = None

        if spec.use_phase:
            self.phase1 = nn.Parameter(torch.zeros((), dtype=torch.float32))
            if spec.depth == 2:
                self.phase2 = nn.Parameter(torch.zeros((), dtype=torch.float32))
            else:
                self.register_parameter("phase2", None)
        else:
            self.register_parameter("phase1", None)
            self.register_parameter("phase2", None)

        # Always-on affine readout: z_j = beta * alpha_j + b_j.
        self.beta = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(8, dtype=torch.float32))

    @property
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _initial_state(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or x.size(1) != 64:
            raise ValueError(f"QFNN inputs must have shape [batch,64], got {tuple(x.shape)}")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("QFNN amplitude inputs must be finite.")
        norms = torch.linalg.vector_norm(x, dim=1, keepdim=True)
        if bool((norms <= 1e-12).any()):
            count = int((norms <= 1e-12).sum().item())
            raise ValueError(
                f"QFNN received {count} zero-norm amplitude input(s); normalization is undefined."
            )
        normalized = x / norms
        x_complex = normalized.to(complex_dtype(x.dtype))
        state = torch.zeros(
            x.size(0), 2**self.n_total, dtype=x_complex.dtype, device=x.device
        )
        work_indices = torch.arange(64, device=x.device, dtype=torch.long)
        full_indices = work_indices << (self.n_total - 6)
        state[:, full_indices] = x_complex
        return state, x_complex

    def _uniform(self, state: torch.Tensor, qubits: Sequence[int]) -> torch.Tensor:
        gate = hadamard(dtype=state.dtype, device=state.device)
        for qubit in qubits:
            state = apply_one_qubit_gate(state, self.n_total, int(qubit), gate)
        return state

    def _layer1(self, state: torch.Tensor, x_state: torch.Tensor) -> torch.Tensor:
        state = self._uniform(state, self.A1 + [self.c1])
        references = self.layer1_references.all_states(dtype=state.dtype, device=state.device)
        for _ in range(self.spec.degree):
            # SELECT_j R_theta_j, controlled jointly on A1=j and c1=1.
            for branch, reference in enumerate(references):
                state = conditioned_reflection(
                    state,
                    n_qubits=self.n_total,
                    targets=self.W,
                    reference=reference,
                    controls=self.A1 + [self.c1],
                    control_bits=bits_of_int(branch, 3) + [1],
                )
            # One shared R_x on the complete c1=1 arm, independent of address.
            state = conditioned_reflection(
                state,
                n_qubits=self.n_total,
                targets=self.W,
                reference=x_state,
                controls=[self.c1],
                control_bits=[1],
            )
        if self.phase1 is not None:
            state = apply_one_qubit_gate(
                state,
                self.n_total,
                self.c1,
                rz(self.phase1, dtype=state.dtype, device=state.device),
            )
        return apply_one_qubit_gate(
            state, self.n_total, self.c1, hadamard(dtype=state.dtype, device=state.device)
        )

    def first_layer_state(self, x: torch.Tensor) -> torch.Tensor:
        state, x_state = self._initial_state(x)
        return self._layer1(state, x_state)

    def _incoming_big_reference(self, layer1_state: torch.Tensor) -> torch.Tensor:
        # A2+c2 remain |0...0>, so the nonzero column is precisely the retained
        # normalized W+A1+c1 incoming state.  No measurement or re-encoding.
        big_dimension = 2 ** (6 + 3 + 1)
        rest_dimension = 2 ** (self.n_total - 10)
        view = layer1_state.view(layer1_state.size(0), big_dimension, rest_dimension)
        incoming = view[:, :, 0]
        return incoming / torch.linalg.vector_norm(incoming, dim=1, keepdim=True).clamp_min(1e-12)

    def _layer2(self, layer1_state: torch.Tensor) -> torch.Tensor:
        if self.spec.depth != 2 or self.layer2_references is None or self.c2 is None:
            raise RuntimeError("Second layer requested from a depth-one model.")
        incoming = self._incoming_big_reference(layer1_state)
        state = self._uniform(layer1_state, self.A2 + [self.c2])
        references = self.layer2_references.all_states(dtype=state.dtype, device=state.device)
        small_targets = self.A1 + [self.c1]
        big_targets = self.W + self.A1 + [self.c1]
        for _ in range(self.spec.degree):
            # I_W tensor SELECT_j R_theta_j^(2), controlled on A2=j,c2=1.
            for branch, reference in enumerate(references):
                state = conditioned_reflection(
                    state,
                    n_qubits=self.n_total,
                    targets=small_targets,
                    reference=reference,
                    controls=self.A2 + [self.c2],
                    control_bits=bits_of_int(branch, 3) + [1],
                )
            # Global reflection about the whole retained layer-one state.
            state = conditioned_reflection(
                state,
                n_qubits=self.n_total,
                targets=big_targets,
                reference=incoming,
                controls=[self.c2],
                control_bits=[1],
            )
        if self.phase2 is not None:
            state = apply_one_qubit_gate(
                state,
                self.n_total,
                self.c2,
                rz(self.phase2, dtype=state.dtype, device=state.device),
            )
        return apply_one_qubit_gate(
            state, self.n_total, self.c2, hadamard(dtype=state.dtype, device=state.device)
        )

    @staticmethod
    def probabilities_to_responses(probabilities: torch.Tensor) -> torch.Tensor:
        # Uniform address weights w_j=1/8 imply alpha_j=(p(j,0)-p(j,1))/w_j.
        paired = probabilities.view(probabilities.size(0), 8, 2)
        return 8.0 * (paired[:, :, 0] - paired[:, :, 1])

    def clean_output_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        state1 = self.first_layer_state(x)
        if self.spec.depth == 1:
            return marginal_probabilities(state1, self.n_total, self.A1 + [self.c1])
        state2 = self._layer2(state1)
        assert self.c2 is not None
        return marginal_probabilities(state2, self.n_total, self.A2 + [self.c2])

    def forward(
        self,
        x: torch.Tensor,
        *,
        shots: int = 0,
        return_details: bool = False,
    ) -> Any:
        clean_probability = self.clean_output_probabilities(x)
        clean_response = self.probabilities_to_responses(clean_probability)
        if int(shots) > 0:
            noisy_probability = multinomial_probabilities(clean_probability, int(shots))
            noisy_response = self.probabilities_to_responses(noisy_probability)
            response = straight_through(noisy_response, clean_response)
        else:
            noisy_probability = None
            response = clean_response
        logits = self.beta * response.to(self.beta.dtype) + self.bias
        if return_details:
            return {
                "logits": logits,
                "clean_output_probabilities": clean_probability,
                "shot_output_probabilities": noisy_probability,
                "clean_responses": clean_response,
                "effective_responses": response,
            }
        return logits


# =============================================================================
# Training recipes and metrics
# =============================================================================


@dataclass(frozen=True)
class Recipe:
    depth: int = 1
    degree: int = 2
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    label_smoothing: float = 0.02
    entropy_weight: float = 0.0


def default_recipe() -> Recipe:
    return Recipe()


def recipe_from_mapping(values: Mapping[str, Any]) -> Recipe:
    defaults = asdict(default_recipe())
    for key in defaults:
        if key in values:
            defaults[key] = values[key]
    return Recipe(
        depth=int(defaults["depth"]),
        degree=int(defaults["degree"]),
        optimizer=str(defaults["optimizer"]),
        scheduler=str(defaults["scheduler"]),
        learning_rate=float(defaults["learning_rate"]),
        weight_decay=float(defaults["weight_decay"]),
        batch_size=int(defaults["batch_size"]),
        label_smoothing=float(defaults["label_smoothing"]),
        entropy_weight=float(defaults["entropy_weight"]),
    )


def make_model(recipe: Recipe, args: argparse.Namespace, device: torch.device) -> CoherentQFNN:
    model = CoherentQFNN(
        ModelSpec(
            depth=int(recipe.depth),
            degree=int(recipe.degree),
            ansatz_depth=int(args.reference_ansatz_depth),
            use_phase=bool(args.use_phase),
        )
    )
    with torch.no_grad():
        model.beta.fill_(float(args.beta_init))
        model.bias.zero_()
    return model.to(device)


def make_optimizer_and_scheduler(
    model: nn.Module,
    recipe: Recipe,
    *,
    epochs: int,
    steps_per_epoch: int,
) -> Tuple[optim.Optimizer, Optional[Any], bool]:
    if recipe.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(), lr=recipe.learning_rate, weight_decay=recipe.weight_decay
        )
    elif recipe.optimizer == "radam":
        optimizer = optim.RAdam(
            model.parameters(), lr=recipe.learning_rate, weight_decay=recipe.weight_decay
        )
    elif recipe.optimizer == "rmsprop":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=recipe.learning_rate,
            weight_decay=recipe.weight_decay,
            alpha=0.99,
            momentum=0.0,
        )
    else:
        raise ValueError(f"Unknown optimizer: {recipe.optimizer}")

    if recipe.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(epochs)), eta_min=0.01 * recipe.learning_rate
        )
        per_batch = False
    elif recipe.scheduler == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=recipe.learning_rate,
            epochs=max(1, int(epochs)),
            steps_per_epoch=max(1, int(steps_per_epoch)),
            pct_start=0.3,
            div_factor=15.0,
            final_div_factor=1000.0,
        )
        per_batch = True
    elif recipe.scheduler == "none":
        scheduler = None
        per_batch = False
    else:
        raise ValueError(f"Unknown scheduler: {recipe.scheduler}")
    return optimizer, scheduler, per_batch


def training_loss(logits: torch.Tensor, labels: torch.Tensor, recipe: Recipe) -> torch.Tensor:
    loss = F.cross_entropy(logits, labels, label_smoothing=float(recipe.label_smoothing))
    if recipe.entropy_weight > 0.0:
        probabilities = F.softmax(logits, dim=1)
        entropy = -(
            probabilities * torch.log(probabilities.clamp_min(1e-12))
        ).sum(dim=1).mean()
        # Positive weight penalizes high predictive entropy.
        loss = loss + float(recipe.entropy_weight) * entropy
    return loss


@torch.no_grad()
def evaluate_loader(
    model: CoherentQFNN,
    loader: DataLoader,
    *,
    device: torch.device,
    recipe: Recipe,
    shots: int = 0,
) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    total_ce = 0.0
    total_objective = 0.0
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(features, shots=int(shots))
        batch = int(labels.numel())
        total += batch
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_ce += float(F.cross_entropy(logits, labels, reduction="sum").item())
        total_objective += float(training_loss(logits, labels, recipe).item()) * batch
    return {
        "accuracy": 100.0 * correct / max(1, total),
        "cross_entropy": total_ce / max(1, total),
        "objective_loss": total_objective / max(1, total),
        "n": int(total),
        "shots": int(shots),
    }


def train_with_validation(
    *,
    bundle: DataBundle,
    recipe: Recipe,
    seed: int,
    epochs: int,
    args: argparse.Namespace,
    device: torch.device,
    run_dir: Path,
    save_checkpoint: bool,
    verbose_prefix: str,
) -> Dict[str, Any]:
    """Train without ever receiving or evaluating a test loader."""
    set_seed(seed, deterministic=not args.allow_nondeterminism)
    train_per_class = int(bundle.metadata.get("train_per_class_cap", -1))
    provenance = training_run_provenance(
        args=args,
        device=device,
        dataset=bundle.dataset,
        recipe=recipe,
        seed=int(seed),
        epochs=int(epochs),
        train_per_class=train_per_class,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.csv"
    if history_path.exists():
        history_path.unlink()

    train_loader = make_loader(
        bundle.train,
        batch_size=recipe.batch_size,
        shuffle=True,
        seed=seed,
        workers=args.workers,
        device=device,
    )
    val_loader = make_loader(
        bundle.val,
        batch_size=args.eval_batch_size,
        shuffle=False,
        seed=args.split_seed,
        workers=args.workers,
        device=device,
    )
    model = make_model(recipe, args, device)
    optimizer, scheduler, scheduler_per_batch = make_optimizer_and_scheduler(
        model, recipe, epochs=epochs, steps_per_epoch=len(train_loader)
    )

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    best_val_accuracy = -math.inf
    best_val_ce = math.inf
    history: List[Dict[str, Any]] = []
    start_time = time.perf_counter()

    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_total = 0
        train_correct = 0
        train_loss_sum = 0.0
        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, shots=int(args.train_shots))
            loss = training_loss(logits, labels, recipe)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.gradient_clip))
            optimizer.step()
            if scheduler is not None and scheduler_per_batch:
                scheduler.step()
            batch = int(labels.numel())
            train_total += batch
            train_correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
            train_loss_sum += float(loss.detach().item()) * batch

        if scheduler is not None and not scheduler_per_batch:
            scheduler.step()

        validation = evaluate_loader(
            model, val_loader, device=device, recipe=recipe, shots=0
        )
        row = {
            "epoch": int(epoch),
            "train_accuracy": 100.0 * train_correct / max(1, train_total),
            "train_objective_loss": train_loss_sum / max(1, train_total),
            "validation_accuracy": float(validation["accuracy"]),
            "validation_cross_entropy": float(validation["cross_entropy"]),
            "validation_objective_loss": float(validation["objective_loss"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        append_csv(history_path, row)

        improved = validation["accuracy"] > best_val_accuracy + 1e-12
        tied_better_loss = (
            abs(validation["accuracy"] - best_val_accuracy) <= 1e-12
            and validation["cross_entropy"] < best_val_ce
        )
        if improved or tied_better_loss:
            best_val_accuracy = float(validation["accuracy"])
            best_val_ce = float(validation["cross_entropy"])
            best_epoch = int(epoch)
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

        print(
            f"[{verbose_prefix}] epoch {epoch:03d}/{epochs:03d} "
            f"train={row['train_accuracy']:.2f}% "
            f"val={row['validation_accuracy']:.2f}% "
            f"best={best_val_accuracy:.2f}%@{best_epoch}"
        )

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint.")
    elapsed = time.perf_counter() - start_time
    checkpoint_path = run_dir / "best_checkpoint.pt"
    if save_checkpoint:
        atomic_torch_save(
            checkpoint_path,
            {
                "model_state": best_state,
                "recipe": asdict(recipe),
                "model_spec": asdict(model.spec),
                "best_epoch": best_epoch,
                "best_validation_accuracy": best_val_accuracy,
                "seed": int(seed),
                "provenance": provenance,
            },
        )

    result = {
        "dataset": bundle.dataset,
        "seed": int(seed),
        "epochs": int(epochs),
        "recipe": asdict(recipe),
        "model_spec": asdict(model.spec),
        "train_shots": int(args.train_shots),
        "parameter_count": int(model.trainable_parameter_count),
        "best_epoch": int(best_epoch),
        "best_validation_accuracy": float(best_val_accuracy),
        "best_validation_cross_entropy": float(best_val_ce),
        "wall_seconds_train_and_validation": float(elapsed),
        "data": bundle.metadata,
        "history": history,
        "checkpoint_path": str(checkpoint_path) if save_checkpoint else None,
        "provenance": provenance,
    }
    atomic_write_json(run_dir / "train_result.json", result)
    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Return the state only to short search runs, which do not write checkpoints.
    if not save_checkpoint:
        result["_best_state"] = best_state
    return result


def load_checkpoint_model(
    checkpoint_path: Path,
    recipe: Recipe,
    args: argparse.Namespace,
    device: torch.device,
) -> CoherentQFNN:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stored_provenance = payload.get("provenance")
    if not isinstance(stored_provenance, Mapping):
        raise ValueError(f"Checkpoint has no provenance record: {checkpoint_path}")
    stored_without_fingerprint = dict(stored_provenance)
    stored_fingerprint = stored_without_fingerprint.pop("training_fingerprint_sha256", None)
    recomputed_fingerprint = stable_hash(stored_without_fingerprint, length=64)
    if stored_fingerprint != recomputed_fingerprint:
        raise ValueError(
            f"Checkpoint provenance fingerprint is invalid: {checkpoint_path}"
        )
    current_core = core_provenance(args, device)
    checked_keys = (
        "experiment_version",
        "source_sha256",
        "torch_version",
        "device",
        "allow_nondeterminism",
        "gradient_clip",
        "train_shots",
        "reference_ansatz_depth",
        "use_phase",
        "beta_init",
    )
    mismatches = {
        key: {"stored": stored_provenance.get(key), "current": current_core.get(key)}
        for key in checked_keys
        if stored_provenance.get(key) != current_core.get(key)
    }
    if payload.get("recipe") != asdict(recipe):
        mismatches["recipe"] = {
            "stored": payload.get("recipe"),
            "current": asdict(recipe),
        }
    if mismatches:
        raise ValueError(
            f"Checkpoint provenance does not match the current experiment: {mismatches}"
        )
    model = make_model(recipe, args, device)
    if payload.get("model_spec") != asdict(model.spec):
        raise ValueError(
            f"Checkpoint model specification mismatch: stored={payload.get('model_spec')}, "
            f"current={asdict(model.spec)}"
        )
    model.load_state_dict(payload["model_state"])
    return model


# =============================================================================
# Per-dataset Optuna search (validation only)
# =============================================================================


def suggest_recipe(trial: Any, args: argparse.Namespace) -> Recipe:
    optimizer_name = trial.suggest_categorical("optimizer", list(args.search_optimizers))
    scheduler_name = trial.suggest_categorical("scheduler", list(args.search_schedulers))
    if optimizer_name == "rmsprop":
        learning_rate = trial.suggest_float("learning_rate", 2e-4, 2e-2, log=True)
    else:
        learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True)
    return Recipe(
        depth=int(trial.suggest_categorical("depth", list(args.depths))),
        degree=int(trial.suggest_categorical("degree", list(args.degrees))),
        optimizer=str(optimizer_name),
        scheduler=str(scheduler_name),
        learning_rate=float(learning_rate),
        weight_decay=float(trial.suggest_float("weight_decay", 1e-7, 2e-3, log=True)),
        batch_size=int(trial.suggest_categorical("batch_size", list(args.search_batch_sizes))),
        label_smoothing=float(
            trial.suggest_categorical("label_smoothing", [0.0, 0.02, 0.05, 0.10])
        ),
        entropy_weight=float(
            trial.suggest_categorical("entropy_weight", [0.0, 1e-4, 3e-4, 1e-3, 3e-3])
        ),
    )


def _study_storage(path: Path) -> str:
    return "sqlite:///" + path.resolve().as_posix()


def search_one_dataset(
    dataset: str,
    *,
    args: argparse.Namespace,
    device: torch.device,
    outdir: Path,
) -> Dict[str, Any]:
    if optuna is None:
        raise RuntimeError("Optuna is required for --stage search/all. Install it with: pip install optuna")

    print(f"\n[search] Preparing {dataset}; the official test partition is not loaded.")
    bundle = prepare_train_val(
        dataset,
        root=Path(args.data_root),
        download=not args.no_download,
        split_seed=args.split_seed,
        val_fraction=args.val_fraction,
        val_per_class=args.val_per_class,
        train_per_class=args.search_max_per_class,
    )
    sampler_config = {
        "name": SAMPLER_NAME,
        "optuna_version": getattr(optuna, "__version__", None),
        "seed": int(args.search_seed),
        "n_startup_trials": min(5, max(2, int(args.trials) // 3)),
        "multivariate": SAMPLER_MULTIVARIATE,
    }
    search_signature = {
        "scope": "substantial validation search; not exhaustive",
        "provenance": core_provenance(args, device),
        "dataset": dataset,
        "classes": list(CLASSES),
        "data_root": str(Path(args.data_root).resolve()),
        "split_seed": int(args.split_seed),
        "val_fraction": float(args.val_fraction),
        "val_per_class": int(args.val_per_class),
        "search_max_per_class": int(args.search_max_per_class),
        "epochs": int(args.search_epochs),
        "target_total_trials": int(args.trials),
        "depths": list(args.depths),
        "degrees": list(args.degrees),
        "search_seeds": list(args.search_seeds),
        "search_optimizers": list(args.search_optimizers),
        "search_schedulers": list(args.search_schedulers),
        "search_batch_sizes": list(args.search_batch_sizes),
        "sampler": sampler_config,
        "architecture_pair_coverage": "one enqueued trial per depth-degree pair when budget permits",
        "objective": "mean best exact-validation accuracy across search seeds",
    }
    signature = stable_hash(search_signature, 16)
    suffix_token = safe_suffix_token(args.study_suffix)
    artifact_key = signature if not suffix_token else f"{signature}__{suffix_token}"
    study_dir = outdir / "studies"
    study_dir.mkdir(parents=True, exist_ok=True)
    db_path = study_dir / f"{dataset_label(dataset)}_{artifact_key}.db"
    study_name = f"qfnn_{dataset_label(dataset)}_{artifact_key}"
    sampler = optuna.samplers.TPESampler(
        seed=sampler_config["seed"],
        n_startup_trials=sampler_config["n_startup_trials"],
        multivariate=sampler_config["multivariate"],
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=_study_storage(db_path),
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )
    architecture_pairs = [
        (int(depth), int(degree)) for depth in args.depths for degree in args.degrees
    ]
    if not study.trials and int(args.trials) >= len(architecture_pairs):
        # Guarantee that the modest default budget evaluates every requested
        # depth-degree pair at least once.  Remaining hyperparameters are still
        # sampled by TPE for these enqueued trials.
        for depth, degree in architecture_pairs:
            study.enqueue_trial({"depth": depth, "degree": degree})
    elif int(args.trials) < len(architecture_pairs):
        print(
            f"[search/{dataset_label(dataset)}] warning: trials={args.trials} is below "
            f"the {len(architecture_pairs)} requested depth-degree pairs; full pair coverage "
            "is not possible. Use --stage ablation for a guaranteed 2x3 diagnostic."
        )
    search_root = outdir / "search" / dataset_label(dataset) / artifact_key
    search_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        search_root / "study_config.json",
        {
            "artifact_key": artifact_key,
            "study_name": study_name,
            "study_suffix_raw": args.study_suffix,
            "search_signature": search_signature,
        },
    )

    def objective(trial: Any) -> float:
        recipe = suggest_recipe(trial, args)
        seed_results = []
        for seed in args.search_seeds:
            trial_dir = search_root / f"trial_{trial.number:04d}" / f"seed_{int(seed):04d}"
            try:
                run = train_with_validation(
                    bundle=bundle,
                    recipe=recipe,
                    seed=int(seed),
                    epochs=int(args.search_epochs),
                    args=args,
                    device=device,
                    run_dir=trial_dir,
                    save_checkpoint=False,
                    verbose_prefix=f"search/{dataset_label(dataset)}/t{trial.number}/s{seed}",
                )
            except torch.cuda.OutOfMemoryError as error:
                trial.set_user_attr("cuda_out_of_memory", True)
                trial.set_user_attr("cuda_oom_batch_size", int(recipe.batch_size))
                gc.collect()
                torch.cuda.empty_cache()
                raise optuna.TrialPruned(
                    "CUDA out of memory in the literal statevector simulator; "
                    "the trial was pruned. Reduce --search-batch-sizes if this recurs."
                ) from error
            seed_results.append(run)
        scores = [float(run["best_validation_accuracy"]) for run in seed_results]
        objective_value = float(np.mean(scores))
        trial.set_user_attr("recipe", asdict(recipe))
        trial.set_user_attr("validation_accuracy_by_seed", scores)
        trial.set_user_attr("parameter_count", int(seed_results[0]["parameter_count"]))
        trial.set_user_attr("train_shots_fixed", int(args.train_shots))
        trial.set_user_attr("experiment_version", EXPERIMENT_VERSION)
        trial.set_user_attr("source_sha256", source_sha256())
        artifact = {
            "trial": int(trial.number),
            "objective_mean_best_validation_accuracy": objective_value,
            "recipe": asdict(recipe),
            "validation_accuracy_by_seed": scores,
            "parameter_count": int(seed_results[0]["parameter_count"]),
            "test_data_used": False,
            "search_scope": "substantial validation search; not exhaustive",
            "provenance": core_provenance(args, device),
        }
        atomic_write_json(search_root / f"trial_{trial.number:04d}" / "trial_result.json", artifact)
        for run in seed_results:
            run.pop("_best_state", None)
        return objective_value

    completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = max(0, int(args.trials) - completed)
    print(
        f"[search/{dataset_label(dataset)}] study has {completed} completed trial(s); "
        f"target={args.trials}, running {remaining}."
    )
    if remaining:
        # Do not silently convert implementation/data errors into failed trials.
        # Optuna persists completed trials, so a corrected run can resume.
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    if not study.best_trial.user_attrs.get("recipe"):
        raise RuntimeError("The best completed Optuna trial has no stored recipe.")
    recipe = recipe_from_mapping(study.best_trial.user_attrs["recipe"])

    trial_rows = []
    for trial in study.trials:
        trial_rows.append(
            {
                "number": int(trial.number),
                "state": str(trial.state.name),
                "value": trial.value,
                **{f"param_{k}": v for k, v in trial.params.items()},
                "validation_accuracy_by_seed": json.dumps(
                    trial.user_attrs.get("validation_accuracy_by_seed", [])
                ),
                "parameter_count": trial.user_attrs.get("parameter_count"),
            }
        )
    write_csv(search_root / "optuna_trials.csv", trial_rows)
    result = {
        "dataset": dataset,
        "study_name": study.study_name,
        "storage": str(db_path),
        "n_completed_trials": sum(
            t.state == optuna.trial.TrialState.COMPLETE for t in study.trials
        ),
        "best_trial_number": int(study.best_trial.number),
        "best_validation_objective": float(study.best_value),
        "recipe": asdict(recipe),
        "train_shots_fixed_not_tuned": int(args.train_shots),
        "test_data_used": False,
        "search_signature": search_signature,
        "search_artifact_key": artifact_key,
        "search_artifact_root": str(search_root),
        "search_scope": "substantial validation search; not exhaustive",
    }
    atomic_write_json(search_root / "selected_recipe.json", result)
    return result


def save_selected_recipes(outdir: Path, selected: Mapping[str, Mapping[str, Any]]) -> None:
    atomic_write_json(outdir / "selected_recipes.json", selected)
    rows = []
    for dataset, record in selected.items():
        recipe = recipe_from_mapping(record["recipe"])
        rows.append(
            [
                dataset_label(dataset),
                record.get("best_trial_number", "manual"),
                f"{float(record.get('best_validation_objective', float('nan'))):.3f}",
                recipe.depth,
                recipe.degree,
                recipe.optimizer,
                recipe.scheduler,
                f"{recipe.learning_rate:.3g}",
                f"{recipe.weight_decay:.3g}",
                recipe.batch_size,
                recipe.label_smoothing,
                recipe.entropy_weight,
            ]
        )
    text = "# Selected per-dataset QFNN recipes\n\n"
    text += (
        "Selection used validation splits from official training data; no test metrics were used. "
        "The default sixty-trial, two-seed study is a substantial validation search, "
        "but it is not an exhaustive architecture or optimizer search.\n\n"
    )
    text += markdown_table(
        [
            "Dataset",
            "Trial",
            "Validation objective (%)",
            "Depth",
            "Degree",
            "Optimizer",
            "Scheduler",
            "LR",
            "Weight decay",
            "Batch",
            "Label smoothing",
            "Entropy penalty",
        ],
        rows,
    )
    (outdir / "selected_recipes.md").write_text(text, encoding="utf-8")


def load_selected_recipes(outdir: Path) -> Dict[str, Dict[str, Any]]:
    path = outdir / "selected_recipes.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run --stage search first, or use --stage all."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(k): dict(v) for k, v in payload.items()}


# =============================================================================
# Final six-seed runs, delayed test evaluation, and plots
# =============================================================================


def final_run_directory(
    outdir: Path,
    dataset: str,
    seed: int,
    recipe: Recipe,
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    fingerprint = {
        "training": training_run_provenance(
            args=args,
            device=device,
            dataset=dataset,
            recipe=recipe,
            seed=int(seed),
            epochs=int(args.final_epochs),
            train_per_class=int(args.final_max_per_class),
        ),
        "final_seed_set": list(args.final_seeds),
        "test_max_per_class": int(args.test_max_per_class),
        "eval_shots": int(args.eval_shots),
        "shot_repeats": int(args.shot_repeats),
    }
    signature = stable_hash(fingerprint, length=16)
    return outdir / "final" / dataset_label(dataset) / f"seed_{int(seed):04d}" / signature


def train_final_checkpoints(
    dataset: str,
    *,
    recipe: Recipe,
    args: argparse.Namespace,
    device: torch.device,
    outdir: Path,
) -> List[Dict[str, Any]]:
    print(
        f"\n[final/{dataset_label(dataset)}] Training and selecting all checkpoints "
        "before the official test partition is loaded."
    )
    bundle = prepare_train_val(
        dataset,
        root=Path(args.data_root),
        download=not args.no_download,
        split_seed=args.split_seed,
        val_fraction=args.val_fraction,
        val_per_class=args.val_per_class,
        train_per_class=args.final_max_per_class,
    )
    results: List[Dict[str, Any]] = []
    for seed in args.final_seeds:
        run_dir = final_run_directory(outdir, dataset, int(seed), recipe, args, device)
        cached = run_dir / "train_result.json"
        checkpoint = run_dir / "best_checkpoint.pt"
        expected_provenance = training_run_provenance(
            args=args,
            device=device,
            dataset=dataset,
            recipe=recipe,
            seed=int(seed),
            epochs=int(args.final_epochs),
            train_per_class=int(args.final_max_per_class),
        )
        cache_valid = False
        if cached.exists() and checkpoint.exists() and not args.force:
            with cached.open("r", encoding="utf-8") as handle:
                result = json.load(handle)
            cache_valid = result.get("provenance") == expected_provenance
            if cache_valid:
                print(f"[resume/{dataset_label(dataset)}/seed={seed}] using provenance-matched checkpoint.")
            else:
                print(f"[resume/{dataset_label(dataset)}/seed={seed}] provenance mismatch; retraining.")
        if not cache_valid:
            try:
                result = train_with_validation(
                    bundle=bundle,
                    recipe=recipe,
                    seed=int(seed),
                    epochs=int(args.final_epochs),
                    args=args,
                    device=device,
                    run_dir=run_dir,
                    save_checkpoint=True,
                    verbose_prefix=f"final/{dataset_label(dataset)}/s{seed}",
                )
            except torch.cuda.OutOfMemoryError as error:
                gc.collect()
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"CUDA out of memory during final {dataset_label(dataset)} seed {seed}. "
                    "The selected batch size is part of the recipe; rerun the search with "
                    "smaller --search-batch-sizes rather than silently changing the final recipe."
                ) from error
        if not Path(result["checkpoint_path"]).exists():
            raise FileNotFoundError(f"Missing selected checkpoint: {result['checkpoint_path']}")
        results.append(result)
    return results


def evaluate_selected_checkpoints_on_test(
    dataset: str,
    *,
    recipe: Recipe,
    train_results: List[Dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> List[Dict[str, Any]]:
    # This is deliberately the first test-data call in the final workflow.
    print(
        f"[test/{dataset_label(dataset)}] All {len(train_results)} checkpoint(s) are fixed. "
        "Loading official test data now."
    )
    test_set, test_meta = prepare_official_test(
        dataset,
        root=Path(args.data_root),
        download=not args.no_download,
        max_per_class=int(args.test_max_per_class),
        selection_seed=int(args.split_seed),
    )
    completed: List[Dict[str, Any]] = []
    for train_result in train_results:
        run_dir = Path(train_result["checkpoint_path"]).parent
        result_path = run_dir / "result.json"
        if result_path.exists() and not args.force:
            with result_path.open("r", encoding="utf-8") as handle:
                cached_result = json.load(handle)
            cache_matches = (
                int(cached_result.get("eval_shots", -1)) == int(args.eval_shots)
                and int(cached_result.get("shot_repeats_count", -1)) == int(args.shot_repeats)
                and int(cached_result.get("test_metadata", {}).get("test_per_class_cap", -1))
                == int(args.test_max_per_class)
            )
            if cache_matches:
                completed.append(cached_result)
                continue

        seed = int(train_result["seed"])
        set_seed(seed + 1_000_003, deterministic=not args.allow_nondeterminism)
        loader = make_loader(
            test_set,
            batch_size=int(args.eval_batch_size),
            shuffle=False,
            seed=int(args.split_seed),
            workers=int(args.workers),
            device=device,
        )
        model = load_checkpoint_model(
            Path(train_result["checkpoint_path"]), recipe, args, device
        )
        exact = evaluate_loader(model, loader, device=device, recipe=recipe, shots=0)
        shot_repeats = []
        for repeat in range(int(args.shot_repeats)):
            # The model and batch order stay fixed; only multinomial draws vary.
            set_seed(seed * 1_000_003 + repeat + 17, deterministic=not args.allow_nondeterminism)
            shot_repeats.append(
                evaluate_loader(
                    model,
                    loader,
                    device=device,
                    recipe=recipe,
                    shots=int(args.eval_shots),
                )
            )
        result = copy.deepcopy(train_result)
        result.update(
            {
                "test_evaluated_after_checkpoint_selection": True,
                "test_metadata": test_meta,
                "test_exact": exact,
                "test_shot_repeats": shot_repeats,
                "test_shot_accuracy_summary": sample_mean_std(
                    [entry["accuracy"] for entry in shot_repeats]
                ),
                "eval_shots": int(args.eval_shots),
                "shot_repeats_count": int(args.shot_repeats),
            }
        )
        atomic_write_json(result_path, result)
        completed.append(result)
        print(
            f"[test/{dataset_label(dataset)}/seed={seed}] "
            f"exact={exact['accuracy']:.2f}%; "
            f"shots={result['test_shot_accuracy_summary']['mean']:.2f}% "
            f"+/- {result['test_shot_accuracy_summary']['std']:.2f}%"
        )
        del model, loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return completed


def aggregate_curves(results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if not results:
        return []
    histories = [result["history"] for result in results]
    epoch_count = min(len(history) for history in histories)
    train_shots = int(results[0]["train_shots"])
    train_mode = (
        f"finite-shot multinomial N={train_shots}" if train_shots > 0 else "exact statevector"
    )
    rows: List[Dict[str, Any]] = []
    for index in range(epoch_count):
        train = [float(history[index]["train_accuracy"]) for history in histories]
        validation = [float(history[index]["validation_accuracy"]) for history in histories]
        train_summary = sample_mean_std(train)
        val_summary = sample_mean_std(validation)
        rows.append(
            {
                "epoch": float(histories[0][index]["epoch"]),
                "train_accuracy_mean": train_summary["mean"],
                "train_accuracy_std": train_summary["std"],
                "validation_accuracy_mean": val_summary["mean"],
                "validation_accuracy_std": val_summary["std"],
                "train_accuracy_mode": train_mode,
                "validation_accuracy_mode": "exact statevector",
            }
        )
    return rows


def plot_accuracy_curves(
    dataset: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    reports_dir: Path,
    seed_count: int,
    train_shots: int,
) -> None:
    if not rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] matplotlib unavailable; CSV curve retained ({exc}).")
        return
    epochs = np.asarray([row["epoch"] for row in rows], dtype=np.float64)
    train_mean = np.asarray([row["train_accuracy_mean"] for row in rows])
    train_std = np.asarray([row["train_accuracy_std"] for row in rows])
    val_mean = np.asarray([row["validation_accuracy_mean"] for row in rows])
    val_std = np.asarray([row["validation_accuracy_std"] for row in rows])

    fig, axis = plt.subplots(figsize=(12, 6.2), constrained_layout=True)
    train_label = (
        f"QFNN train (multinomial, N={int(train_shots)})"
        if int(train_shots) > 0
        else "QFNN train (exact)"
    )
    axis.plot(epochs, train_mean, color="#1f77b4", linewidth=2.0, label=train_label)
    axis.fill_between(
        epochs, train_mean - train_std, train_mean + train_std, color="#1f77b4", alpha=0.18
    )
    axis.plot(
        epochs,
        val_mean,
        color="#d62728",
        linewidth=2.0,
        linestyle="--",
        label="QFNN validation (exact)",
    )
    axis.fill_between(
        epochs, val_mean - val_std, val_mean + val_std, color="#d62728", alpha=0.18
    )
    axis.set_title(
        f"{dataset_label(dataset)} QFNN train/validation accuracy "
        f"(mean +/- sample std over {seed_count} seeds)"
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy (%)")
    axis.grid(True, alpha=0.28)
    axis.legend(loc="best")
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = reports_dir / f"{dataset_label(dataset)}_accuracy"
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def abstract_resource_accounting(recipe: Recipe) -> Dict[str, Any]:
    """High-level oracle counts; deliberately not compiled gate estimates."""
    layer_count = int(recipe.depth)
    degree = int(recipe.degree)
    address_qubits_per_layer = 3
    ancillas_per_layer = 1
    logical_qubits = 6 + layer_count * (address_qubits_per_layer + ancillas_per_layer)
    # Each Chebyshev step contains one multiplexed reference SELECT reflection
    # and one current-state reflection.  The simulator resolves SELECT into
    # eight mutually exclusive controlled branches, hence the additional count.
    abstract_blocks = 2 * degree * layer_count
    branch_resolved_blocks = (8 + 1) * degree * layer_count
    base_calls = (1 + 2 * degree) ** layer_count
    return {
        "accounting_scope": (
            "abstract oracle level only; not a compiled gate-count, two-qubit-depth, "
            "shot-cost, or wall-time equivalence"
        ),
        "logical_qubits": int(logical_qubits),
        "abstract_controlled_reflection_blocks_per_forward": int(abstract_blocks),
        "abstract_block_convention": (
            "one branch-multiplexed reference SELECT plus one current-state reflection "
            "per Chebyshev step per layer"
        ),
        "branch_resolved_conditioned_reflections_in_this_simulator_per_forward": int(
            branch_resolved_blocks
        ),
        "exact_prep_unprep_base_data_oracle_calls_per_forward": int(base_calls),
        "base_data_oracle_call_formula": "(1 + 2*degree)**depth",
        "simulator_note": (
            "the statevector code applies reflections algebraically; the prep/unprep count "
            "is the exact recursive-oracle count under the stated implementation, not a "
            "count inferred from simulator wall time"
        ),
    }


def summarize_dataset_final(
    dataset: str,
    recipe: Recipe,
    results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    exact = [float(result["test_exact"]["accuracy"]) for result in results]
    seed_shot_means = [
        float(result["test_shot_accuracy_summary"]["mean"]) for result in results
    ]
    all_shot_repeats = [
        float(repeat["accuracy"])
        for result in results
        for repeat in result["test_shot_repeats"]
    ]
    validation = [float(result["best_validation_accuracy"]) for result in results]
    wall_seconds = [
        float(result["wall_seconds_train_and_validation"]) for result in results
    ]
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "source_sha256": source_sha256(),
        "dataset": dataset,
        "recipe": asdict(recipe),
        "n_seeds": len(results),
        "seeds": [int(result["seed"]) for result in results],
        "parameter_count": int(results[0]["parameter_count"]),
        "best_validation_accuracy_across_seeds": sample_mean_std(validation),
        "test_exact_accuracy_across_seeds": sample_mean_std(exact),
        "test_shot_accuracy_seed_means": sample_mean_std(seed_shot_means),
        "test_shot_accuracy_all_repeats": sample_mean_std(all_shot_repeats),
        "eval_shots": int(results[0]["eval_shots"]),
        "shot_repeats_per_seed": int(results[0]["shot_repeats_count"]),
        "training_and_validation_wall_seconds_across_seeds": sample_mean_std(wall_seconds),
        "resource_accounting": abstract_resource_accounting(recipe),
        "runs": list(results),
    }


def write_final_reports(outdir: Path, summaries: Mapping[str, Mapping[str, Any]]) -> None:
    reports = outdir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    atomic_write_json(reports / "final_results.json", summaries)
    rows = []
    md_rows = []
    for dataset, summary in summaries.items():
        recipe = recipe_from_mapping(summary["recipe"])
        exact = summary["test_exact_accuracy_across_seeds"]
        shot = summary["test_shot_accuracy_seed_means"]
        resources = summary["resource_accounting"]
        wall = summary["training_and_validation_wall_seconds_across_seeds"]
        row = {
            "experiment_version": summary["experiment_version"],
            "source_sha256": summary["source_sha256"],
            "dataset": dataset_label(dataset),
            "depth": recipe.depth,
            "degree": recipe.degree,
            "optimizer": recipe.optimizer,
            "scheduler": recipe.scheduler,
            "learning_rate": recipe.learning_rate,
            "weight_decay": recipe.weight_decay,
            "batch_size": recipe.batch_size,
            "label_smoothing": recipe.label_smoothing,
            "entropy_weight": recipe.entropy_weight,
            "parameter_count": summary["parameter_count"],
            "seeds": json.dumps(summary["seeds"]),
            "test_exact_accuracy_mean": exact["mean"],
            "test_exact_accuracy_std_across_seeds": exact["std"],
            "test_shot_accuracy_mean": shot["mean"],
            "test_shot_accuracy_std_across_seeds": shot["std"],
            "eval_shots": summary["eval_shots"],
            "shot_repeats_per_seed": summary["shot_repeats_per_seed"],
            "logical_qubits": resources["logical_qubits"],
            "abstract_controlled_reflection_blocks_per_forward": resources[
                "abstract_controlled_reflection_blocks_per_forward"
            ],
            "branch_resolved_conditioned_reflections_simulator_per_forward": resources[
                "branch_resolved_conditioned_reflections_in_this_simulator_per_forward"
            ],
            "exact_prep_unprep_base_data_oracle_calls_per_forward": resources[
                "exact_prep_unprep_base_data_oracle_calls_per_forward"
            ],
            "resource_accounting_scope": resources["accounting_scope"],
            "training_and_validation_wall_seconds_mean": wall["mean"],
            "training_and_validation_wall_seconds_std_across_seeds": wall["std"],
        }
        rows.append(row)
        md_rows.append(
            [
                row["dataset"],
                f"{recipe.depth}/{recipe.degree}",
                summary["parameter_count"],
                f"{exact['mean']:.2f} +/- {exact['std']:.2f}",
                f"{shot['mean']:.2f} +/- {shot['std']:.2f}",
                summary["eval_shots"],
                summary["shot_repeats_per_seed"],
                resources["logical_qubits"],
                resources["abstract_controlled_reflection_blocks_per_forward"],
                resources["exact_prep_unprep_base_data_oracle_calls_per_forward"],
                f"{wall['mean']:.1f} +/- {wall['std']:.1f}",
            ]
        )
    write_csv(reports / "final_results.csv", rows)
    text = "# Final QFNN results\n\n"
    text += (
        "Recipes and checkpoints were selected using validation data from the official "
        "training partition. The official test partition was evaluated only after all "
        "per-seed checkpoints for a dataset were fixed. Reported +/- values across seeds "
        "are sample standard deviations.\n\n"
    )
    text += (
        "Resource columns are exact only at the stated abstract oracle level. They are not "
        "compiled gate counts, two-qubit depths, physical runtimes, or claims of a fair "
        "hardware-resource match. One abstract reflection pair means one branch-multiplexed "
        "reference SELECT and one current-state reflection. Base-data calls use "
        "$(1+2d)^L$ for uniform degree $d$ and depth $L$.\n\n"
    )
    text += markdown_table(
        [
            "Dataset",
            "Depth/degree",
            "Parameters",
            "Exact test accuracy (%)",
            "Finite-shot test accuracy (%)",
            "Shots",
            "Repeats/seed",
            "Logical qubits",
            "Abstract reflection blocks/forward",
            "Prep/unprep base-data calls/forward",
            "Train+validation wall time (s)",
        ],
        md_rows,
    )
    (reports / "final_results.md").write_text(text, encoding="utf-8")


def run_final(
    datasets: Sequence[str],
    *,
    selected: Mapping[str, Mapping[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    outdir: Path,
) -> Dict[str, Dict[str, Any]]:
    summaries: Dict[str, Dict[str, Any]] = {}
    for dataset in datasets:
        if dataset not in selected:
            raise KeyError(f"No selected recipe exists for {dataset}.")
        signature = selected[dataset].get("search_signature", {})
        selected_core = signature.get("provenance", {})
        expected_top_level = {
            "split_seed": int(args.split_seed),
            "val_fraction": float(args.val_fraction),
            "val_per_class": int(args.val_per_class),
        }
        mismatches = {
            key: {"selected": signature.get(key), "current": value}
            for key, value in expected_top_level.items()
            if key in signature and signature.get(key) != value
        }
        current_core = core_provenance(args, device)
        for key, value in current_core.items():
            if selected_core.get(key) != value:
                mismatches[f"provenance.{key}"] = {
                    "selected": selected_core.get(key),
                    "current": value,
                }
        if mismatches:
            raise ValueError(
                f"Final settings do not match the settings used to select {dataset}: "
                f"{mismatches}. Reuse the search settings or start a new study."
            )
        recipe = recipe_from_mapping(selected[dataset]["recipe"])
        train_results = train_final_checkpoints(
            dataset, recipe=recipe, args=args, device=device, outdir=outdir
        )
        completed = evaluate_selected_checkpoints_on_test(
            dataset,
            recipe=recipe,
            train_results=train_results,
            args=args,
            device=device,
        )
        summary = summarize_dataset_final(dataset, recipe, completed)
        summaries[dataset] = summary
        curve_rows = aggregate_curves(completed)
        reports = outdir / "reports"
        write_csv(reports / f"{dataset_label(dataset)}_curves.csv", curve_rows)
        plot_accuracy_curves(
            dataset,
            curve_rows,
            reports_dir=reports,
            seed_count=len(completed),
            train_shots=int(args.train_shots),
        )
        # Keep reports usable if a later dataset run is interrupted.
        write_final_reports(outdir, summaries)
    return summaries


# =============================================================================
# Optional compact FashionMNIST validation-only depth x degree ablation
# =============================================================================


def run_compact_ablation(
    *,
    args: argparse.Namespace,
    device: torch.device,
    outdir: Path,
    selected: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    dataset = "FashionMNIST"
    if selected is not None and dataset in selected:
        base = recipe_from_mapping(selected[dataset]["recipe"])
    else:
        base = default_recipe()
    print("\n[ablation] Compact FashionMNIST validation-only depth x degree grid.")
    bundle = prepare_train_val(
        dataset,
        root=Path(args.data_root),
        download=not args.no_download,
        split_seed=args.split_seed,
        val_fraction=args.val_fraction,
        val_per_class=args.val_per_class,
        train_per_class=args.ablation_max_per_class,
    )
    rows: List[Dict[str, Any]] = []
    for depth in (1, 2):
        for degree in (1, 2, 3):
            recipe = Recipe(**{**asdict(base), "depth": depth, "degree": degree})
            for seed in args.ablation_seeds:
                run_dir = (
                    outdir
                    / "ablation"
                    / dataset_label(dataset)
                    / f"depth{depth}_degree{degree}"
                    / f"seed_{int(seed):04d}"
                )
                run = train_with_validation(
                    bundle=bundle,
                    recipe=recipe,
                    seed=int(seed),
                    epochs=int(args.ablation_epochs),
                    args=args,
                    device=device,
                    run_dir=run_dir,
                    save_checkpoint=False,
                    verbose_prefix=f"ablation/d{depth}/D{degree}/s{seed}",
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "depth": depth,
                        "degree": degree,
                        "seed": int(seed),
                        "best_validation_accuracy": run["best_validation_accuracy"],
                        "best_epoch": run["best_epoch"],
                        "parameter_count": run["parameter_count"],
                        "test_data_used": False,
                    }
                )
                run.pop("_best_state", None)
    summaries = []
    for depth in (1, 2):
        for degree in (1, 2, 3):
            values = [
                float(row["best_validation_accuracy"])
                for row in rows
                if row["depth"] == depth and row["degree"] == degree
            ]
            stats = sample_mean_std(values)
            summaries.append(
                {
                    "depth": depth,
                    "degree": degree,
                    "validation_accuracy_mean": stats["mean"],
                    "validation_accuracy_std": stats["std"],
                    "n_seeds": stats["n"],
                    "test_data_used": False,
                }
            )
    payload = {
        "dataset": dataset,
        "purpose": "compact architectural diagnostic, validation only",
        "test_data_used": False,
        "base_recipe_except_depth_degree": asdict(base),
        "runs": rows,
        "summary": summaries,
    }
    ablation_dir = outdir / "ablation" / dataset_label(dataset)
    atomic_write_json(ablation_dir / "ablation_results.json", payload)
    write_csv(ablation_dir / "ablation_runs.csv", rows)
    write_csv(ablation_dir / "ablation_summary.csv", summaries)
    md_rows = [
        [
            row["depth"],
            row["degree"],
            f"{row['validation_accuracy_mean']:.2f} +/- {row['validation_accuracy_std']:.2f}",
            row["n_seeds"],
        ]
        for row in summaries
    ]
    (ablation_dir / "ablation_summary.md").write_text(
        "# Compact FashionMNIST depth-degree ablation\n\n"
        "This diagnostic uses validation data only; it does not inspect the official test set.\n\n"
        + markdown_table(["Depth", "Degree", "Best validation accuracy (%)", "Seeds"], md_rows),
        encoding="utf-8",
    )
    return payload


# =============================================================================
# No-download mathematical and autodiff checks
# =============================================================================


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, tolerance: float) -> None:
    error = float((actual - expected).abs().max().detach().cpu().item())
    print(f"[self-test] {name}: max error={error:.3e}")
    if not math.isfinite(error) or error > float(tolerance):
        raise AssertionError(f"{name} failed: max error {error} > {tolerance}")


def run_self_test() -> None:
    """Exercise exact layer laws, marginals, shot draws, and backward passes."""
    device = torch.device("cpu")
    set_seed(1234)
    x = torch.rand(2, 64, dtype=torch.float32, device=device)
    x = x / torch.linalg.vector_norm(x, dim=1, keepdim=True)

    try:
        amplitude_preprocess(np.zeros((1, 8, 8), dtype=np.float32))
    except ValueError as error:
        if "zero-norm" not in str(error):
            raise
        print("[self-test] zero-norm preprocessing rejection: PASS")
    else:
        raise AssertionError("Zero-norm preprocessing input was not rejected.")

    for degree in (1, 2, 3):
        model = CoherentQFNN(ModelSpec(depth=1, degree=degree, ansatz_depth=1)).to(device)
        details = model(x, shots=0, return_details=True)
        probabilities = details["clean_output_probabilities"]
        _assert_close(
            f"depth1 degree{degree} probability normalization",
            probabilities.sum(dim=1),
            torch.ones(2),
            2e-5,
        )
        references = model.layer1_references.all_states(
            dtype=torch.complex64, device=device
        )
        reference_matrix = torch.stack(references, dim=0)
        overlaps = torch.abs(x.to(torch.complex64) @ torch.conj(reference_matrix).T).square()
        expected = chebyshev_t(2.0 * overlaps - 1.0, degree)
        _assert_close(
            f"depth1 degree{degree} Chebyshev response",
            details["clean_responses"],
            expected,
            2e-4,
        )

    model2 = CoherentQFNN(ModelSpec(depth=2, degree=2, ansatz_depth=1)).to(device)
    try:
        model2(torch.zeros(1, 64), shots=0)
    except ValueError as error:
        if "zero-norm" not in str(error):
            raise
        print("[self-test] zero-norm model-input rejection: PASS")
    else:
        raise AssertionError("Zero-norm model input was not rejected.")
    state1 = model2.first_layer_state(x)
    details2 = model2(x, shots=0, return_details=True)
    assert model2.layer2_references is not None
    references2 = torch.stack(
        model2.layer2_references.all_states(dtype=torch.complex64, device=device), dim=0
    )
    incoming = model2._incoming_big_reference(state1).view(2, 64, 16)
    # alpha_j = ||(I_W tensor <theta_j|) |Phi^(1)>||^2.
    projected = torch.einsum("bws,js->bjw", incoming, torch.conj(references2))
    overlaps2 = projected.abs().square().sum(dim=2)
    expected2 = chebyshev_t(2.0 * overlaps2 - 1.0, 2)
    _assert_close(
        "depth2 local-projector Chebyshev response",
        details2["clean_responses"],
        expected2,
        3e-4,
    )
    _assert_close(
        "depth2 final marginal normalization",
        details2["clean_output_probabilities"].sum(dim=1),
        torch.ones(2),
        2e-5,
    )

    set_seed(9876)
    noisy = multinomial_probabilities(details2["clean_output_probabilities"], 257)
    _assert_close("multinomial row sums", noisy.sum(dim=1), torch.ones(2), 2e-6)
    labels = torch.tensor([0, 1], dtype=torch.long)
    logits = model2(x, shots=257)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    finite_gradients = [
        bool(torch.isfinite(parameter.grad).all())
        for parameter in model2.parameters()
        if parameter.grad is not None
    ]
    if not finite_gradients or not all(finite_gradients):
        raise AssertionError("Backward pass produced a missing/non-finite gradient set.")
    print(
        f"[self-test] backward: loss={float(loss.item()):.6f}, "
        f"finite gradient tensors={len(finite_gradients)}"
    )
    print("[self-test] PASS: no dataset was downloaded or loaded.")


# =============================================================================
# Command line
# =============================================================================


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Standalone QFNN-only Optuna and final-evaluation pipeline. "
            "Search/selection uses validation data only."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=["all", "search", "final", "ablation", "self-test"],
        default="search",
        help=(
            "Workflow stage. Search is the safe default; 'all' explicitly requests "
            "search plus the long final evaluation."
        ),
    )
    parser.add_argument("--outdir", default="qfnn_experiments")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(SUPPORTED_DATASETS),
        default=list(DEFAULT_DATASETS),
        help="CIFAR10 is supported but intentionally not in the default set.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--workers", type=int, default=0, help="Zero is safest on Windows.")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--allow-nondeterminism", action="store_true")
    parser.add_argument("--force", action="store_true", help="Redo cached final runs/results.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tiny debugging configuration; combine with --stage all for end-to-end smoke.",
    )

    # Fixed data protocol.
    parser.add_argument("--split-seed", type=int, default=314159)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--val-per-class", type=int, default=200)
    parser.add_argument("--search-max-per-class", type=int, default=300)
    parser.add_argument("--final-max-per-class", type=int, default=1000)
    parser.add_argument(
        "--test-max-per-class",
        type=int,
        default=0,
        help="0 uses every class-0..7 item in the official test partition.",
    )

    # Fixed simulator/architecture settings.  Shots and phase are not tuned.
    parser.add_argument("--train-shots", type=int, default=6000)
    parser.add_argument("--eval-shots", type=int, default=6000)
    parser.add_argument("--shot-repeats", type=int, default=5)
    parser.add_argument("--use-phase", action="store_true", help="Opt in to trainable per-layer RZ phases.")
    parser.add_argument("--reference-ansatz-depth", type=int, default=4)
    parser.add_argument("--beta-init", type=float, default=1.0)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--eval-batch-size", type=int, default=128)

    # One independent Optuna study per dataset. --trials is target total.
    parser.add_argument(
        "--trials",
        type=int,
        default=60,
        help="Target total trials per dataset; substantial by default, but not exhaustive.",
    )
    parser.add_argument(
        "--search-epochs",
        type=int,
        default=30,
        help="Epochs for each search-seed training inside every Optuna trial.",
    )
    parser.add_argument("--search-seed", type=int, default=271828)
    parser.add_argument(
        "--search-seeds",
        type=int,
        nargs="+",
        default=[101, 102],
        help=(
            "Training seeds evaluated in every trial; Optuna maximizes their mean "
            "best validation accuracy."
        ),
    )
    parser.add_argument("--depths", type=int, nargs="+", choices=[1, 2], default=[1, 2])
    parser.add_argument("--degrees", type=int, nargs="+", choices=[1, 2, 3], default=[1, 2, 3])
    parser.add_argument(
        "--search-optimizers", nargs="+", choices=["adamw", "radam", "rmsprop"], default=["adamw", "rmsprop"]
    )
    parser.add_argument(
        "--search-schedulers", nargs="+", choices=["cosine", "onecycle", "none"], default=["cosine", "onecycle", "none"]
    )
    parser.add_argument(
        "--search-batch-sizes",
        type=int,
        nargs="+",
        default=[32, 64],
        help="Candidate training batches; kept conservative for the literal 14-qubit simulator.",
    )
    parser.add_argument(
        "--study-suffix",
        default="",
        help="Optional explicit suffix for a fresh study under otherwise identical settings.",
    )

    # Final evaluation.
    parser.add_argument(
        "--final-epochs",
        type=int,
        default=100,
        help="Epochs for each final multi-seed training after recipe selection.",
    )
    parser.add_argument(
        "--final-seeds",
        type=int,
        nargs="+",
        default=[41, 42, 43, 44, 45, 46],
        help="Independent seeds used for the final frozen-recipe evaluation.",
    )

    # Optional compact ablation, not part of the default all-stage run.
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--ablation-epochs", type=int, default=20)
    parser.add_argument("--ablation-seeds", type=int, nargs="+", default=[41])
    parser.add_argument("--ablation-max-per-class", type=int, default=300)

    args = parser.parse_args(argv)
    if args.quick:
        args.datasets = ["MNIST"]
        args.trials = 2
        args.search_epochs = 2
        args.search_seeds = [101]
        args.search_max_per_class = 16
        args.final_max_per_class = 24
        args.val_per_class = 8
        args.test_max_per_class = 8
        args.final_epochs = 2
        args.final_seeds = [41]
        args.shot_repeats = 2
        args.train_shots = min(args.train_shots, 257)
        args.eval_shots = min(args.eval_shots, 257)
        args.reference_ansatz_depth = min(args.reference_ansatz_depth, 1)
        args.search_batch_sizes = [16]
        args.eval_batch_size = 32

    def require_unique(name: str, values: Sequence[Any]) -> None:
        if len(values) != len(set(values)):
            parser.error(f"--{name} contains duplicate values: {list(values)}")

    def require_seed(name: str, value: int) -> None:
        if not 0 <= int(value) <= 2**32 - 1:
            parser.error(f"--{name} must be in [0, 2**32-1]")

    if not math.isfinite(float(args.val_fraction)) or not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be finite and lie strictly between 0 and 1")
    positive_names = (
        "trials",
        "search_epochs",
        "final_epochs",
        "ablation_epochs",
        "val_per_class",
        "search_max_per_class",
        "final_max_per_class",
        "ablation_max_per_class",
        "reference_ansatz_depth",
        "eval_batch_size",
        "shot_repeats",
        "eval_shots",
    )
    for name in positive_names:
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if int(args.test_max_per_class) < 0:
        parser.error("--test-max-per-class must be nonnegative (0 means the full test partition)")
    if int(args.train_shots) < 0:
        parser.error("--train-shots must be nonnegative (0 means exact-statevector training)")
    if int(args.workers) < 0:
        parser.error("--workers must be nonnegative")
    if not math.isfinite(float(args.gradient_clip)) or float(args.gradient_clip) <= 0.0:
        parser.error("--gradient-clip must be finite and strictly positive")
    if not math.isfinite(float(args.beta_init)):
        parser.error("--beta-init must be finite")
    if not str(args.outdir).strip() or not str(args.data_root).strip():
        parser.error("--outdir and --data-root must be nonempty")
    if len(str(args.study_suffix)) > 100:
        parser.error("--study-suffix must contain at most 100 characters")
    if any(int(batch) <= 0 for batch in args.search_batch_sizes):
        parser.error("--search-batch-sizes values must all be positive")

    unique_lists = {
        "datasets": args.datasets,
        "search-seeds": args.search_seeds,
        "final-seeds": args.final_seeds,
        "ablation-seeds": args.ablation_seeds,
        "depths": args.depths,
        "degrees": args.degrees,
        "search-optimizers": args.search_optimizers,
        "search-schedulers": args.search_schedulers,
        "search-batch-sizes": args.search_batch_sizes,
    }
    for name, values in unique_lists.items():
        require_unique(name, values)
    require_seed("split-seed", args.split_seed)
    require_seed("search-seed", args.search_seed)
    for name, values in (
        ("search-seeds", args.search_seeds),
        ("final-seeds", args.final_seeds),
        ("ablation-seeds", args.ablation_seeds),
    ):
        for value in values:
            require_seed(name, int(value))
    return args


def run_metadata(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "source_sha256": source_sha256(),
        "core_provenance": core_provenance(args, device),
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version,
        "torch": torch.__version__,
        "optuna": getattr(optuna, "__version__", None),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "classes": list(CLASSES),
        "input": "8x8 grayscale amplitude encoding on six work qubits",
        "model": "QFNN only; no ModelB or external local model module",
        "architecture_choices": {"depth": list(args.depths), "degree": list(args.degrees)},
        "phase": "enabled" if args.use_phase else "disabled (default unphased architecture)",
        "head": "always-on trainable scalar beta plus eight trainable class biases",
        "train_shots": {
            "value": int(args.train_shots),
            "sampling": "one multinomial draw over the full final address+ancilla distribution",
            "tuned": False,
            "gradient": "straight-through estimator in simulator",
        },
        "selection_protocol": (
            "Optuna and epoch selection use a fixed stratified validation split from official "
            "training data. Official test data is loaded only after all final checkpoints "
            "for a dataset are fixed."
        ),
        "search_scope": (
            "substantial validation search; the default sixty trials evaluated under "
            "two seeds are not an exhaustive architecture or optimizer search"
        ),
        "resource_reporting_scope": (
            "logical qubits and reflection/data-oracle counts are abstract oracle-level "
            "quantities, not compiled gate counts or two-qubit depths"
        ),
        "arguments": vars(args),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.stage == "self-test":
        run_self_test()
        return

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    device = safe_device(args.device)
    atomic_write_json(outdir / "run_config.json", run_metadata(args, device))
    print(f"[run] output={outdir}")
    print(f"[run] device={device}; datasets={args.datasets}; stage={args.stage}")
    print(
        f"[run] train shots={args.train_shots} (fixed, not tuned); "
        f"phase={'on' if args.use_phase else 'off'}; affine head=on"
    )
    if args.stage in ("all", "search"):
        print(
            f"[run] substantial validation search: {args.trials} trial(s) per dataset x "
            f"{len(args.search_seeds)} seed(s) x {args.search_epochs} epoch(s), "
            "with validation-only selection."
        )
        if device.type == "cpu":
            print(
                "[run] warning: depth-two trials retain a 14-qubit complex statevector and "
                "can be slow and memory-intensive on CPU. Default search batches are 32/64; "
                "run --stage all --quick first and prefer CUDA for the full study."
            )
    if args.stage in ("all", "final") and not args.quick:
        print(
            f"[run] long final workload requested: {len(args.datasets)} dataset(s) x "
            f"{len(args.final_seeds)} seeds x {args.final_epochs} epochs, followed by exact "
            f"and {args.shot_repeats}x finite-shot test evaluation. The run is resumable."
        )
        if device.type == "cpu":
            print(
                "[run] warning: the literal depth-two 14-qubit statevector path is slow on CPU; "
                "use --quick first and prefer a CUDA device for the full run."
            )

    selected: Dict[str, Dict[str, Any]]
    if args.stage in ("all", "search"):
        selected = {}
        existing_path = outdir / "selected_recipes.json"
        if existing_path.exists():
            try:
                selected.update(load_selected_recipes(outdir))
            except Exception:
                pass
        for dataset in args.datasets:
            selected[dataset] = search_one_dataset(
                dataset, args=args, device=device, outdir=outdir
            )
            save_selected_recipes(outdir, selected)
    else:
        try:
            selected = load_selected_recipes(outdir)
        except FileNotFoundError:
            if args.stage == "ablation":
                selected = {}
            else:
                raise

    if args.stage in ("all", "final"):
        run_final(
            args.datasets,
            selected=selected,
            args=args,
            device=device,
            outdir=outdir,
        )

    if args.stage == "ablation" or args.run_ablation:
        run_compact_ablation(
            args=args,
            device=device,
            outdir=outdir,
            selected=selected,
        )

    print(f"\nDone. Results are in: {outdir}")


if __name__ == "__main__":
    main()
