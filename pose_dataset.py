"""Dataset utilities for training a VAE on SMPL-X pose pickles.

Defaults are aligned to the VAE plan in MD files/VAE_MOTION_PRIOR_PLAN.md,
but this loader uses the `datasett` PKL files by default.
"""

from __future__ import annotations

import glob
import io
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray


class _CPUUnpickler(pickle.Unpickler):
    """Loads CUDA pickles on CPU-only machines."""

    def find_class(self, module: str, name: str):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def load_pickle_cpu(path: str) -> Dict:
    with open(path, "rb") as f:
        return _CPUUnpickler(f).load()


def _slice_pose_dims(smplx_182: np.ndarray, pose_mode: str) -> np.ndarray:
    """Return selected SMPL-X dimensions.

    Expected layout for 182D rows:
    [root3, body63, lhand45, rhand45, jaw3, betas10, expr10, transl_or_cam3]
    """
    if smplx_182.ndim != 2:
        raise ValueError(f"Expected 2D SMPL-X array [T, D], got shape {smplx_182.shape}")

    if smplx_182.shape[1] == 182:
        if pose_mode == "full182":
            return smplx_182
        if pose_mode == "pose156":
            return smplx_182[:, :156]
        if pose_mode == "pose169":
            return smplx_182[:, :169]
        raise ValueError(f"Unsupported pose_mode={pose_mode}")

    if smplx_182.shape[1] == 156:
        if pose_mode != "pose156":
            raise ValueError(
                "Input is already 156D. Use pose_mode='pose156' for this dataset."
            )
        return smplx_182

    raise ValueError(f"Unsupported SMPL-X dim={smplx_182.shape[1]} (expected 156 or 182)")


def _pad_or_truncate(seq: np.ndarray, seq_len: int) -> np.ndarray:
    t, d = seq.shape
    if t == seq_len:
        return seq
    if t > seq_len:
        return seq[:seq_len]

    padded = np.zeros((seq_len, d), dtype=np.float32)
    padded[:t] = seq
    if t > 0:
        padded[t:] = seq[t - 1]
    return padded


class SignLanguagePoseDataset(Dataset):
    """Loads PKL files as fixed-length SMPL-X pose sequences."""

    def __init__(
        self,
        pkl_dir: str,
        seq_len: int = 300,
        pose_mode: str = "full182",
        root_relative: bool = True,
        stats: Optional[NormalizationStats] = None,
        pkl_files: Optional[Sequence[str]] = None,
    ):
        self.pkl_dir = pkl_dir
        self.seq_len = seq_len
        self.pose_mode = pose_mode
        self.root_relative = root_relative
        self.stats = stats

        if pkl_files is None:
            self.pkl_files = sorted(glob.glob(os.path.join(pkl_dir, "*.pkl")))
        else:
            self.pkl_files = []
            for p in pkl_files:
                if os.path.isabs(p) or os.path.dirname(p):
                    self.pkl_files.append(os.path.normpath(p))
                else:
                    self.pkl_files.append(os.path.join(pkl_dir, p))

        if not self.pkl_files:
            raise ValueError(f"No .pkl files found in {pkl_dir}")

    def __len__(self) -> int:
        return len(self.pkl_files)

    def _extract_smplx(self, data: Dict) -> np.ndarray:
        if "smplx" not in data:
            raise KeyError("Missing 'smplx' key in pose pickle")

        smplx_arr = np.asarray(data["smplx"], dtype=np.float32)
        seq = _slice_pose_dims(smplx_arr, self.pose_mode).astype(np.float32)

        # Root-relative behavior: zero translation/camera shift channel in full182.
        if self.root_relative and self.pose_mode == "full182" and seq.shape[1] == 182:
            seq[:, 179:182] = 0.0

        return _pad_or_truncate(seq, self.seq_len)

    def __getitem__(self, idx: int) -> torch.Tensor:
        pkl_path = self.pkl_files[idx]
        data = load_pickle_cpu(pkl_path)
        seq = self._extract_smplx(data)

        if self.stats is not None:
            seq = (seq - self.stats.mean) / self.stats.std

        return torch.from_numpy(seq.astype(np.float32))


def compute_normalization_stats(
    pkl_paths: Sequence[str],
    seq_len: int = 300,
    pose_mode: str = "full182",
    root_relative: bool = True,
    max_sequences: Optional[int] = None,
    seed: int = 42,
) -> NormalizationStats:
    """Compute mean/std over frames in sequences.

    For large datasets, set max_sequences to a smaller number (e.g. 1024-4096)
    to estimate stats quickly.
    """
    if not pkl_paths:
        raise ValueError("pkl_paths is empty")

    running_sum = None
    running_sq = None
    total = 0

    dataset = SignLanguagePoseDataset(
        pkl_dir=os.path.dirname(pkl_paths[0]),
        seq_len=seq_len,
        pose_mode=pose_mode,
        root_relative=root_relative,
        pkl_files=pkl_paths,
    )

    n_total = len(dataset)
    if max_sequences is None or max_sequences <= 0 or max_sequences >= n_total:
        indices = range(n_total)
    else:
        rng = np.random.default_rng(seed)
        indices = rng.choice(n_total, size=int(max_sequences), replace=False)

    for i in indices:
        seq = dataset[int(i)].numpy()
        if running_sum is None:
            running_sum = np.zeros(seq.shape[1], dtype=np.float64)
            running_sq = np.zeros(seq.shape[1], dtype=np.float64)

        running_sum += seq.sum(axis=0)
        running_sq += np.square(seq).sum(axis=0)
        total += seq.shape[0]

    mean = (running_sum / total).astype(np.float32)
    var = (running_sq / total) - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)

    return NormalizationStats(mean=mean, std=std)


def save_stats(path: str, stats: NormalizationStats) -> None:
    np.savez(path, mean=stats.mean, std=stats.std)


def load_stats(path: str) -> NormalizationStats:
    data = np.load(path)
    return NormalizationStats(mean=data["mean"].astype(np.float32), std=data["std"].astype(np.float32))


def split_paths(pkl_paths: Sequence[str], train_ratio: float = 0.95, seed: int = 42) -> Tuple[List[str], List[str]]:
    paths = list(pkl_paths)
    rng = np.random.default_rng(seed)
    rng.shuffle(paths)

    split_idx = max(1, int(len(paths) * train_ratio))
    split_idx = min(split_idx, len(paths) - 1) if len(paths) > 1 else len(paths)

    train = paths[:split_idx]
    val = paths[split_idx:]
    if not val:
        val = train[-1:]
    return train, val
