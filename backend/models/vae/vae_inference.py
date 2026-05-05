"""Inference helpers for VAE latent caching and blending.

Examples:
  1) Build latent cache from datasett:
     python vae_inference.py cache --pkl-dir data/raw_poses/datasett --ckpt checkpoints/vae/vae_best.pt --stats checkpoints/vae/norm_stats.npz

  2) Blend two files and save output npz/npy:
     python vae_inference.py blend --cache checkpoints/vae/latent_cache.npz --ckpt checkpoints/vae/vae_best.pt --stats checkpoints/vae/norm_stats.npz --files 01621.pkl 02011.pkl --weights 0.6 0.4 --out output/vae_blend
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from backend.core.pose_dataset import load_pickle_cpu, load_stats
from vae_model import SignLanguageVAE


def _extract_pose(data: Dict, pose_mode: str, root_relative: bool) -> np.ndarray:
    smplx = np.asarray(data["smplx"], dtype=np.float32)
    if smplx.ndim != 2:
        raise ValueError(f"Expected [T, D], got {smplx.shape}")

    if smplx.shape[1] == 182:
        if pose_mode == "full182":
            seq = smplx.copy()
        elif pose_mode == "pose169":
            seq = smplx[:, :169].copy()
        elif pose_mode == "pose156":
            seq = smplx[:, :156].copy()
        else:
            raise ValueError(f"Unknown pose_mode={pose_mode}")
    elif smplx.shape[1] == 156 and pose_mode == "pose156":
        seq = smplx.copy()
    else:
        raise ValueError(
            f"PKL has {smplx.shape[1]} dims but pose_mode={pose_mode} was requested"
        )

    if root_relative and pose_mode == "full182" and seq.shape[1] == 182:
        seq[:, 179:182] = 0.0

    return seq


def _pad_or_truncate(seq: np.ndarray, seq_len: int) -> np.ndarray:
    t, d = seq.shape
    if t == seq_len:
        return seq
    if t > seq_len:
        return seq[:seq_len]

    out = np.zeros((seq_len, d), dtype=np.float32)
    out[:t] = seq
    if t > 0:
        out[t:] = seq[t - 1]
    return out


def load_model(ckpt_path: str, device: torch.device) -> Tuple[SignLanguageVAE, Dict]:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = SignLanguageVAE(
        seq_len=cfg["seq_len"],
        pose_dim=ckpt["pose_dim"],
        latent_dim=cfg["latent_dim"],
        hidden_dim=cfg["hidden_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


@torch.no_grad()
def encode_sequence(model: SignLanguageVAE, seq_norm: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(seq_norm.astype(np.float32)).unsqueeze(0).to(device)
    mu, _ = model.encode(x)
    return mu.squeeze(0).cpu().numpy().astype(np.float32)


@torch.no_grad()
def decode_latent(model: SignLanguageVAE, latent: np.ndarray, device: torch.device) -> np.ndarray:
    z = torch.from_numpy(latent.astype(np.float32)).unsqueeze(0).to(device)
    pred = model.decode(z).squeeze(0).cpu().numpy().astype(np.float32)
    return pred


def save_sequence(prefix: str, seq: np.ndarray) -> Tuple[str, str]:
    npy_path = f"{prefix}.npy"
    npz_path = f"{prefix}.npz"
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)

    np.save(npy_path, seq)
    np.savez(npz_path, smplx=seq)
    return npy_path, npz_path


def build_latent_cache(
    pkl_dir: str,
    ckpt_path: str,
    stats_path: str,
    output_path: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(ckpt_path, device)
    stats = load_stats(stats_path)

    paths = sorted(glob.glob(os.path.join(pkl_dir, "*.pkl")))
    if not paths:
        raise ValueError(f"No PKLs found in {pkl_dir}")

    latents = {}
    for i, path in enumerate(paths, start=1):
        data = load_pickle_cpu(path)
        seq = _extract_pose(data, cfg["pose_mode"], cfg.get("root_relative", True))
        seq = _pad_or_truncate(seq, cfg["seq_len"])
        seq_norm = (seq - stats.mean) / stats.std
        mu = encode_sequence(model, seq_norm, device)
        latents[os.path.basename(path)] = mu

        if i % 100 == 0 or i == len(paths):
            print(f"[CACHE] {i}/{len(paths)}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(output_path, **latents)
    print(f"[DONE] Saved latent cache: {output_path}")


def blend_from_files(
    files: Sequence[str],
    weights: Sequence[float],
    cache_path: str,
    ckpt_path: str,
    stats_path: str,
    out_prefix: str,
):
    if len(files) != len(weights):
        raise ValueError("--files and --weights must have same length")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(ckpt_path, device)
    stats = load_stats(stats_path)

    cache = np.load(cache_path)
    z_list = []
    for f in files:
        key = os.path.basename(f)
        if key not in cache.files:
            raise KeyError(f"Latent for {key} not found in cache")
        z_list.append(cache[key])

    z = np.stack(z_list, axis=0)
    w = np.asarray(weights, dtype=np.float32)
    w = w / (w.sum() + 1e-8)
    z_blend = (z * w[:, None]).sum(axis=0).astype(np.float32)

    pred_norm = decode_latent(model, z_blend, device)
    pred = (pred_norm * stats.std) + stats.mean

    if cfg.get("root_relative", True) and cfg["pose_mode"] == "full182" and pred.shape[1] == 182:
        pred[:, 179:182] = 0.0

    npy_path, npz_path = save_sequence(out_prefix, pred)
    print(f"[DONE] Saved blended motion: {npy_path} and {npz_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="VAE inference")
    sub = parser.add_subparsers(dest="cmd", required=True)

    cache_cmd = sub.add_parser("cache", help="Build latent cache from PKLs")
    cache_cmd.add_argument("--pkl-dir", type=str, default="data/raw_poses/datasett")
    cache_cmd.add_argument("--ckpt", type=str, required=True)
    cache_cmd.add_argument("--stats", type=str, required=True)
    cache_cmd.add_argument("--out", type=str, default="checkpoints/vae_weights/vae/latent_cache.npz")

    blend_cmd = sub.add_parser("blend", help="Blend cached latents and decode")
    blend_cmd.add_argument("--cache", type=str, required=True)
    blend_cmd.add_argument("--ckpt", type=str, required=True)
    blend_cmd.add_argument("--stats", type=str, required=True)
    blend_cmd.add_argument("--files", nargs="+", required=True)
    blend_cmd.add_argument("--weights", nargs="+", type=float, required=True)
    blend_cmd.add_argument("--out", type=str, default="output/vae_blended")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.cmd == "cache":
        build_latent_cache(
            pkl_dir=args.pkl_dir,
            ckpt_path=args.ckpt,
            stats_path=args.stats,
            output_path=args.out,
        )
        return

    if args.cmd == "blend":
        blend_from_files(
            files=args.files,
            weights=args.weights,
            cache_path=args.cache,
            ckpt_path=args.ckpt,
            stats_path=args.stats,
            out_prefix=args.out,
        )
        return

    raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
