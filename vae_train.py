"""Train a lightweight VAE on SMPL-X pose sequences from PKL files.

Example:
    python vae_train.py --pkl-dir datasett --pose-mode full182 --epochs 200
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pose_dataset import (
    SignLanguagePoseDataset,
    compute_normalization_stats,
    load_stats,
    save_stats,
    split_paths,
)
from vae_model import SignLanguageVAE


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float,
):
    recon_loss = F.mse_loss(recon, target, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + kl_weight * kl
    return loss, recon_loss, kl


def train_epoch(model, loader, optimizer, device, kl_weight, grad_clip):
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    for poses in loader:
        poses = poses.to(device)
        recon, mu, logvar = model(poses)
        loss, recon_loss, kl = vae_loss(recon, poses, mu, logvar, kl_weight)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl.item()

    n = max(1, len(loader))
    return {
        "loss": total_loss / n,
        "recon": total_recon / n,
        "kl": total_kl / n,
    }


@torch.no_grad()
def eval_epoch(model, loader, device, kl_weight):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    for poses in loader:
        poses = poses.to(device)
        recon, mu, logvar = model(poses)
        loss, recon_loss, kl = vae_loss(recon, poses, mu, logvar, kl_weight)

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl.item()

    n = max(1, len(loader))
    return {
        "loss": total_loss / n,
        "recon": total_recon / n,
        "kl": total_kl / n,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train SMPL-X VAE")
    parser.add_argument("--pkl-dir", type=str, default="datasett")
    parser.add_argument("--out-dir", type=str, default="checkpoints/vae")
    parser.add_argument("--seq-len", type=int, default=300)
    parser.add_argument("--pose-mode", type=str, default="full182", choices=["full182", "pose169", "pose156"])
    parser.add_argument("--root-relative", action="store_true", default=True)
    parser.add_argument("--no-root-relative", dest="root_relative", action="store_false")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--train-ratio", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--kl-max", type=float, default=1e-3)
    parser.add_argument("--kl-warmup-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (epochs)")
    parser.add_argument(
        "--recompute-stats",
        action="store_true",
        help="Force recomputing normalization stats even if norm_stats.npz exists",
    )
    parser.add_argument(
        "--stats-max-seqs",
        type=int,
        default=2048,
        help="Max sequences to use for computing normalization stats (0=all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    all_paths = sorted(glob.glob(os.path.join(args.pkl_dir, "*.pkl")))
    if not all_paths:
        raise ValueError(f"No PKL files found in {args.pkl_dir}")

    train_paths, val_paths = split_paths(all_paths, train_ratio=args.train_ratio, seed=args.seed)

    print(f"[INFO] Total files: {len(all_paths)} | Train: {len(train_paths)} | Val: {len(val_paths)}")
    stats_path = os.path.join(args.out_dir, "norm_stats.npz")

    if os.path.exists(stats_path) and not args.recompute_stats:
        print(f"[INFO] Loading cached normalization stats: {stats_path}")
        stats = load_stats(stats_path)
    else:
        max_seqs = None if args.stats_max_seqs <= 0 else int(args.stats_max_seqs)
        if max_seqs is None:
            print("[INFO] Computing normalization stats from FULL train split...")
        else:
            print(f"[INFO] Computing normalization stats from a sample of {max_seqs} train sequences...")
        stats = compute_normalization_stats(
            train_paths,
            seq_len=args.seq_len,
            pose_mode=args.pose_mode,
            root_relative=args.root_relative,
            max_sequences=max_seqs,
            seed=args.seed,
        )
        save_stats(stats_path, stats)
        print(f"[INFO] Saved normalization stats: {stats_path}")

    train_ds = SignLanguagePoseDataset(
        pkl_dir=args.pkl_dir,
        seq_len=args.seq_len,
        pose_mode=args.pose_mode,
        root_relative=args.root_relative,
        stats=stats,
        pkl_files=train_paths,
    )
    val_ds = SignLanguagePoseDataset(
        pkl_dir=args.pkl_dir,
        seq_len=args.seq_len,
        pose_mode=args.pose_mode,
        root_relative=args.root_relative,
        stats=stats,
        pkl_files=val_paths,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    sample = train_ds[0]
    pose_dim = int(sample.shape[-1])
    print(f"[INFO] Input shape: [T={args.seq_len}, D={pose_dim}]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = SignLanguageVAE(
        seq_len=args.seq_len,
        pose_dim=pose_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_val = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        kl_progress = min(1.0, epoch / max(1, args.kl_warmup_epochs))
        kl_weight = args.kl_max * kl_progress

        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            kl_weight=kl_weight,
            grad_clip=args.grad_clip,
        )
        val_metrics = eval_epoch(
            model=model,
            loader=val_loader,
            device=device,
            kl_weight=kl_weight,
        )

        epoch_log = {
            "epoch": epoch,
            "kl_weight": kl_weight,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_log)

        print(
            f"[E{epoch:03d}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"train_recon={train_metrics['recon']:.6f} "
            f"val_recon={val_metrics['recon']:.6f} "
            f"kl_w={kl_weight:.6f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": vars(args),
            "pose_dim": pose_dim,
        }

        last_path = os.path.join(args.out_dir, "vae_last.pt")
        torch.save(ckpt, last_path)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_path = os.path.join(args.out_dir, "vae_best.pt")
            torch.save(ckpt, best_path)
            print(f"[INFO] New best checkpoint: {best_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[INFO] Early stopping triggered after {epoch} epochs (patience={args.patience})")
                break

    history_path = os.path.join(args.out_dir, "train_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    config_path = os.path.join(args.out_dir, "train_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("[DONE] Training complete")
    print(f"[DONE] Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()
