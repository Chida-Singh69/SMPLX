#!/usr/bin/env python
"""
Full VAE Training Pipeline: Train → Cache → Test Blend

Runs all 3 steps sequentially. If training succeeds but later steps fail,
you can fix and rerun without retraining (checkpoint is saved).

Usage:
    python run_full_pipeline.py --pkl-dir how2sign_pkls_cropTrue_shapeFalse --epochs 100
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run command and check for errors."""
    print(f"\n{'='*70}")
    print(f"[PIPELINE] {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=os.getcwd())
    if result.returncode != 0:
        print(f"\n[ERROR] {description} failed with exit code {result.returncode}")
        return False
    print(f"\n[SUCCESS] {description} completed")
    return True


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description="Full VAE Pipeline: Train → Cache → Blend")
    parser.add_argument("--pkl-dir", type=str, default="data/raw_poses/how2sign_pkls_cropTrue_shapeFalse")
    parser.add_argument("--out-dir", type=str, default="checkpoints/vae_weights/vae_h2s")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=300)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--stats-max-seqs",
        type=int,
        default=2048,
        help="Max sequences to use for computing normalization stats during training (0=all)",
    )
    parser.add_argument(
        "--recompute-stats",
        action="store_true",
        help="Force recomputing normalization stats (ignores cached norm_stats.npz)",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain even if checkpoints already exist (no prompt)",
    )
    parser.add_argument("--skip-blend-test", action="store_true", help="Skip blend test at end")
    args = parser.parse_args()

    pkl_dir = Path(args.pkl_dir)
    out_dir = Path(args.out_dir)

    # ========== STEP 1: TRAINING ==========
    print("\n" + "="*70)
    print("[PIPELINE] STEP 1: TRAINING VAE")
    print("="*70)
    
    if (out_dir / "vae_best.pt").exists() and not args.retrain:
        print(f"[INFO] Found existing checkpoint: {out_dir}/vae_best.pt")
        print("[INFO] Skipping training (pass --retrain to retrain)")
    else:
        train_cmd = [
            sys.executable, os.path.join(script_dir, "vae_train.py"),
            "--pkl-dir", str(pkl_dir),
            "--out-dir", str(out_dir),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--seq-len", str(args.seq_len),
            "--patience", str(args.patience),
            "--num-workers", str(args.num_workers),
            "--stats-max-seqs", str(args.stats_max_seqs),
        ]
        if args.recompute_stats:
            train_cmd.append("--recompute-stats")
        if not run_command(train_cmd, "VAE Training"):
            sys.exit(1)

    # ========== STEP 2: BUILD LATENT CACHE ==========
    print("\n" + "="*70)
    print("[PIPELINE] STEP 2: BUILD LATENT CACHE")
    print("="*70)

    ckpt_path = out_dir / "vae_best.pt"
    stats_path = out_dir / "norm_stats.npz"
    cache_path = out_dir / "latent_cache.npz"

    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    if not stats_path.exists():
        print(f"[ERROR] Stats not found: {stats_path}")
        sys.exit(1)

    cache_cmd = [
        sys.executable, os.path.join(script_dir, "vae_inference.py"), "cache",
        "--pkl-dir", str(pkl_dir),
        "--ckpt", str(ckpt_path),
        "--stats", str(stats_path),
        "--out", str(cache_path),
    ]
    if not run_command(cache_cmd, "Build Latent Cache"):
        sys.exit(1)

    # ========== STEP 3: TEST BLEND ==========
    if not args.skip_blend_test:
        print("\n" + "="*70)
        print("[PIPELINE] STEP 3: TEST BLEND (2 random sequences)")
        print("="*70)

        # Find 2 random PKL files
        import glob
        pkl_files = sorted(glob.glob(str(pkl_dir / "*.pkl")))
        if len(pkl_files) < 2:
            print(f"[WARNING] Less than 2 PKL files found in {pkl_dir}, skipping blend test")
        else:
            test_files = [pkl_files[0], pkl_files[len(pkl_files)//2]]
            blend_out = out_dir / "test_blend"
            blend_cmd = [
                sys.executable, os.path.join(script_dir, "vae_inference.py"), "blend",
                "--cache", str(cache_path),
                "--ckpt", str(ckpt_path),
                "--stats", str(stats_path),
                "--files", test_files[0], test_files[1],
                "--weights", "0.7", "0.3",
                "--out", str(blend_out),
            ]
            if not run_command(blend_cmd, "Test Blend"):
                print(f"[WARNING] Blend test failed, but training & cache are intact")
            else:
                print(f"\n[INFO] Blended output saved to:")
                print(f"  - {blend_out}.npy")
                print(f"  - {blend_out}.npz")

    # ========== PIPELINE COMPLETE ==========
    print("\n" + "="*70)
    print("[PIPELINE] ✅ COMPLETE")
    print("="*70)
    print(f"\nCheckpoints saved to: {out_dir}")
    print(f"  - vae_best.pt       (best model)")
    print(f"  - vae_last.pt       (last model)")
    print(f"  - norm_stats.npz    (normalization stats)")
    print(f"  - train_history.json (metrics)")
    print(f"  - latent_cache.npz  (compressed latents for 30K sequences)")
    print(f"\nNext steps:")
    print(f"  1. Verify blended outputs look reasonable")
    print(f"  2. Integrate into sentence_to_smplx.py API")
    print(f"  3. Test end-to-end: text → poses → animation")


if __name__ == "__main__":
    main()
