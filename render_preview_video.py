"""Render a quick MP4 preview from a saved SMPL-X motion (.npy/.npz).

Works with outputs from `vae_inference.py` / `run_full_pipeline.py`, e.g.
`checkpoints/vae_h2s/test_blend.npy`.

Usage:
  python render_preview_video.py --input checkpoints/vae_h2s/test_blend.npy --out output/test_blend.mp4

Notes:
- Accepts SMPL-X vectors of length 156/169/182.
- Rendering uses `SentenceToSMPLX` (pyrender offscreen if available; otherwise fallback).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from sentence_to_smplx import SentenceToSMPLX


def load_motion(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if p.suffix.lower() == ".npy":
        seq = np.load(str(p))
    elif p.suffix.lower() == ".npz":
        data = np.load(str(p))
        if "smplx" in data.files:
            seq = data["smplx"]
        else:
            # Fallback: first array
            seq = data[data.files[0]]
    else:
        raise ValueError("--input must be a .npy or .npz file")

    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2:
        raise ValueError(f"Expected [T, D] array, got shape {seq.shape}")

    if seq.shape[1] not in (156, 169, 182):
        raise ValueError(f"Unexpected SMPL-X dim D={seq.shape[1]} (expected 156/169/182)")

    return seq


def parse_args():
    ap = argparse.ArgumentParser(description="Render MP4 preview from SMPL-X motion")
    ap.add_argument("--input", required=True, help="Path to .npy/.npz motion")
    ap.add_argument("--out", default="output/preview.mp4", help="Output mp4 path")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"])
    ap.add_argument("--model-path", default="models")
    ap.add_argument("--max-frames", type=int, default=240, help="Render at most this many frames")
    return ap.parse_args()


def main():
    args = parse_args()

    seq = load_motion(args.input)
    pose_data = {"smplx": seq, "fps": args.fps}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    animator = SentenceToSMPLX(model_path=args.model_path, gender=args.gender, device=None)
    animator.render_animation(
        pose_data=pose_data,
        save_path=str(out_path),
        fps=args.fps,
        max_frames=args.max_frames,
    )

    print(f"[DONE] Wrote preview video: {out_path}")


if __name__ == "__main__":
    main()
