"""Render a quick MP4 preview from a saved SMPL-X motion (.npy/.npz/.pkl).

Works with outputs from `vae_inference.py` / `run_full_pipeline.py`, e.g.
`checkpoints/vae_h2s/test_blend.npy`.

Usage:
  python render_preview_video.py --input checkpoints/vae_h2s/test_blend.npy --out output/test_blend.mp4
    python render_preview_video.py --input generated/well_that_is_quite_strange.pkl --out output/well_that_is_quite_strange.mp4
    python render_preview_video.py --input how2sign_pkls_cropTrue_shapeFalse/_-adcxjm1R4_0-8-rgb_front.pkl --out output/h2s_sample.mp4

Notes:
- Accepts SMPL-X vectors of length 156/169/182.
- Rendering uses `SentenceToSMPLX` (pyrender offscreen if available; otherwise fallback).
- Some How2Sign `.pkl` files contain CUDA-serialized torch tensors; this script loads them on CPU.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import pickle
import io

import numpy as np
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sentence_to_smplx import SentenceToSMPLX


def load_pkl_cpu(path: str):
    """Load a pickle that may contain CUDA-serialized torch tensors.

    Many How2Sign-derived pkls include extra torch tensors (e.g. validity masks)
    that were saved on GPU. We always remap those storages to CPU for portability.
    """

    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                import torch

                def _load(b):
                    try:
                        return torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
                    except TypeError:
                        return torch.load(io.BytesIO(b), map_location='cpu')

                return _load
            return super().find_class(module, name)

    with open(path, 'rb') as f:
        return CPU_Unpickler(f).load()


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
    elif p.suffix.lower() == ".pkl":
        data = load_pkl_cpu(str(p))
        if isinstance(data, dict) and "smplx" in data:
            seq = data["smplx"]
        else:
            seq = data
    else:
        raise ValueError("--input must be a .npy, .npz, or .pkl file")

    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2:
        raise ValueError(f"Expected [T, D] array, got shape {seq.shape}")

    if seq.shape[1] not in (156, 169, 182):
        raise ValueError(f"Unexpected SMPL-X dim D={seq.shape[1]} (expected 156/169/182)")

    return seq


def parse_args():
    ap = argparse.ArgumentParser(description="Render MP4 preview from SMPL-X motion")
    ap.add_argument("--input", required=True, help="Path to .npy/.npz motion")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "data", "mp4_outputs", "preview.mp4"), help="Output mp4 path")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"])
    ap.add_argument("--model-path", default="models")
    ap.add_argument("--max-frames", type=int, default=240, help="Render at most this many frames")
    ap.add_argument("--device", default="cpu", help="Device to use (e.g., cpu, cuda)")
    ap.add_argument("--text", type=str, default=None, help="Optional text to render as proportional subtitles")
    return ap.parse_args()


def main():
    args = parse_args()

    seq = load_motion(args.input)
    pose_data = {"smplx": seq, "fps": args.fps}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    subtitle_timeline = None
    if args.text:
        # Create a basic proportional cumulative timeline
        T = min(seq.shape[0], args.max_frames) if args.max_frames else seq.shape[0]
        words = args.text.split()
        if words:
            total_chars = sum(len(w) for w in words)
            timeline = []
            curr = 0.0
            prefix = ""
            for w in words:
                frames = T * (len(w) / total_chars) if total_chars > 0 else T / len(words)
                start = int(round(curr))
                curr += frames
                end = int(round(curr)) - 1
                prefix = (prefix + " " + w).strip()
                timeline.append({'start_frame': start, 'end_frame': max(start, end), 'text': prefix})
            subtitle_timeline = timeline

    animator = SentenceToSMPLX(model_path=args.model_path, gender=args.gender, device=args.device)
    animator.render_animation(
        pose_data=pose_data,
        save_path=str(out_path),
        fps=args.fps,
        max_frames=args.max_frames,
        subtitle_timeline=subtitle_timeline
    )

    print(f"[DONE] Wrote preview video: {out_path}")


if __name__ == "__main__":
    main()
