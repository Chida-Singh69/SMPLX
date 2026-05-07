"""
render_mvp_demo.py
==================
Batch-renders 9 SilentVoice demo section videos:
  3 sections x 3 genders (neutral, male, female) = 9 MP4s

Each section concatenates PKL pose sequences with 6-frame blend
transitions and embeds per-sentence captions into the video.

Usage:
    python scripts/render_mvp_demo.py
    python scripts/render_mvp_demo.py --section 1 --gender male
    python scripts/render_mvp_demo.py --gender female
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import pickle
import io
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.core.sentence_to_smplx import SentenceToSMPLX

PKL_DIR      = ROOT / "data" / "raw_poses" / "how2sign_pkls_cropTrue_shapeFalse"
MVP_DIR      = ROOT / "data" / "mvp"
MAPPING_FILE = MVP_DIR / "mvp_demo_mapping.json"

BLEND_FRAMES    = 6
FPS_DEFAULT     = 15
MAX_PER_CLIP    = 100   # cap per sentence (~6.7s at 15fps)
GENDERS         = ["neutral", "male", "female"]


# ── PKL loader ─────────────────────────────────────────────────────────────
def load_pkl_cpu(path: str) -> dict:
    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch.storage" and name == "_load_from_bytes":
                import torch
                def _load(b):
                    try:    return torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
                    except: return torch.load(io.BytesIO(b), map_location="cpu")
                return _load
            return super().find_class(module, name)
    with open(path, "rb") as f:
        return CPU_Unpickler(f).load()


def extract_smplx(pkl_path: str) -> np.ndarray:
    data = load_pkl_cpu(pkl_path)
    arr = np.asarray(data["smplx"], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 182:
        raise ValueError(f"Unexpected smplx shape {arr.shape}")
    return arr


def blend_transition(a: np.ndarray, b: np.ndarray, n: int = BLEND_FRAMES) -> np.ndarray:
    end, start = a[-1], b[0]
    alphas = np.linspace(0, 1, n + 2)[1:-1]
    return np.stack([(1 - al) * end + al * start for al in alphas])


def build_subtitle_timeline(sentence_texts: list[str], frame_counts: list[int]) -> list[dict]:
    """
    Build frame-level subtitle timeline from per-sentence texts and frame counts.
    Each entry: {start_frame, end_frame, text}
    Matches the format expected by SentenceToSMPLX.render_animation().
    """
    timeline = []
    cursor = 0
    for text, n_frames in zip(sentence_texts, frame_counts):
        start = cursor
        end   = cursor + n_frames - 1
        timeline.append({"start_frame": start, "end_frame": end, "text": text})
        cursor += n_frames + BLEND_FRAMES   # account for blend gap
    return timeline


def build_section_data(sentence_defs: list[dict], max_per_clip: int):
    """Load PKLs, concatenate with blend transitions, build subtitle timeline.
    Returns (pose_array [T,182], subtitle_timeline).
    """
    arrays, texts = [], []
    for s in sentence_defs:
        path = PKL_DIR / s["pkl"]
        if not path.exists():
            print(f"  [WARN] Missing: {s['pkl']} - skipping")
            continue
        arr = extract_smplx(str(path))[:max_per_clip]
        arrays.append(arr)
        texts.append(s["text"])
        print(f"  Loaded {s['pkl']} -> {arr.shape[0]} frames")

    if not arrays:
        raise RuntimeError("No valid PKL files for section")

    # Build subtitle timeline BEFORE concatenation (frame counts per clip)
    frame_counts = [a.shape[0] for a in arrays]
    subtitle_timeline = build_subtitle_timeline(texts, frame_counts)

    # Concatenate with blend transitions
    concat = arrays[0]
    for nxt in arrays[1:]:
        trans  = blend_transition(concat, nxt)
        concat = np.vstack([concat, trans, nxt])

    return concat, subtitle_timeline


def render_section_gender(section: dict, gender: str, animator: SentenceToSMPLX,
                           fps: int, max_per_clip: int):
    out_path = ROOT / section["videos"][gender]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[Section {section['id']} | {gender}] {section['title']}")
    print("-" * 56)

    seq, subtitle_timeline = build_section_data(section["sentences"], max_per_clip)
    print(f"  Total frames: {seq.shape[0]}  (~{seq.shape[0]/fps:.1f}s)")

    pose_data = {"smplx": seq, "fps": fps}
    animator.render_animation(
        pose_data,
        save_path=str(out_path),
        fps=fps,
        subtitle_timeline=subtitle_timeline,
    )
    print(f"  [SAVED] {out_path.name}")
    return str(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gender",       default=None,   choices=GENDERS,
                    help="Render only this gender (omit = all 3)")
    ap.add_argument("--section",      type=int, default=None, choices=[1, 2, 3],
                    help="Render only this section (omit = all 3)")
    ap.add_argument("--fps",          type=int, default=FPS_DEFAULT)
    ap.add_argument("--max-per-clip", type=int, default=MAX_PER_CLIP)
    ap.add_argument("--model-path",   default="models")
    ap.add_argument("--device",       default="cpu")
    args = ap.parse_args()

    with open(MAPPING_FILE) as f:
        mapping = json.load(f)

    sections = mapping["sections"]
    if args.section:
        sections = [s for s in sections if s["id"] == args.section]

    genders = [args.gender] if args.gender else GENDERS

    results = {}
    for gender in genders:
        print(f"\n{'='*60}")
        print(f"Initialising animator: gender={gender}")
        print(f"{'='*60}")
        animator = SentenceToSMPLX(
            model_path=args.model_path,
            gender=gender,
            device=args.device,
        )
        for sec in sections:
            key = f"s{sec['id']}_{gender}"
            try:
                path = render_section_gender(sec, gender, animator, args.fps, args.max_per_clip)
                results[key] = "OK  " + path
            except Exception as e:
                results[key] = "FAIL " + str(e)
                print(f"  [FAIL] {e}")

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
