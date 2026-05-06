import os
import re
import pickle
from typing import Dict, List, Optional

import numpy as np

from backend.core.sentence_to_smplx import SentenceToSMPLX


class PoseAssembler:
    """Assemble per-frame pose pickles into SMPL-X animation parameters."""

    FRAME_FILE_PATTERN = re.compile(r"_(\d+)_3D\.pkl$")

    def __init__(self, poses_root: str):
        self.poses_root = poses_root

    def list_folders(self) -> List[str]:
        """Return folder names that contain frame pose files."""
        if not os.path.isdir(self.poses_root):
            return []

        folders: List[str] = []
        for name in sorted(os.listdir(self.poses_root)):
            full = os.path.join(self.poses_root, name)
            if not os.path.isdir(full):
                continue
            if any(f.endswith("_3D.pkl") for f in os.listdir(full)):
                folders.append(name)
        return folders

    def _frame_files(self, folder_path: str) -> List[str]:
        files = [f for f in os.listdir(folder_path) if f.endswith("_3D.pkl")]

        def frame_index(filename: str) -> int:
            match = self.FRAME_FILE_PATTERN.search(filename)
            return int(match.group(1)) if match else 10**9

        return sorted(files, key=lambda f: (frame_index(f), f))

    @staticmethod
    def _to_1d_np(value, expected: int, key: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.shape[0] != expected:
            raise ValueError(f"Invalid shape for '{key}': expected {expected}, got {arr.shape[0]}")
        return arr

    @staticmethod
    def _load_pickle(path: str) -> Dict:
        with open(path, "rb") as f:
            return pickle.load(f)

    def assemble_folder(self, folder_name: str) -> Dict[str, np.ndarray]:
        """
        Load all frame files in a folder and build an animation dict.

        Output format:
            {
                'smplx': np.ndarray [N, 156],
                'fps': 15
            }
        """
        folder_path = os.path.join(self.poses_root, folder_name)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Pose folder not found: {folder_path}")

        frame_files = self._frame_files(folder_path)
        if not frame_files:
            raise ValueError(f"No frame pickle files found in: {folder_path}")

        frames: List[np.ndarray] = []
        for filename in frame_files:
            frame_path = os.path.join(folder_path, filename)
            frame_data = self._load_pickle(frame_path)

            try:
                root = self._to_1d_np(frame_data["smplx_root_pose"], 3, "smplx_root_pose")
                body = self._to_1d_np(frame_data["smplx_body_pose"], 63, "smplx_body_pose")
                left = self._to_1d_np(frame_data["smplx_lhand_pose"], 45, "smplx_lhand_pose")
                right = self._to_1d_np(frame_data["smplx_rhand_pose"], 45, "smplx_rhand_pose")
            except KeyError as exc:
                raise KeyError(f"Missing key {exc} in frame file: {frame_path}") from exc

            frames.append(np.concatenate([root, body, left, right], axis=0).astype(np.float32))

        return {
            "smplx": np.stack(frames, axis=0),
            "fps": 15,
        }


def render_pose_folder(
    folder_name: str,
    poses_root: str,
    output_path: str,
    gender: str = "neutral",
    model_path: str = "models",
    max_frames: Optional[int] = None,
) -> str:
    """Assemble a pose folder and render it to an MP4 file."""
    assembler = PoseAssembler(poses_root)
    pose_data = assembler.assemble_folder(folder_name)

    animator = SentenceToSMPLX(model_path=model_path, gender=gender, device="cpu")
    animator.render_animation(
        pose_data=pose_data,
        save_path=output_path,
        fps=int(pose_data.get("fps", 15)),
        max_frames=max_frames,
    )
    return output_path
