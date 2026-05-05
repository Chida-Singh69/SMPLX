"""
Sign Language Dataset Adapter for Motion Diffusion Model (MDM).

This module provides a PyTorch Dataset that loads unified SMPL-X pkl files
and prepares them for training with MDM-style text-to-motion diffusion models.

Format: Each pkl has {'smplx': np.ndarray [T, 182]} and a mapping JSON maps
        pkl filenames to English sentence translations.

Usage:
    from sign_language_dataset import SignLanguageDataset, SignLanguageDataModule
    
    dataset = SignLanguageDataset(
        pkl_dir='unified_pkls/',
        mapping_path='mapping.json',
        max_frames=300,
    )
"""

import os
import json
import pickle
import io
import random
import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] PyTorch not available")


class CPU_Unpickler(pickle.Unpickler):
    """Handles torch tensors saved on GPU."""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
        return super().find_class(module, name)


def load_pkl(path: str) -> dict:
    with open(path, 'rb') as f:
        return CPU_Unpickler(f).load()


class SignLanguageDataset(Dataset):
    """
    Dataset for training text-conditioned motion generation on ASL SMPL-X data.
    
    Each sample returns:
        motion:  [max_frames, pose_dim]  float32 tensor (zero-padded)
        length:  int                      actual motion length
        text:    str                      English sentence
        mask:    [max_frames]             bool tensor (True for valid frames)
    """
    
    def __init__(
        self,
        pkl_dir: str,
        mapping_path: str,
        max_frames: int = 300,
        pose_dim: int = 182,
        min_frames: int = 10,
        augment: bool = True,
        normalize: bool = True,
        stats_path: Optional[str] = None,
    ):
        super().__init__()
        self.pkl_dir = pkl_dir
        self.max_frames = max_frames
        self.pose_dim = pose_dim
        self.min_frames = min_frames
        self.augment = augment
        self.normalize = normalize
        
        # Load mapping
        with open(mapping_path, 'r', encoding='utf-8') as f:
            full_mapping = json.load(f)
        
        # Filter to only existing pkl files
        self.entries = []
        missing = 0
        for pkl_name, text in full_mapping.items():
            pkl_path = os.path.join(pkl_dir, pkl_name)
            if os.path.exists(pkl_path):
                self.entries.append((pkl_name, text))
            else:
                missing += 1
        
        print(f"[SignLanguageDataset] Loaded {len(self.entries)} samples "
              f"({missing} missing pkl files skipped)")
        
        # Compute or load normalization stats
        self.mean = np.zeros(pose_dim, dtype=np.float32)
        self.std = np.ones(pose_dim, dtype=np.float32)
        
        if normalize:
            if stats_path and os.path.exists(stats_path):
                stats = np.load(stats_path)
                self.mean = stats['mean']
                self.std = stats['std']
                print(f"[SignLanguageDataset] Loaded normalization stats from {stats_path}")
            else:
                print("[SignLanguageDataset] Computing normalization stats...")
                self._compute_stats()
                if stats_path:
                    np.savez(stats_path, mean=self.mean, std=self.std)
                    print(f"[SignLanguageDataset] Saved stats to {stats_path}")
    
    def _compute_stats(self, max_samples: int = 5000):
        """Compute per-dimension mean and std across the dataset."""
        all_frames = []
        indices = list(range(min(len(self.entries), max_samples)))
        random.shuffle(indices)
        
        for idx in indices[:max_samples]:
            pkl_name, _ = self.entries[idx]
            try:
                data = load_pkl(os.path.join(self.pkl_dir, pkl_name))
                smplx = data['smplx'] if isinstance(data, dict) else data
                if hasattr(smplx, 'numpy'):
                    smplx = smplx.numpy()
                # Ensure correct dim
                if smplx.shape[1] < self.pose_dim:
                    padded = np.zeros((smplx.shape[0], self.pose_dim), dtype=np.float32)
                    padded[:, :smplx.shape[1]] = smplx
                    smplx = padded
                elif smplx.shape[1] > self.pose_dim:
                    smplx = smplx[:, :self.pose_dim]
                all_frames.append(smplx)
            except Exception:
                continue
        
        if all_frames:
            all_data = np.concatenate(all_frames, axis=0)
            self.mean = all_data.mean(axis=0).astype(np.float32)
            self.std = all_data.std(axis=0).astype(np.float32)
            # Avoid division by zero
            self.std[self.std < 1e-6] = 1.0
            print(f"  Stats computed from {len(all_frames)} sequences, "
                  f"{all_data.shape[0]} total frames")
    
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx: int) -> Dict[str, object]:
        pkl_name, text = self.entries[idx]
        
        # Load SMPL-X params
        data = load_pkl(os.path.join(self.pkl_dir, pkl_name))
        smplx = data['smplx'] if isinstance(data, dict) else data
        if hasattr(smplx, 'numpy'):
            smplx = smplx.numpy()
        smplx = smplx.astype(np.float32)
        
        # Ensure correct dimension
        T = smplx.shape[0]
        if smplx.shape[1] < self.pose_dim:
            padded = np.zeros((T, self.pose_dim), dtype=np.float32)
            padded[:, :smplx.shape[1]] = smplx
            smplx = padded
        elif smplx.shape[1] > self.pose_dim:
            smplx = smplx[:, :self.pose_dim]
        
        # Skip too-short sequences
        if T < self.min_frames:
            T = self.min_frames
            smplx_padded = np.zeros((T, self.pose_dim), dtype=np.float32)
            smplx_padded[:smplx.shape[0]] = smplx
            smplx = smplx_padded
        
        # Random crop / truncate to max_frames
        if T > self.max_frames:
            if self.augment:
                start = random.randint(0, T - self.max_frames)
            else:
                start = 0
            smplx = smplx[start:start + self.max_frames]
            T = self.max_frames
        
        # Data augmentation
        if self.augment:
            # Small noise on pose params (not on shape/expression)
            noise = np.random.normal(0, 0.002, size=smplx[:T, :156].shape).astype(np.float32)
            smplx[:T, :156] += noise
        
        # Normalize
        if self.normalize:
            smplx = (smplx - self.mean) / self.std
        
        # Pad to max_frames
        motion = np.zeros((self.max_frames, self.pose_dim), dtype=np.float32)
        motion[:T] = smplx[:T]
        
        # Create mask
        mask = np.zeros(self.max_frames, dtype=bool)
        mask[:T] = True
        
        return {
            'motion': torch.from_numpy(motion),        # [max_frames, pose_dim]
            'length': T,                                 # int
            'text': text,                                # str
            'mask': torch.from_numpy(mask),              # [max_frames]
            'filename': pkl_name,                        # for debugging
        }
    
    def denormalize(self, motion: np.ndarray) -> np.ndarray:
        """Reverse normalization for inference output."""
        return motion * self.std + self.mean


class SignLanguageDataModule:
    """Convenience wrapper for train/val/test splits."""
    
    def __init__(
        self,
        pkl_dir: str,
        mapping_path: str,
        max_frames: int = 300,
        batch_size: int = 64,
        num_workers: int = 4,
        val_split: float = 0.05,
        test_split: float = 0.05,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Create full dataset (without augmentation for splitting)
        full_dataset = SignLanguageDataset(
            pkl_dir=pkl_dir,
            mapping_path=mapping_path,
            max_frames=max_frames,
            augment=False,
            normalize=True,
            stats_path=os.path.join(pkl_dir, 'norm_stats.npz'),
        )
        
        # Split
        n = len(full_dataset)
        n_test = int(n * test_split)
        n_val = int(n * val_split)
        n_train = n - n_val - n_test
        
        generator = torch.Generator().manual_seed(seed)
        self.train_set, self.val_set, self.test_set = torch.utils.data.random_split(
            full_dataset, [n_train, n_val, n_test], generator=generator
        )
        
        # Enable augmentation for training subset
        # (Augmentation is applied per-item, so we set it on the full dataset
        #  and rely on the training subset wrapper)
        full_dataset.augment = True
        
        print(f"[DataModule] Train: {n_train}, Val: {n_val}, Test: {n_test}")
        
        # Store stats for inference denormalization
        self.mean = full_dataset.mean
        self.std = full_dataset.std
    
    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
            pin_memory=True, drop_last=True,
            collate_fn=self._collate,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            collate_fn=self._collate,
        )
    
    @staticmethod
    def _collate(batch):
        """Custom collate that handles variable text strings."""
        return {
            'motion': torch.stack([b['motion'] for b in batch]),
            'length': torch.tensor([b['length'] for b in batch]),
            'text': [b['text'] for b in batch],
            'mask': torch.stack([b['mask'] for b in batch]),
        }


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python sign_language_dataset.py <pkl_dir> <mapping_json>")
        print("Example: python sign_language_dataset.py unified_pkls/ mapping.json")
        sys.exit(1)
    
    pkl_dir = sys.argv[1]
    mapping = sys.argv[2]
    
    ds = SignLanguageDataset(
        pkl_dir=pkl_dir,
        mapping_path=mapping,
        max_frames=300,
        augment=False,
        normalize=False,
    )
    
    print(f"\nDataset size: {len(ds)}")
    
    if len(ds) > 0:
        sample = ds[0]
        print(f"Sample motion shape: {sample['motion'].shape}")
        print(f"Sample length: {sample['length']}")
        print(f"Sample text: {sample['text'][:80]}...")
        print(f"Sample mask sum: {sample['mask'].sum().item()} valid frames")
        
        # Stats
        lengths = []
        for i in range(min(100, len(ds))):
            lengths.append(ds[i]['length'])
        lengths = np.array(lengths)
        print(f"\nSequence length stats (first {len(lengths)} samples):")
        print(f"  Min: {lengths.min()}, Max: {lengths.max()}, "
              f"Mean: {lengths.mean():.1f}, Median: {np.median(lengths):.1f}")
