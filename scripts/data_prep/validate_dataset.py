"""
Strict Data Validation Engine for Sign Language Diffusion Model
This script rigorously checks the dataset for corruption, shape mismatches,
and mapping alignment before launching an expensive A100 training job.
"""

import os
import sys
import json
import pickle
import io
import argparse
import numpy as np
from tqdm import tqdm

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _torch_load_compat(obj, map_location=None):
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to load this object")
    try:
        return torch.load(obj, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(obj, map_location=map_location)


class CPU_Unpickler(pickle.Unpickler):
    """Safely loads PyTorch tensors saved on a GPU into CPU memory."""
    def find_class(self, module, name):
        if HAS_TORCH and module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: _torch_load_compat(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)


def load_pkl(path: str):
    with open(path, 'rb') as f:
        return CPU_Unpickler(f).load()


def to_numpy(tensor):
    if HAS_TORCH and isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)


def validate_dataset(data_dir: str, mapping_path: str, pose_dim: int = 182, output_quarantine: str = "quarantine_list.txt"):
    print("="*60)
    print(" STARTING STRICT DATASET VALIDATION")
    print("="*60)
    
    if not os.path.exists(data_dir):
        print(f"[FAIL] Data directory does not exist: {data_dir}")
        sys.exit(1)
        
    if not os.path.exists(mapping_path):
        print(f"[FAIL] Mapping JSON does not exist: {mapping_path}")
        sys.exit(1)

    print(f"Loading mapping: {mapping_path}")
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
        
    total_files = len(mapping)
    print(f"Found {total_files} entries in the mapping file.")
    
    if total_files == 0:
        print("[FAIL] Mapping file is empty.")
        sys.exit(1)

    failed_files = []
    
    # Iterate with progress bar
    for pkl_name in tqdm(mapping.keys(), desc="Validating PKLs"):
        pkl_path = os.path.join(data_dir, pkl_name)
        
        # 1. Existence Check
        if not os.path.exists(pkl_path):
            failed_files.append((pkl_name, "File missing from data directory"))
            continue
            
        try:
            # 2. Unpickling Check
            data = load_pkl(pkl_path)
            
            # 3. Schema Check
            if isinstance(data, dict):
                if 'smplx' not in data:
                    failed_files.append((pkl_name, "Dictionary missing 'smplx' key"))
                    continue
                smplx = data['smplx']
            else:
                smplx = data
                
            # 4. Shape & NaN Check
            smplx_np = to_numpy(smplx)
            if len(smplx_np.shape) != 2:
                failed_files.append((pkl_name, f"Invalid shape: expected 2D, got {smplx_np.shape}"))
                continue
                
            T, D = smplx_np.shape
            if D not in [169, 182]:
                failed_files.append((pkl_name, f"Invalid pose dimension: expected 169 or 182, got {D}"))
                continue
                
            if np.isnan(smplx_np).any():
                failed_files.append((pkl_name, "Contains NaN values"))
                continue
                
            if np.isinf(smplx_np).any():
                failed_files.append((pkl_name, "Contains Inf values"))
                continue
                
        except Exception as e:
            failed_files.append((pkl_name, f"Unpickling/Processing crash: {str(e)}"))

    print("\n" + "="*60)
    print(" VALIDATION COMPLETE")
    print("="*60)
    
    if failed_files:
        print(f"[FAIL] Found {len(failed_files)} problematic files!")
        with open(output_quarantine, 'w') as f:
            for pkl, reason in failed_files:
                f.write(f"{pkl} | {reason}\n")
        print(f"A detailed list of bad files has been saved to '{output_quarantine}'.")
        print("ACTION REQUIRED: Remove these entries from your mapping.json before training.")
        sys.exit(1)
    else:
        print("[SUCCESS] Zero errors detected! All files are perfectly formatted.")
        print("[SUCCESS] The dataset is 100% ready for A100 Training.")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strict Dataset Validator for MDM")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to directory containing .pkl files")
    parser.add_argument("--mapping", type=str, required=True, help="Path to JSON mapping file")
    
    args = parser.parse_args()
    validate_dataset(args.data_dir, args.mapping)
