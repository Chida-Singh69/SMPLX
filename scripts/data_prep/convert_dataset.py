"""
Universal dataset converter for ASL SMPL-X datasets.
Converts Neural Sign Actors / SignAvatars / any SMPL-X format → unified format.

Usage:
    python convert_dataset.py --input <nsa_data_dir> --output <output_dir> --mapping <mapping_json>
    python convert_dataset.py --inspect <path_to_single_pkl_or_npz>
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] torch not available. Will skip torch tensor conversion.")


def _torch_load_compat(obj, map_location=None):
    """Compatibility wrapper for PyTorch 2.6+ (weights_only default True)."""
    if not HAS_TORCH:
        raise RuntimeError("torch required to load torch-serialized objects")
    try:
        return torch.load(obj, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(obj, map_location=map_location)


class CPU_Unpickler(pickle.Unpickler):
    """Unpickler that forces all tensors to CPU."""
    def find_class(self, module, name):
        if HAS_TORCH and module == 'torch.storage' and name == '_load_from_bytes':
            import io
            return lambda b: _torch_load_compat(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)


def load_any(path: str) -> dict:
    """Load pkl or npz file, handling CUDA tensors gracefully."""
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.npz':
        return dict(np.load(path, allow_pickle=True))
    elif ext == '.npy':
        return {'data': np.load(path, allow_pickle=True)}
    elif ext == '.pkl' or ext == '.pickle':
        with open(path, 'rb') as f:
            return CPU_Unpickler(f).load()
    elif ext == '.pt' or ext == '.pth':
        if not HAS_TORCH:
            raise RuntimeError(f"torch required to load {ext} files")
        return _torch_load_compat(path, map_location='cpu')
    else:
        raise ValueError(f"Unknown file extension: {ext}")


def to_numpy(v) -> np.ndarray:
    """Convert any tensor/array to numpy."""
    if HAS_TORCH and isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    elif isinstance(v, np.ndarray):
        return v
    elif isinstance(v, (list, tuple)):
        return np.array(v)
    return np.array(v)


def inspect_file(path: str):
    """Pretty-print the structure of a pkl/npz file."""
    print(f"\n{'='*70}")
    print(f"  FILE: {path}")
    print(f"  SIZE: {os.path.getsize(path):,} bytes")
    print(f"{'='*70}")
    
    data = load_any(path)
    _inspect_recursive(data, indent=2)


def _inspect_recursive(obj, indent=2, max_depth=4, depth=0):
    """Recursively inspect data structure."""
    prefix = " " * indent
    
    if depth > max_depth:
        print(f"{prefix}... (max depth reached)")
        return
    
    if isinstance(obj, dict):
        print(f"{prefix}dict with {len(obj)} keys: {list(obj.keys())}")
        for k, v in obj.items():
            if hasattr(v, 'shape'):
                arr = to_numpy(v) if not isinstance(v, np.ndarray) else v
                print(f"{prefix}  '{k}': {type(v).__name__} shape={arr.shape} dtype={arr.dtype}"
                      f" range=[{arr.min():.3f}, {arr.max():.3f}]")
            elif isinstance(v, dict):
                print(f"{prefix}  '{k}': dict →")
                _inspect_recursive(v, indent + 4, max_depth, depth + 1)
            elif isinstance(v, (list, tuple)):
                print(f"{prefix}  '{k}': {type(v).__name__} len={len(v)}")
                if len(v) > 0 and hasattr(v[0], 'shape'):
                    print(f"{prefix}    [0]: {type(v[0]).__name__} shape={v[0].shape}")
            elif isinstance(v, (int, float, str, bool)):
                print(f"{prefix}  '{k}': {type(v).__name__} = {str(v)[:100]}")
            else:
                print(f"{prefix}  '{k}': {type(v).__name__}")
    elif hasattr(obj, 'shape'):
        arr = to_numpy(obj) if not isinstance(obj, np.ndarray) else obj
        print(f"{prefix}array shape={arr.shape} dtype={arr.dtype}")
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{type(obj).__name__} len={len(obj)}")
    else:
        print(f"{prefix}{type(obj).__name__}: {str(obj)[:100]}")


# ---------------------------------------------------------------------------
# Format Detection
# ---------------------------------------------------------------------------

def detect_format(data: dict) -> str:
    """Detect the format of SMPL-X data.
    
    Returns one of:
        'signavatars_182'  - SignAvatars/your current format: data['smplx'] is [N, 182]
        'signavatars_169'  - Unsmooth version: data['unsmooth_smplx'] is [N, 169]
        'separate_keys'    - NSA style: separate keys for each parameter
        'nested_smplx'     - data['smplx'] is a dict with sub-keys
        'flat_array'       - Just a flat numpy array [N, D]
        'unknown'
    """
    if not isinstance(data, dict):
        if hasattr(data, 'shape') and len(data.shape) == 2:
            return 'flat_array'
        return 'unknown'
    
    # Check for SignAvatars format: data['smplx'] is [N, 182]
    if 'smplx' in data:
        smplx = data['smplx']
        if hasattr(smplx, 'shape') or isinstance(smplx, np.ndarray):
            arr = to_numpy(smplx)
            if len(arr.shape) == 2 and arr.shape[1] in (156, 169, 182):
                return f'signavatars_{arr.shape[1]}'
        elif isinstance(smplx, dict):
            return 'nested_smplx'
    
    if 'unsmooth_smplx' in data and 'smplx' not in data:
        arr = to_numpy(data['unsmooth_smplx'])
        if len(arr.shape) == 2:
            return f'signavatars_{arr.shape[1]}'
    
    # Check for separate parameter keys (NSA format)
    param_keys = {'body_pose', 'global_orient', 'left_hand_pose', 'right_hand_pose'}
    alt_keys = {'body_pose', 'global_orient', 'lhand_pose', 'rhand_pose'}
    alt_keys2 = {'poses', 'trans', 'betas'}  # Another common format
    
    if param_keys.issubset(set(data.keys())) or alt_keys.issubset(set(data.keys())):
        return 'separate_keys'
    
    if alt_keys2.issubset(set(data.keys())):
        return 'separate_keys'
    
    # Check if there's a nested structure
    for key in data:
        v = data[key]
        if isinstance(v, dict) and any(k in v for k in ['body_pose', 'global_orient', 'poses']):
            return 'nested_smplx'
    
    return 'unknown'


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------

def convert_separate_keys_to_182(data: dict) -> np.ndarray:
    """Convert separate SMPL-X parameter keys to flat [N, 182] array.
    
    Handles various naming conventions from NSA, SMPL-X library, etc.
    """
    def get_param(names, default_dim, data):
        """Try multiple key names for a parameter."""
        for name in names:
            if name in data:
                v = to_numpy(data[name])
                if len(v.shape) == 1:
                    # Single frame or per-sequence param (like betas)
                    return v
                return v
        return None
    
    # Extract parameters with fallback key names
    global_orient = get_param(['global_orient', 'root_pose', 'root_orient'], 3, data)
    body_pose = get_param(['body_pose'], 63, data)
    left_hand = get_param(['left_hand_pose', 'lhand_pose'], 45, data)
    right_hand = get_param(['right_hand_pose', 'rhand_pose'], 45, data)
    jaw_pose = get_param(['jaw_pose'], 3, data)
    betas = get_param(['betas', 'shape'], 10, data)
    expression = get_param(['expression', 'exp'], 10, data)
    transl = get_param(['transl', 'trans', 'cam_trans', 'translation'], 3, data)
    
    if global_orient is None or body_pose is None:
        raise ValueError(f"Missing required params. Available keys: {list(data.keys())}")
    
    N = body_pose.shape[0]
    
    # Ensure all have correct shapes
    def ensure_shape(arr, target_cols, name):
        if arr is None:
            return np.zeros((N, target_cols), dtype=np.float32)
        arr = to_numpy(arr).astype(np.float32)
        if len(arr.shape) == 1:
            # Per-sequence param (e.g., betas) — broadcast to all frames
            arr = np.tile(arr[None, :], (N, 1))
        if arr.shape[1] != target_cols:
            print(f"  [WARN] {name}: expected {target_cols} cols, got {arr.shape[1]}. Padding/truncating.")
            if arr.shape[1] < target_cols:
                arr = np.pad(arr, ((0, 0), (0, target_cols - arr.shape[1])))
            else:
                arr = arr[:, :target_cols]
        return arr
    
    global_orient = ensure_shape(global_orient, 3, 'global_orient')
    body_pose = ensure_shape(body_pose, 63, 'body_pose')
    left_hand = ensure_shape(left_hand, 45, 'left_hand_pose')
    right_hand = ensure_shape(right_hand, 45, 'right_hand_pose')
    jaw_pose = ensure_shape(jaw_pose, 3, 'jaw_pose')
    betas = ensure_shape(betas, 10, 'betas')
    expression = ensure_shape(expression, 10, 'expression')
    transl = ensure_shape(transl, 3, 'transl')
    
    # Assemble: [N, 182]
    flat = np.concatenate([
        global_orient,   # 0:3
        body_pose,       # 3:66
        left_hand,       # 66:111
        right_hand,      # 111:156
        jaw_pose,        # 156:159
        betas,           # 159:169
        expression,      # 169:179
        transl,          # 179:182
    ], axis=1)
    
    assert flat.shape == (N, 182), f"Expected (N, 182), got {flat.shape}"
    return flat.astype(np.float32)


def convert_nested_smplx_to_182(data: dict) -> np.ndarray:
    """Convert nested smplx dict to flat [N, 182]."""
    smplx_data = data.get('smplx', data)
    
    if isinstance(smplx_data, dict):
        # Try smooth version first
        if 'smooth_smplx' in smplx_data:
            arr = to_numpy(smplx_data['smooth_smplx'])
            if len(arr.shape) == 2 and arr.shape[1] in (156, 169, 182):
                return pad_to_182(arr)
        
        # Try extracting separate keys from the nested dict
        try:
            return convert_separate_keys_to_182(smplx_data)
        except ValueError:
            pass
        
        # Try first available key
        for key in smplx_data:
            v = smplx_data[key]
            if hasattr(v, 'shape'):
                arr = to_numpy(v)
                if len(arr.shape) == 2 and arr.shape[1] >= 156:
                    return pad_to_182(arr)
    
    raise ValueError(f"Cannot extract SMPL-X params from nested dict. Keys: {list(smplx_data.keys()) if isinstance(smplx_data, dict) else 'not dict'}")


def pad_to_182(arr: np.ndarray) -> np.ndarray:
    """Pad a [N, D] array to [N, 182] with zeros if D < 182."""
    if arr.shape[1] == 182:
        return arr.astype(np.float32)
    elif arr.shape[1] < 182:
        padded = np.zeros((arr.shape[0], 182), dtype=np.float32)
        padded[:, :arr.shape[1]] = arr
        return padded
    else:
        # Truncate to 182
        return arr[:, :182].astype(np.float32)


def convert_to_unified(data: dict) -> dict:
    """Convert any detected format to unified format: {'smplx': [N, 182], ...}"""
    fmt = detect_format(data)
    print(f"  Detected format: {fmt}")
    
    if fmt.startswith('signavatars_'):
        dim = int(fmt.split('_')[1])
        key = 'smplx' if 'smplx' in data else 'unsmooth_smplx'
        arr = to_numpy(data[key])
        smplx_182 = pad_to_182(arr)
    elif fmt == 'separate_keys':
        smplx_182 = convert_separate_keys_to_182(data)
    elif fmt == 'nested_smplx':
        smplx_182 = convert_nested_smplx_to_182(data)
    elif fmt == 'flat_array':
        arr = to_numpy(data) if not isinstance(data, np.ndarray) else data
        smplx_182 = pad_to_182(arr)
    else:
        raise ValueError(f"Unknown format: {fmt}. Please inspect the file manually.")
    
    # Build unified output dict
    result = {
        'smplx': smplx_182,
    }
    
    # Preserve useful metadata if present
    if isinstance(data, dict):
        for meta_key in ['left_valid', 'right_valid', 'total_valid_index', 'height', 'width']:
            if meta_key in data:
                result[meta_key] = data[meta_key]
        
        # Also keep unsmooth version if available
        if 'unsmooth_smplx' in data:
            result['unsmooth_smplx'] = to_numpy(data['unsmooth_smplx'])
    
    return result


# ---------------------------------------------------------------------------
# Batch Conversion
# ---------------------------------------------------------------------------

def convert_directory(input_dir: str, output_dir: str, mapping_path: Optional[str] = None):
    """Convert all pkl/npz files in input_dir to unified format in output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all data files
    extensions = {'.pkl', '.pickle', '.npz', '.npy', '.pt', '.pth'}
    files = []
    for f in sorted(os.listdir(input_dir)):
        if os.path.splitext(f)[1].lower() in extensions:
            files.append(f)
    
    # Also check subdirectories (some datasets nest by video ID)
    for subdir in sorted(os.listdir(input_dir)):
        subpath = os.path.join(input_dir, subdir)
        if os.path.isdir(subpath):
            for f in sorted(os.listdir(subpath)):
                if os.path.splitext(f)[1].lower() in extensions:
                    files.append(os.path.join(subdir, f))
    
    print(f"\nFound {len(files)} data files in {input_dir}")
    
    # Load existing mapping if provided
    mapping = {}
    if mapping_path and os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        print(f"Loaded existing mapping with {len(mapping)} entries")
    
    converted = 0
    failed = 0
    new_mapping = {}
    
    for i, filename in enumerate(files):
        input_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(os.path.basename(filename))[0] + '.pkl'
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            data = load_any(input_path)
            result = convert_to_unified(data)
            
            # Save
            with open(output_path, 'wb') as f:
                pickle.dump(result, f)
            
            # Update mapping
            base_name = os.path.basename(filename)
            if base_name in mapping:
                new_mapping[output_filename] = mapping[base_name]
            elif output_filename in mapping:
                new_mapping[output_filename] = mapping[output_filename]
            
            converted += 1
            
            if (i + 1) % 100 == 0 or i == 0:
                frames = result['smplx'].shape[0]
                print(f"  [{i+1}/{len(files)}] {filename} -> {output_filename} ({frames} frames)")
                
        except Exception as e:
            print(f"  [FAIL] {filename}: {e}")
            failed += 1
    
    # Save new mapping
    if new_mapping:
        mapping_out = os.path.join(output_dir, 'mapping.json')
        with open(mapping_out, 'w', encoding='utf-8') as f:
            json.dump(new_mapping, f, indent=2, ensure_ascii=False)
        print(f"\nSaved mapping with {len(new_mapping)} entries to {mapping_out}")
    
    print(f"\n{'='*50}")
    print(f"  Conversion complete!")
    print(f"  Converted: {converted}")
    print(f"  Failed:    {failed}")
    print(f"  Output:    {output_dir}")
    print(f"{'='*50}")
    
    return converted, failed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Universal SMPL-X dataset converter")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect a single file')
    inspect_parser.add_argument('path', help='Path to pkl/npz file')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert a directory of files')
    convert_parser.add_argument('--input', required=True, help='Input directory with NSA/SignAvatars data')
    convert_parser.add_argument('--output', required=True, help='Output directory for unified pkl files')
    convert_parser.add_argument('--mapping', help='Optional: existing mapping JSON (pkl → sentence)')
    
    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge multiple mapping JSONs')
    merge_parser.add_argument('--inputs', nargs='+', required=True, help='Input mapping JSON files')
    merge_parser.add_argument('--output', required=True, help='Output merged mapping JSON')
    
    args = parser.parse_args()
    
    if args.command == 'inspect':
        inspect_file(args.path)
    elif args.command == 'convert':
        convert_directory(args.input, args.output, args.mapping)
    elif args.command == 'merge':
        merged = {}
        for inp in args.inputs:
            with open(inp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} entries from {inp}")
            merged.update(data)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        print(f"\nMerged {len(merged)} total entries → {args.output}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
