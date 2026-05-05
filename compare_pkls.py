import sys
import pickle
import numpy as np

f1 = sys.argv[1] if len(sys.argv) > 1 else r'd:\Chida\Projects\SMPLX\generated\now_this_is_very_this_is_actually_an_insert_of_a_kind_of_an_envelope_for_stationary_and_this_is_a_very_italian_design.pkl'
f2 = sys.argv[2] if len(sys.argv) > 2 else r'd:\Chida\Projects\SMPLX\how2sign_pkls_cropTrue_shapeFalse\_-adcxjm1R4_0-8-rgb_front.pkl'

def analyze(path):
    print(f"--- {path.split(chr(92))[-1]} ---")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if 'smplx' in data:
        arr = np.array(data['smplx'], dtype=np.float32)
    else:
        return
    print(f"Shape: {arr.shape}")
    
    regions = {
        'Global Orient (0-3)': (0, 3),
        'Body (3-66)': (3, 66),
        'Left Hand (66-111)': (66, 111),
        'Right Hand (111-156)': (111, 156),
        'Jaw (156-159)': (156, 159),
        'Shape (159-169)': (159, 169),
        'Expression (169-179)': (169, 179),
        'Transl (179-182)': (179, 182),
    }
    for name, (s, e) in regions.items():
        if arr.shape[1] > s:
            region = arr[:, s:min(e, arr.shape[1])]
            print(f"  {name}: min={region.min():.3f} max={region.max():.3f} mean={region.mean():.3f} std={region.std():.3f}")
    
    # Frame velocity
    if arr.shape[0] > 1:
        vel = np.abs(np.diff(arr, axis=0))
        print(f"  Frame velocity: mean={vel.mean():.4f} max={vel.max():.4f}")
    
    # Motion range (how much does each joint actually move?)
    if arr.shape[0] > 1:
        per_dim_range = arr.max(axis=0) - arr.min(axis=0)
        body_range = per_dim_range[3:66].mean()
        hand_range = per_dim_range[66:156].mean()
        print(f"  Body motion range (mean): {body_range:.4f}")
        print(f"  Hand motion range (mean): {hand_range:.4f}")
    
    # Global orient analysis
    print(f"  GO first frame: {arr[0, :3]}")
    print(f"  GO last frame:  {arr[-1, :3]}")
    print()

analyze(f1)
analyze(f2)
