import pickle
import sys

pkl_path = r'd:\Chida\Projects\SMPLX\how2sign_pkls_cropTrue_shapeFalse\_-adcxjm1R4_0-8-rgb_front.pkl'
try:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded: {pkl_path}")
    print(f"Keys: {list(data.keys())}")
    for k, v in data.items():
        if hasattr(v, 'shape'):
            print(f"{k}: type={type(v)}, shape={v.shape}")
        elif isinstance(v, list) or isinstance(v, dict):
            print(f"{k}: type={type(v)}, length={len(v)}")
        else:
            print(f"{k}: type={type(v)}")
except Exception as e:
    print(f"Error reading pkl: {e}")
