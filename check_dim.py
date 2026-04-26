import torch
import smplx
import numpy as np

# Load SMPL-X neutral model
device = 'cpu'
model_path = r'd:\\Chida\\Projects\\SMPLX\\models'
smplx_model = smplx.create(
    model_path, 
    model_type='smplx',
    gender='NEUTRAL', 
    use_face_contour=False,
    ext='npz'
)

neutral_output = smplx_model(
    body_pose=torch.zeros((1, 63)),
    global_orient=torch.zeros((1, 3)),
    return_verts=True
)

v = neutral_output.vertices[0].detach().numpy()
print(f"Max Y: {v[:, 1].max()}")
print(f"Min Y: {v[:, 1].min()}")
print(f"Max Z: {v[:, 2].max()} (Front)")
print(f"Min Z: {v[:, 2].min()} (Back)")
print(f"Face front roughly Y between {v[:, 1].max() - 0.2} and {v[:, 1].max()}?")
