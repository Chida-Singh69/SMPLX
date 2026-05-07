"""
Generate male fringe hair GLB for SMPL-X avatar.
Shape: tight skull cap + front fringe sweeping over forehead.
Crown at local Y=0, fringe hangs forward/down at front.
Run: python scripts/generate_male_fringe_glb.py
"""
import os, sys, numpy as np

try:
    import trimesh
except ImportError:
    sys.exit("pip install trimesh")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "assets", "hair", "male_fringe.glb")

sphere = trimesh.creation.icosphere(subdivisions=5, radius=1.0)
v = sphere.vertices.copy()

# ── Scale: tight cap, fringe droops at front only ─────────────────────────
X_R       = 0.095   # head width
Z_R       = 0.090   # head depth
UP_R      = 0.080   # low crown (tight male cap)
DOWN_R    = 0.025   # very little drape on sides/back (tight)
FRINGE_R  = 0.072   # fringe hangs down at FRONT

sv = v.copy()
sv[:, 0] = v[:, 0] * X_R
sv[:, 2] = v[:, 2] * Z_R

# Y: up → tight, down → depends on front/back
for i in range(len(v)):
    if v[i, 1] >= 0:
        sv[i, 1] = v[i, 1] * UP_R
    else:
        # Front (z > 0): allow fringe to droop more
        front_blend = np.clip(v[i, 2], 0, 1)
        y_r = DOWN_R + (FRINGE_R - DOWN_R) * front_blend
        sv[i, 1] = v[i, 1] * y_r

# ── Remove back-bottom and sides-bottom (keep fringe at front only) ────────
# Unit sphere keep rules:
#   - Upper hemisphere always kept (y > 0)
#   - Lower hemi: only keep if front-facing (z > -0.1) and not too low
#   - Fringe zone (z > 0.4): keep deeper down
#   - Sides/back (z < -0.1): cut off early

def keep_mask_fn(v):
    y, z, x = v[:, 1], v[:, 2], v[:, 0]
    upper       = y > 0
    # Side/back lower band: just a tiny rim
    side_back   = (y > -0.15) & (z < 0.1)
    # Fringe: front-facing, allow deeper hang, face cutout at very bottom
    fringe_zone = (z > 0.10) & (y > -0.80) & (y < 0.35)
    # Face cutout: very front-center, below forehead — show face
    is_face     = (z > 0.70) & (np.abs(x) < 0.55) & (y < 0.10)
    kept = (upper | side_back | fringe_zone) & ~is_face
    return kept

mask      = keep_mask_fn(v)
keep_idx  = np.where(mask)[0]
keep_set  = set(keep_idx.tolist())

face_ok   = np.array([all(fi in keep_set for fi in f) for f in sphere.faces])
cap_faces = sphere.faces[face_ok]
old2new   = {old: new for new, old in enumerate(keep_idx)}
new_faces = np.array([[old2new[fi] for fi in f] for f in cap_faces])

fringe = trimesh.Trimesh(vertices=sv[keep_idx], faces=new_faces, process=True)

# Shift crown to Y=0
crown = fringe.vertices[:, 1].max()
fringe.apply_translation([0, -crown, 0])

# Light smooth — preserve fringe shape
trimesh.smoothing.filter_laplacian(fringe, iterations=3)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fringe.export(OUT)
lo, hi = fringe.vertices[:,1].min(), fringe.vertices[:,1].max()
print(f"Saved male fringe: {OUT}")
print(f"  Verts: {len(fringe.vertices)},  Y: {lo:.4f} to {hi:.4f}")
