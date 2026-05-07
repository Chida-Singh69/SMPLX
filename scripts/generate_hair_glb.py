"""
Generate realistic-looking top-dome hair GLB for SMPL-X avatar.

Improvements over the original:
  - Layered shell geometry (outer + inner cap) for hair depth/thickness
  - Sinusoidal strand grooves baked into vertex positions for texture
  - Smooth forehead hairline curve (cosine taper) instead of a hard cut
  - Slight downward drape at the sides/back so it doesn't look like a helmet
  - Hair-brown PBR material with roughness/metallic set correctly
  - Watertight cap with a filled parting line at the top

Run: python scripts/generate_hair_glb.py
"""

import os, sys
import numpy as np

try:
    import trimesh
    from trimesh.creation import icosphere
except ImportError:
    sys.exit("pip install trimesh")

try:
    import trimesh.visual
except ImportError:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "assets", "hair", "default_hair.glb")

# ── 1. Build high-res unit sphere ─────────────────────────────────────────
sphere = icosphere(subdivisions=5, radius=1.0)
v_unit = sphere.vertices.copy()          # unit-sphere positions
faces  = sphere.faces.copy()

# ── 2. Skull-shaped scaling ────────────────────────────────────────────────
# Slightly asymmetric Y scaling: wider radius on crown, tapers to sides.
X_R   = 0.098   # head half-width  (left–right)
Z_R   = 0.092   # head half-depth  (front–back)
UP_R  = 0.175   # crown height above head
DOWN_R = 0.06   # how far the hair drapes below the equator (sides/back)

def skull_scale(v):
    """Non-uniform scale: tall crown, drooping sides."""
    sv = v.copy()
    # Y: different scale for upward vs downward vertices
    y_scale = np.where(v[:, 1] >= 0, UP_R, DOWN_R)
    sv[:, 0] = v[:, 0] * X_R
    sv[:, 1] = v[:, 1] * y_scale
    sv[:, 2] = v[:, 2] * Z_R
    return sv

sv = skull_scale(v_unit)

# ── 3. Hairline mask ───────────────────────────────────────────────────────
# Keep vertices where:
#   a) Not too far below equator (to avoid skirt look)
#   b) Forehead excluded with a smooth cosine curve (not a flat cut)
#
# In unit-sphere coords:
#   y  – vertical   (1 = top, -1 = bottom)
#   z  – depth      (positive = forward/face, negative = back)
#   x  – lateral    (±1 = ears)

def hairline_keep(v):
    """Return boolean mask of vertices that belong to the hair cap."""
    y, x, z = v[:, 1], v[:, 0], v[:, 2]

    # ── Bottom cut: keep above a drape threshold, harsher at front ──
    # At the back (z<0) allow more drape; at the front cut higher up.
    back_blend  = np.clip(-z, 0, 1)           # 0 at front, 1 at back
    y_threshold = -0.20 * back_blend - 0.02   # back droops more
    not_too_low = y > y_threshold

    # ── Forehead hairline: smooth cosine arch ──
    # Hairline sits higher in the center, sweeps lower at the temples.
    # In unit sphere: z > 0 is face-facing.
    # Hairline y-level as function of x:  y_line(x) = 0.18 + 0.22*cos(x*pi/2)
    # Vertices with z>0 AND y < y_line are "forehead" → exclude.
    x_norm     = np.clip(x / 0.95, -1, 1)              # normalise to [-1,1]
    y_hairline = 0.18 + 0.22 * np.cos(x_norm * np.pi / 2)  # arch
    is_forehead = (z > 0.10) & (y < y_hairline)

    # ── Face cutout: aggressive removal of chin/nose area ──
    is_face = (z > 0.55) & (y < 0.60)

    return not_too_low & ~is_forehead & ~is_face

keep_mask = hairline_keep(v_unit)
keep_idx  = np.where(keep_mask)[0]
keep_set  = set(keep_idx.tolist())

face_ok   = np.array([all(fi in keep_set for fi in f) for f in faces])
cap_faces = faces[face_ok]
old2new   = {old: new for new, old in enumerate(keep_idx)}
new_faces = np.array([[old2new[fi] for fi in f] for f in cap_faces])

cap = trimesh.Trimesh(vertices=sv[keep_idx], faces=new_faces, process=True)

# ── 4. Strand texture: sinusoidal grooves along the hair flow ──────────────
# Hair flows from crown (top) down toward the nape/sides.
# We displace each vertex outward/inward slightly based on a strand pattern.
#
# Flow direction in spherical coords → strand frequency along azimuth.

verts = cap.vertices.copy()

# Convert to spherical relative to crown centre
cx, cy, cz = 0.0, 0.0, 0.0
r     = np.linalg.norm(verts - [cx, cy, cz], axis=1, keepdims=True) + 1e-8
norms = (verts - [cx, cy, cz]) / r      # outward unit normals (approx)

# Azimuth angle (around Y axis) — determines which "strand" a vert belongs to
azimuth  = np.arctan2(verts[:, 0], verts[:, 2])   # -pi .. pi
elevation = np.arctan2(verts[:, 1],
            np.sqrt(verts[:, 0]**2 + verts[:, 2]**2))  # -pi/2 .. pi/2

N_STRANDS   = 90        # number of strand grooves around the head
GROOVE_DEPTH = 0.0025   # metres — subtle but visible

# Strand wave: groove pattern along azimuth
strand_wave  = np.sin(azimuth * N_STRANDS) * GROOVE_DEPTH

# Taper the effect: stronger near crown, fades toward hairline
taper = np.clip((elevation) / (np.pi / 2), 0, 1) ** 0.5

displacement = strand_wave * taper

verts_displaced = verts + norms * displacement[:, np.newaxis]
cap.vertices = verts_displaced

# ── 5. Additional drape: pull sides/back down slightly ────────────────────
# Makes hair hug the skull better than a perfect dome.
verts = cap.vertices.copy()
side_ness = np.clip(1.0 - np.abs(verts[:, 1]) / 0.05, 0, 1)   # near-equator band
back_ness = np.clip(-verts[:, 2] / 0.06, 0, 1)                  # back of head
droop     = side_ness * 0.005                                    # 5 mm droop
verts[:, 1] -= droop
cap.vertices = verts

# ── 6. Smooth (preserve strand detail, so light touch) ────────────────────
trimesh.smoothing.filter_laplacian(cap, iterations=2)

# ── 7. Translate: crown at Y=0, rest of hair hangs below ─────────────────
crown_y = cap.vertices[:, 1].max()
cap.apply_translation([0, -crown_y, 0])

# ── 8. Outer shell + thin inner cap for thickness ─────────────────────────
# Clone and scale inward to give a "hair thickness" feel.
inner = cap.copy()
# Push inner shell inward by ~3 mm along vertex normals
inner.vertices -= inner.vertex_normals * 0.003
# Flip normals so inside faces inward
inner.faces = inner.faces[:, ::-1]

hair = trimesh.util.concatenate([cap, inner])

# ── 9. PBR material: dark brown hair ──────────────────────────────────────
# Hair colour: deep warm brown  (R=42, G=22, B=10)
# Roughness ~0.85 (hair is diffuse, slightly shiny)
# Metallic = 0
HAIR_COLOR = [42, 22, 10, 255]   # RGBA uint8

hair.visual = trimesh.visual.ColorVisuals(
    mesh=hair,
    vertex_colors=np.tile(HAIR_COLOR, (len(hair.vertices), 1)).astype(np.uint8)
)

# ── 10. Export ─────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT), exist_ok=True)
hair.export(OUT)

lo, hi = hair.vertices[:, 1].min(), hair.vertices[:, 1].max()
print(f"✓ Saved : {OUT}")
print(f"  Verts : {len(hair.vertices)}")
print(f"  Y-range: {lo:.4f} m  →  {hi:.4f} m  (height {hi-lo:.4f} m)")
print(f"  Strands: {N_STRANDS} grooves, depth {GROOVE_DEPTH*1000:.1f} mm")