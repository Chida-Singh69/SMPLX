"""
Avatar Appearance Module
========================
Outfit, skin-tone, and face-asset definitions for SMPL-X avatars.
Eyes / eyebrows load from pre-built GLB proxy meshes.
"""

import os, numpy as np

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    trimesh = None
    TRIMESH_AVAILABLE = False

try:
    import pyrender
    PYRENDER_AVAILABLE = True
except ImportError:
    pyrender = None
    PYRENDER_AVAILABLE = False

# ── Asset paths ────────────────────────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
_ASSETS = os.path.normpath(os.path.join(_HERE, '..', '..', 'assets'))

_EYE_GLB  = os.path.join(_ASSETS, 'eyes',     'default_sphere.glb')
_BROW_GLB = os.path.join(_ASSETS, 'eyebrows', 'default_brow.glb')


# ── Color palettes ─────────────────────────────────────────────────────
SKIN_TONES = {
    'light':  [0.92, 0.78, 0.68, 1.0],
    'medium': [0.80, 0.60, 0.50, 1.0],
    'tan':    [0.72, 0.52, 0.40, 1.0],
    'brown':  [0.55, 0.38, 0.28, 1.0],
    'dark':   [0.38, 0.26, 0.20, 1.0],
}

EYE_COLORS = {
    'brown': {'iris': [0.35, 0.20, 0.10], 'pupil': [0.02, 0.02, 0.02]},
    'blue':  {'iris': [0.22, 0.42, 0.65], 'pupil': [0.02, 0.02, 0.02]},
}

EYEBROW_COLORS = {
    'black': [0.03, 0.03, 0.03],
    'brown': [0.18, 0.12, 0.08],
}

SCLERA_COLOR = [0.95, 0.95, 0.94]

OUTFIT_CATALOG = {
    'tshirt': {
        'label': 'T-Shirt',
        'color': [0.247, 0.784, 1.0, 1.0],
        'offset': 0.008, 'laplacian': 5,
    },
    'full_sleeve_shirt': {
        'label': 'Full Sleeve Shirt',
        'color': [0.18, 0.10, 0.05, 1.0],
        'offset': 0.008, 'laplacian': 12,
    },
    'long_sleeve_vneck': {
        'label': 'Long Sleeve V-Neck',
        'color': [0.72, 0.30, 0.38, 1.0],
        'offset': 0.007, 'laplacian': 5,
    },
}

GENDER_DEFAULT_OUTFIT = {
    'neutral': 'tshirt',
    'male':    'full_sleeve_shirt',
    'female':  'long_sleeve_vneck',
}


# ── Helpers ────────────────────────────────────────────────────────────

def _load_glb(path):
    """Load first geometry from a GLB. Returns trimesh.Trimesh or None."""
    if not TRIMESH_AVAILABLE or not os.path.exists(path):
        return None
    try:
        scene = trimesh.load(path, force='scene')
        meshes = [g for g in scene.geometry.values()
                  if isinstance(g, trimesh.Trimesh)]
        return meshes[0] if meshes else None
    except Exception as e:
        print(f"[WARN] GLB load failed {path}: {e}")
        return None


def create_proxy_mesh(mesh, indices, offset=0.005, laplacian=0):
    """Push selected vertices along normals → proxy mesh."""
    if indices is None or len(indices) == 0:
        return None
    proxy_verts = np.copy(mesh.vertices)
    normals = mesh.vertex_normals
    proxy_verts[indices] += normals[indices] * offset
    face_in_mask = np.isin(mesh.faces, indices).all(axis=1)
    proxy_faces = mesh.faces[face_in_mask]
    if len(proxy_faces) == 0:
        return None
    p = trimesh.Trimesh(vertices=proxy_verts, faces=proxy_faces, process=False)
    if laplacian > 0:
        try: trimesh.smoothing.filter_laplacian(p, iterations=laplacian)
        except: pass
    return p


# ── Outfit mask functions ──────────────────────────────────────────────

def _compute_tshirt_mask(x, y, z):
    valid_y = (y > -0.42) & (y < 0.18)
    neckline = np.sqrt(x**2 + (y - 0.17)**2 + z**2) > 0.08
    d_l = np.sqrt((x + 0.18)**2 + (y - 0.12)**2 + z**2)
    d_r = np.sqrt((x - 0.18)**2 + (y - 0.12)**2 + z**2)
    in_s = (d_l < 0.22) | (d_r < 0.22) | (np.abs(x) < 0.2)
    return {'body': np.where(valid_y & neckline & in_s)[0]}


def _compute_full_sleeve_shirt_mask(x, y, z):
    valid_y = (y > -0.42) & (y < 0.18)
    # Cylindrical collar: XZ-plane circle only → clean round neckline
    neck_cyl  = np.sqrt(x**2 + z**2)
    neck_band = np.abs(y - 0.155)
    inside_collar = (neck_cyl < 0.068) & (neck_band < 0.055)
    neckline = ~inside_collar
    d_l = np.sqrt((x + 0.18)**2 + (y - 0.12)**2 + z**2)
    d_r = np.sqrt((x - 0.18)**2 + (y - 0.12)**2 + z**2)
    lxz = np.sqrt((x + 0.18)**2 + z**2)
    rxz = np.sqrt((x - 0.18)**2 + z**2)
    in_arm = ((lxz < 0.10) | (rxz < 0.10)) & (y > -0.50) & (y < 0.15)
    in_g = (d_l < 0.42) | (d_r < 0.42) | in_arm | (np.abs(x) < 0.20)
    return {'body': np.where(valid_y & neckline & in_g)[0]}


def _compute_vneck_longsleeve_mask(x, y, z):
    valid_y = (y > -0.42) & (y < 0.18)
    ff = np.clip(z * 6.0, 0.0, 1.0)
    cf = np.clip(1.0 - np.abs(x) * 7.0, 0.0, 1.0)
    excl_r = 0.08 + ff * cf * 0.06
    neckline = np.sqrt(x**2 + (y - 0.17)**2 + z**2) > excl_r
    d_l = np.sqrt((x + 0.18)**2 + (y - 0.12)**2 + z**2)
    d_r = np.sqrt((x - 0.18)**2 + (y - 0.12)**2 + z**2)
    lxz = np.sqrt((x + 0.18)**2 + z**2)
    rxz = np.sqrt((x - 0.18)**2 + z**2)
    in_arm = ((lxz < 0.10) | (rxz < 0.10)) & (y > -0.50) & (y < 0.15)
    in_g = (d_l < 0.42) | (d_r < 0.42) | in_arm | (np.abs(x) < 0.20)
    return {'body': np.where(valid_y & neckline & in_g)[0]}


_MASK_FNS = {
    'tshirt':            _compute_tshirt_mask,
    'full_sleeve_shirt': _compute_full_sleeve_shirt_mask,
    'long_sleeve_vneck': _compute_vneck_longsleeve_mask,
}


# ── Main class ─────────────────────────────────────────────────────────

class AvatarAppearance:
    def __init__(
        self,
        gender='neutral',
        outfit=None,
        skin_tone='medium',
        eye_color='brown',
        eyebrow_color='black',
        hair_color='brown',   # kept for API compat, unused
    ):
        self.gender = gender.lower()
        if outfit is None:
            outfit = GENDER_DEFAULT_OUTFIT.get(self.gender, 'tshirt')
        if outfit not in OUTFIT_CATALOG:
            outfit = 'tshirt'

        self.outfit_key = outfit
        self.config = OUTFIT_CATALOG[outfit].copy()

        # Gender colour override
        if self.gender == 'male':
            self.config['color'] = [0.18, 0.10, 0.05, 1.0]
        elif self.gender == 'female':
            self.config['color'] = [0.72, 0.30, 0.38, 1.0]
        else:
            self.config['color'] = [0.247, 0.784, 1.0, 1.0]

        self.skin_color    = SKIN_TONES[skin_tone]
        self.eye_cfg       = EYE_COLORS[eye_color]
        self.eyebrow_color = EYEBROW_COLORS[eyebrow_color]

        # Runtime state
        self.body_indices  = None
        self._leye_indices = None
        self._reye_indices = None
        self._leye_center  = None
        self._reye_center  = None

        # Preload GLB proxies
        self._eye_glb  = _load_glb(_EYE_GLB)
        self._brow_glb = _load_glb(_BROW_GLB)

        print(f"[Appearance] gender={self.gender} outfit={self.outfit_key} "
              f"eye_glb={'ok' if self._eye_glb else 'MISSING'}")

    # ── Mask computation ───────────────────────────────────────────────

    def compute_masks(self, neutral_vertices, skinning_weights=None):
        x, y, z = neutral_vertices[:,0], neutral_vertices[:,1], neutral_vertices[:,2]
        layer = _MASK_FNS[self.outfit_key](x, y, z)
        self.body_indices = layer['body']
        self._compute_eye_masks(neutral_vertices, skinning_weights)

    def _compute_eye_masks(self, verts, sw):
        def _fallback():
            for side, cx in [('left', 0.032), ('right', -0.032)]:
                d = np.sqrt((verts[:,0]-cx)**2+(verts[:,1]-0.311)**2+(verts[:,2]-0.064)**2)
                idx = np.where(d < 0.016)[0]
                if side == 'left': self._leye_indices = idx
                else:              self._reye_indices = idx

        if sw is not None and sw.shape[1] > 24:
            self._leye_indices = np.where(sw[:,23] > 0.9)[0]
            self._reye_indices = np.where(sw[:,24] > 0.9)[0]
            if len(self._leye_indices)==0 or len(self._reye_indices)==0:
                _fallback()
        else:
            _fallback()

        if len(self._leye_indices) > 0: self._leye_center = verts[self._leye_indices].mean(axis=0)
        if len(self._reye_indices) > 0: self._reye_center = verts[self._reye_indices].mean(axis=0)

    # ── Material helpers ───────────────────────────────────────────────

    def get_skin_material(self):
        if not PYRENDER_AVAILABLE: return None
        return pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, roughnessFactor=0.7,
            baseColorFactor=self.skin_color
        )

    # ── Scene layers ───────────────────────────────────────────────────

    def build_scene_layers(self, mesh):
        """Garment proxy mesh."""
        if not PYRENDER_AVAILABLE or self.body_indices is None:
            return []
        result = []
        p = create_proxy_mesh(
            mesh, self.body_indices,
            offset=self.config['offset'],
            laplacian=self.config['laplacian']
        )
        if p is not None:
            mat = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=self.config['color'],
                doubleSided=True, metallicFactor=0.0, roughnessFactor=0.8
            )
            result.append((p, mat))
        return result

    def build_face_layers(self, mesh):
        """Eyes (vertex-coloured)."""
        if not PYRENDER_AVAILABLE: return []
        result = []
        verts   = mesh.vertices
        normals = mesh.vertex_normals

        # ── Eyes ──────────────────────────────────────────────────────
        for eye_idx, eye_ctr in [
            (self._leye_indices, self._leye_center),
            (self._reye_indices, self._reye_center),
        ]:
            if eye_idx is None or len(eye_idx) == 0: continue
            face_mask  = np.isin(mesh.faces, eye_idx).all(axis=1)
            gfaces     = mesh.faces[face_mask]
            if len(gfaces) == 0: continue

            lv = np.copy(verts[eye_idx]) + normals[eye_idx] * 0.0006
            g2l = {g: l for l, g in enumerate(eye_idx.tolist())}
            lf  = np.array([[g2l[f] for f in face] for face in gfaces])

            dirs = lv - lv.mean(axis=0)
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
            cos_a = dirs @ np.array([0, 0, 1])
            pt, it = np.cos(np.radians(12)), np.cos(np.radians(35))
            p_rgb = (np.array(self.eye_cfg['pupil']) * 255).astype(np.uint8)
            i_rgb = (np.array(self.eye_cfg['iris'])  * 255).astype(np.uint8)
            s_rgb = (np.array(SCLERA_COLOR)           * 255).astype(np.uint8)

            colors = np.zeros((len(eye_idx), 4), np.uint8)
            for i, ca in enumerate(cos_a):
                colors[i, :3] = p_rgb if ca > pt else (i_rgb if ca > it else s_rgb)
                colors[i, 3]  = 255

            sub = trimesh.Trimesh(vertices=lv, faces=lf,
                                  vertex_colors=colors, process=False)
            result.append(pyrender.Mesh.from_trimesh(sub, smooth=False))

        return result