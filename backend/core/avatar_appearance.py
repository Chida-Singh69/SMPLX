"""
Avatar Appearance Module
========================
Centralized outfit and skin-tone definitions for SMPL-X avatars.

Each outfit is defined by vertex-mask functions that select which body
vertices belong to the garment, plus material properties (color, offset,
smoothing).  Masks are computed once from neutral-pose vertex positions
and reused every frame.

Supported outfits:
    tshirt              – Short-sleeve round-neck (sky-blue)
    full_sleeve_shirt   – Full-sleeve round-neck (very dark brown)
    long_sleeve_vneck   – Full-sleeve V-neckline (dark pink)
"""

import numpy as np

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    trimesh = None

try:
    import pyrender
    PYRENDER_AVAILABLE = True
except ImportError:
    PYRENDER_AVAILABLE = False
    pyrender = None


# ─── Color Palettes ────────────────────────────────────────────────────

SKIN_TONES = {
    'light':  [0.92, 0.78, 0.68, 1.0],
    'medium': [0.80, 0.60, 0.50, 1.0],
    'tan':    [0.72, 0.52, 0.40, 1.0],
    'brown':  [0.55, 0.38, 0.28, 1.0],
    'dark':   [0.38, 0.26, 0.20, 1.0],
}

OUTFIT_CATALOG = {
    'tshirt': {
        'label': 'T-Shirt',
        'color': [0.247, 0.784, 1.0, 1.0],       # Sky blue
        'offset': 0.008,
        'laplacian': 2,
    },
    'full_sleeve_shirt': {
        'label': 'Full Sleeve Shirt',
        'color': [0.18, 0.10, 0.05, 1.0],         # Very dark brown
        'offset': 0.008,
        'laplacian': 2,
    },
    'long_sleeve_vneck': {
        'label': 'Long Sleeve V-Neck',
        'color': [0.72, 0.30, 0.38, 1.0],         # Dark pink
        'offset': 0.007,
        'laplacian': 2,
    },
}

# Default outfit per gender
GENDER_DEFAULT_OUTFIT = {
    'neutral': 'tshirt',
    'male':    'full_sleeve_shirt',
    'female':  'long_sleeve_vneck',
}


# ─── Vectorized Proxy-Mesh Builder ────────────────────────────────────

def create_proxy_mesh(mesh, indices, mask_set, offset=0.005, laplacian=0):
    """Build a proxy mesh from selected vertices pushed along normals.

    This is a performance-optimized version using vectorized numpy
    operations instead of Python loops for face filtering.

    Args:
        mesh:      trimesh.Trimesh – full body mesh for the current frame
        indices:   np.ndarray – vertex indices belonging to the garment
        mask_set:  set[int]  – same indices as a set (for O(1) lookup)
        offset:    float – distance (metres) to push vertices outward
        laplacian: int – number of Laplacian smoothing iterations

    Returns:
        trimesh.Trimesh or None if no valid faces
    """
    if indices is None or len(indices) == 0:
        return None

    proxy_verts = np.copy(mesh.vertices)
    normals = mesh.vertex_normals
    proxy_verts[indices] += normals[indices] * offset

    # Vectorized face filtering: keep only faces where ALL 3 verts are in mask
    face_in_mask = np.isin(mesh.faces, indices).all(axis=1)
    proxy_faces = mesh.faces[face_in_mask]

    if len(proxy_faces) == 0:
        return None

    p_mesh = trimesh.Trimesh(vertices=proxy_verts, faces=proxy_faces)

    if laplacian > 0:
        try:
            trimesh.smoothing.filter_laplacian(p_mesh, iterations=laplacian)
        except Exception:
            pass

    return p_mesh


# ─── Mask Computation ─────────────────────────────────────────────────

def _compute_tshirt_mask(x, y, z):
    """T-shirt: torso band + short round sleeves + round neckline.

    Returns dict of layer_name -> vertex indices.
    """
    # Torso Y-band: waist to collarbone
    valid_y = (y > -0.42) & (y < 0.18)

    # Round neckline: carve out a sphere at the throat
    neck_dist = np.sqrt(x**2 + (y - 0.17)**2 + z**2)
    neckline = neck_dist > 0.08

    # Short sleeves: spheres centred on each shoulder joint
    # Left shoulder at x≈-0.18, right at x≈+0.18, both at y≈0.12
    d_left  = np.sqrt((x + 0.18)**2 + (y - 0.12)**2 + z**2)
    d_right = np.sqrt((x - 0.18)**2 + (y - 0.12)**2 + z**2)
    in_sleeves = (d_left < 0.22) | (d_right < 0.22) | (np.abs(x) < 0.2)

    body_mask = valid_y & neckline & in_sleeves
    return {'body': np.where(body_mask)[0]}


def _compute_full_sleeve_shirt_mask(x, y, z):
    """Full-sleeve shirt: torso + full-arm sleeves + round neckline.

    Body uses the same round neckline as the t-shirt (r=0.08) but with
    full-arm sleeves using enlarged shoulder spheres + arm-tube criterion.

    Returns dict of layer_name -> vertex indices.
    """
    # ── Body: full torso + full sleeves ──
    valid_y = (y > -0.42) & (y < 0.18)

    # Standard round neckline (same as t-shirt)
    neck_dist = np.sqrt(x**2 + (y - 0.17)**2 + z**2)
    neckline = neck_dist > 0.08

    # Full sleeves: large shoulder spheres + arm tubes
    d_left  = np.sqrt((x + 0.18)**2 + (y - 0.12)**2 + z**2)
    d_right = np.sqrt((x - 0.18)**2 + (y - 0.12)**2 + z**2)
    in_shoulder_sphere = (d_left < 0.42) | (d_right < 0.42)

    left_arm_xz  = np.sqrt((x + 0.18)**2 + z**2)
    right_arm_xz = np.sqrt((x - 0.18)**2 + z**2)
    in_arm_tube = (
        ((left_arm_xz < 0.10) | (right_arm_xz < 0.10))
        & (y > -0.50) & (y < 0.15)
    )

    in_core = np.abs(x) < 0.20
    in_garment = in_shoulder_sphere | in_arm_tube | in_core
    body_mask = valid_y & neckline & in_garment

    return {'body': np.where(body_mask)[0]}


def _compute_vneck_longsleeve_mask(x, y, z):
    """Long-sleeve V-neck: full torso + full-arm sleeves + V neckline.

    The V-neckline is achieved by varying the exclusion radius around
    the throat: at the back it matches a standard round neck, but at
    the front-centre it widens into a smooth V shape.

    Returns dict of layer_name -> vertex indices.
    """
    # ── Body ──
    valid_y = (y > -0.42) & (y < 0.18)

    # V-neckline:
    # The exclusion radius around the throat varies based on position:
    #   • At the back  (z < 0):  r = 0.08  (standard round neck)
    #   • At front-centre (z > 0, x ≈ 0): r = 0.14  (deeper V)
    #
    # front_factor:  0 at back, ramps to 1 at front
    # centre_factor: 1 at midline, fades to 0 at sides
    front_factor  = np.clip(z * 6.0, 0.0, 1.0)
    centre_factor = np.clip(1.0 - np.abs(x) * 7.0, 0.0, 1.0)
    v_depth = front_factor * centre_factor          # 0..1

    # Interpolate exclusion radius: 0.08 (back) → 0.14 (front-centre)
    exclusion_r = 0.08 + v_depth * 0.06

    neck_dist = np.sqrt(x**2 + (y - 0.17)**2 + z**2)
    neckline = neck_dist > exclusion_r

    # Full sleeves (same logic as hoodie)
    d_left  = np.sqrt((x + 0.18)**2 + (y - 0.12)**2 + z**2)
    d_right = np.sqrt((x - 0.18)**2 + (y - 0.12)**2 + z**2)
    in_shoulder_sphere = (d_left < 0.42) | (d_right < 0.42)

    left_arm_xz  = np.sqrt((x + 0.18)**2 + z**2)
    right_arm_xz = np.sqrt((x - 0.18)**2 + z**2)
    in_arm_tube = (
        ((left_arm_xz < 0.10) | (right_arm_xz < 0.10))
        & (y > -0.50) & (y < 0.15)
    )

    in_core = np.abs(x) < 0.20
    in_garment = in_shoulder_sphere | in_arm_tube | in_core

    body_mask = valid_y & neckline & in_garment
    return {'body': np.where(body_mask)[0]}


# Dispatch table
_MASK_FUNCTIONS = {
    'tshirt':             _compute_tshirt_mask,
    'full_sleeve_shirt':  _compute_full_sleeve_shirt_mask,
    'long_sleeve_vneck':  _compute_vneck_longsleeve_mask,
}


# ─── AvatarAppearance Class ───────────────────────────────────────────

class AvatarAppearance:
    """Manages per-frame garment layers for an SMPL-X avatar.

    Usage:
        appearance = AvatarAppearance(outfit='full_sleeve_shirt', skin_tone='medium')
        appearance.compute_masks(neutral_vertices)   # once at init

        # Per frame:
        layers = appearance.build_scene_layers(frame_mesh)
        for trimesh_obj, material in layers:
            scene.add(pyrender.Mesh.from_trimesh(trimesh_obj, material=material))
    """

    def __init__(self, outfit='tshirt', skin_tone='medium'):
        if outfit not in OUTFIT_CATALOG:
            raise ValueError(
                f"Unknown outfit '{outfit}'. "
                f"Available: {list(OUTFIT_CATALOG.keys())}"
            )
        if skin_tone not in SKIN_TONES:
            raise ValueError(
                f"Unknown skin tone '{skin_tone}'. "
                f"Available: {list(SKIN_TONES.keys())}"
            )

        self.outfit_key = outfit
        self.config = OUTFIT_CATALOG[outfit]
        self.skin_color = SKIN_TONES[skin_tone]

        # Populated by compute_masks()
        self._layers = {}  # layer_name -> {'indices': ndarray, 'mask_set': set}

    def compute_masks(self, neutral_vertices):
        """Compute vertex masks from a neutral-pose body.

        Args:
            neutral_vertices: np.ndarray of shape [V, 3]
        """
        x = neutral_vertices[:, 0]
        y = neutral_vertices[:, 1]
        z = neutral_vertices[:, 2]

        mask_fn = _MASK_FUNCTIONS[self.outfit_key]
        layer_indices = mask_fn(x, y, z)

        self._layers = {}
        for name, indices in layer_indices.items():
            self._layers[name] = {
                'indices': indices,
                'mask_set': set(indices.tolist()),
            }

        total = sum(len(v['indices']) for v in self._layers.values())
        print(
            f"[APPEARANCE] Outfit '{self.config['label']}' — "
            f"{len(self._layers)} layer(s), {total:,} vertices total"
        )

    @property
    def masks_computed(self):
        return len(self._layers) > 0

    def get_skin_material(self):
        """Return a pyrender material for the body skin."""
        if not PYRENDER_AVAILABLE:
            return None
        return pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            roughnessFactor=0.7,
            baseColorFactor=self.skin_color,
        )

    def build_scene_layers(self, mesh):
        """Build garment proxy meshes for the current frame.

        Args:
            mesh: trimesh.Trimesh – the posed SMPL-X body mesh

        Returns:
            list of (trimesh.Trimesh, pyrender.Material) tuples
        """
        if not PYRENDER_AVAILABLE or not self.masks_computed:
            return []

        cfg = self.config
        result = []

        # ── Main body layer ──
        body_data = self._layers.get('body')
        if body_data is not None:
            body_proxy = create_proxy_mesh(
                mesh,
                body_data['indices'],
                body_data['mask_set'],
                offset=cfg['offset'],
                laplacian=cfg['laplacian'],
            )
            if body_proxy is not None:
                mat = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=cfg['color'],
                    doubleSided=True,
                    metallicFactor=0.0,
                    roughnessFactor=0.8,
                )
                result.append((body_proxy, mat))

        return result
