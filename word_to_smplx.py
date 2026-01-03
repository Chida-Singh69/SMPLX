import os
os.environ['PYOPENGL_PLATFORM'] = 'egl' # Attempt to use EGL for headless rendering

import sys
import torch
import smplx
import numpy as np
import imageio
import json
from scipy.ndimage import gaussian_filter1d

# Import rendering dependencies
try:
    import pyrender
    import trimesh
except ImportError:
    pyrender = None
    trimesh = None

# --- Joint Index Reference (SMPL-X) ---
# 0: Global orientation
# 1: Root (pelvis)
# 2: Left hip
# 3: Right hip
# 4: Spine1
# 5: Left knee
# 6: Right knee
# 7: Spine2
# 8: Left ankle
# 9: Right ankle
# 10: Spine3
# 11: Left foot
# 12: Right foot
# 13: Neck
# 14: Left shoulder
# 15: Right shoulder
# 16: Head
# 17: Left elbow
# 18: Right elbow
# 19: Left wrist
# 20: Right wrist
# (Hand joints start at 40 in trial.py)

class WordToSMPLX:
    def __init__(self, model_path="models", gender='neutral', viewport_width=640, viewport_height=480):
        # model_path is "models"
        # The smplx library (when model_type='smplx') expects model_path to be the directory *containing* the 'smplx' subfolder.
        # The actual model files are expected to be in model_path/smplx/
        
        # First, check if the specific model file exists as expected by the library structure
        actual_model_file_location = os.path.join(model_path, 'smplx', f"SMPLX_{gender.upper()}.npz")
        if not os.path.exists(actual_model_file_location):
            raise ValueError(f"Model file not found at: {actual_model_file_location}. Please ensure models are in 'models/smplx/'")

        self.smplx_model = smplx.create(
            model_path=model_path,  # This should be the parent directory, e.g., "models"
            model_type='smplx',     # This tells the library to look inside model_path + '/smplx'
            gender=gender,
            use_pca=False,  # Disable PCA to allow full finger control for sign language
            num_pca_comps=45,  # Full hand pose dimensions
            create_global_orient=True,
            create_body_pose=True,
            create_left_hand_pose=True,
            create_right_hand_pose=True,
            create_jaw_pose=True,
            create_leye_pose=True,
            create_reye_pose=True,
            create_betas=True,
            create_expression=True,
            create_transl=True,
            num_betas=10,
            num_expression_coeffs=10,
            flat_hand_mean=False,  # Allow curved hand poses for better clenching
            batch_size=1
        )

        self.finger_joint_mapping = self._get_finger_joint_mapping()
        self.hand_joint_limits = self._get_hand_joint_limits()
        
        if pyrender and trimesh:
            self.camera = pyrender.PerspectiveCamera(yfov=np.pi / 5.0)
            self.light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
            self.cam_pose = np.eye(4)
            self.cam_pose[2, 3] = 2.0
            self.cam_pose[1, 3] = -0.2
            self.renderer = pyrender.OffscreenRenderer(
                viewport_width=viewport_width,
                viewport_height=viewport_height
            )
        else:
            self.renderer = None

    def _get_finger_joint_mapping(self):
        """Map SMPL-X hand pose indices to anatomical joints for one hand (45 params)."""
        return {
            'thumb': {'cmc': (0, 3), 'mcp': (3, 6), 'ip': (6, 9)},
            'index': {'mcp': (9, 12), 'pip': (12, 15), 'dip': (15, 18)},
            'middle': {'mcp': (18, 21), 'pip': (21, 24), 'dip': (24, 27)},
            'ring': {'mcp': (27, 30), 'pip': (30, 33), 'dip': (33, 36)},
            'pinky': {'mcp': (36, 39), 'pip': (39, 42), 'dip': (42, 45)}
        }

    def _get_hand_joint_limits(self):
        """Define anatomically plausible joint limits in radians."""
        # These are simplified; real anatomy is more complex.
        # Order for 3DoF joint params: [main_flex_ext, abd_add_or_side_flex, twist]
        limits = {
            # Finger MCPs (Metacarpophalangeal joints)
            'mcp_flexion': (0, 1.57),          # 0 to 90 degrees
            'mcp_abduction': (-0.35, 0.35),    # -20 to 20 degrees
            'mcp_twist': (-0.2, 0.2),          # Minimal twist
            # Finger PIPs (Proximal Interphalangeal joints)
            'pip_flexion': (0, 1.75),          # 0 to 100 degrees
            'pip_side_flex': (-0.1, 0.1),      # Very minimal side movement
            'pip_twist': (-0.1, 0.1),          # Very minimal twist
            # Finger DIPs (Distal Interphalangeal joints)
            'dip_flexion': (0, 1.22),          # 0 to 70 degrees
            'dip_side_flex': (-0.05, 0.05),    # Almost no side movement
            'dip_twist': (-0.05, 0.05),        # Almost no twist
            # Thumb - more complex, simplified here
            'thumb_cmc_flexion': (-0.5, 0.9),  # Approx -30 to 50 deg
            'thumb_cmc_abduction': (0, 1.22),  # 0 to 70 deg
            'thumb_cmc_twist': (-0.5, 0.5),
            'thumb_mcp_flexion': (-0.2, 0.9),  # Approx -10 to 50 deg
            'thumb_mcp_abduction': (-0.1, 0.1),# Minimal abduction for thumb MCP
            'thumb_mcp_twist': (-0.2, 0.2),
            'thumb_ip_flexion': (-0.2, 1.4),   # Approx -10 to 80 deg
            'thumb_ip_side_flex': (-0.1, 0.1),
            'thumb_ip_twist': (-0.1, 0.1),
            'general_clamp': (-np.pi, np.pi) # Fallback for unhandled DoFs
        }
        return limits

    def _apply_anatomical_constraints_to_frame(self, hand_pose_frame_np):
        """Applies joint limits to a single frame of hand pose (45 params)."""
        constrained_pose = np.copy(hand_pose_frame_np)
        
        for finger_name, joints in self.finger_joint_mapping.items():
            for joint_type, (start_idx, end_idx) in joints.items():
                joint_params = constrained_pose[start_idx:end_idx] # Should be 3 params

                if finger_name == 'thumb':
                    if joint_type == 'cmc':
                        joint_params[0] = np.clip(joint_params[0], *self.hand_joint_limits['thumb_cmc_flexion'])
                        joint_params[1] = np.clip(joint_params[1], *self.hand_joint_limits['thumb_cmc_abduction'])
                        joint_params[2] = np.clip(joint_params[2], *self.hand_joint_limits['thumb_cmc_twist'])
                    elif joint_type == 'mcp':
                        joint_params[0] = np.clip(joint_params[0], *self.hand_joint_limits['thumb_mcp_flexion'])
                        joint_params[1] = np.clip(joint_params[1], *self.hand_joint_limits['thumb_mcp_abduction'])
                        joint_params[2] = np.clip(joint_params[2], *self.hand_joint_limits['thumb_mcp_twist'])
                    elif joint_type == 'ip':
                        joint_params[0] = np.clip(joint_params[0], *self.hand_joint_limits['thumb_ip_flexion'])
                        joint_params[1] = np.clip(joint_params[1], *self.hand_joint_limits['thumb_ip_side_flex'])
                        joint_params[2] = np.clip(joint_params[2], *self.hand_joint_limits['thumb_ip_twist'])
                else: # Index, Middle, Ring, Pinky
                    if joint_type == 'mcp':
                        joint_params[0] = np.clip(joint_params[0], *self.hand_joint_limits['mcp_flexion'])
                        joint_params[1] = np.clip(joint_params[1], *self.hand_joint_limits['mcp_abduction'])
                        joint_params[2] = np.clip(joint_params[2], *self.hand_joint_limits['mcp_twist'])
                    elif joint_type == 'pip':
                        joint_params[0] = np.clip(joint_params[0], *self.hand_joint_limits['pip_flexion'])
                        joint_params[1] = np.clip(joint_params[1], *self.hand_joint_limits['pip_side_flex'])
                        joint_params[2] = np.clip(joint_params[2], *self.hand_joint_limits['pip_twist'])
                    elif joint_type == 'dip':
                        joint_params[0] = np.clip(joint_params[0], *self.hand_joint_limits['dip_flexion'])
                        joint_params[1] = np.clip(joint_params[1], *self.hand_joint_limits['dip_side_flex'])
                        joint_params[2] = np.clip(joint_params[2], *self.hand_joint_limits['dip_twist'])
                
                constrained_pose[start_idx:end_idx] = joint_params
        return constrained_pose

    def _process_hand_pose_data(self, hand_pose_np, sigma=0.3):
        """Smoothes and applies anatomical constraints to hand pose data."""
        # 1. Smoothing
        smoothed_hand_pose = np.copy(hand_pose_np)
        if smoothed_hand_pose.shape[0] > 1: # Need at least 2 frames to smooth
            for i in range(smoothed_hand_pose.shape[1]): # Iterate over 45 parameters
                smoothed_hand_pose[:, i] = gaussian_filter1d(smoothed_hand_pose[:, i], sigma=sigma, mode='nearest')
        
        # 2. Apply anatomical constraints per frame
        constrained_hand_pose = np.zeros_like(smoothed_hand_pose)
        for i in range(smoothed_hand_pose.shape[0]): # Iterate over frames
            constrained_hand_pose[i, :] = self._apply_anatomical_constraints_to_frame(smoothed_hand_pose[i, :])
            
        # 3. Final global clamp as a safeguard (optional, could be part of _apply_anatomical_constraints_to_frame)
        # constrained_hand_pose = np.clip(constrained_hand_pose, *self.hand_joint_limits['general_clamp'])
        return constrained_hand_pose

    def load_pose_sequence(self, pkl_path):
        # Always load to CPU, allow for CUDA-originated files
        with open(pkl_path, "rb") as f:
            data = torch.load(f, map_location='cpu', weights_only=False)
        return data

    def render_animation(self, pose_data, save_path=None, fps=15):
        smplx_data = pose_data.get('smplx', None)
        if smplx_data is None or not (isinstance(smplx_data, np.ndarray) and isinstance(smplx_data[0], np.ndarray)):
            raise ValueError("'smplx' key missing or has unexpected structure in pose_data.")
        smplx_params = np.stack(smplx_data)  # shape: [N, D]
        N = smplx_params.shape[0]
        global_orient = torch.tensor(smplx_params[:, 0:3], dtype=torch.float32)
        body_pose = torch.tensor(smplx_params[:, 3:66], dtype=torch.float32)
        
        # Process hand poses
        left_hand_raw_np = smplx_params[:, 66:111]
        right_hand_raw_np = smplx_params[:, 111:156]
        
        left_hand_processed_np = self._process_hand_pose_data(left_hand_raw_np)
        right_hand_processed_np = self._process_hand_pose_data(right_hand_raw_np)
        
        left_hand_pose = torch.tensor(left_hand_processed_np, dtype=torch.float32)
        right_hand_pose = torch.tensor(right_hand_processed_np, dtype=torch.float32)

        frames = []
        for i in range(N):
            go = global_orient[i].unsqueeze(0).clone()
            go[0, 0] += np.pi  # Rotate 180° around X axis
            bp = body_pose[i].unsqueeze(0)
            lhp = left_hand_pose[i].unsqueeze(0)
            rhp = right_hand_pose[i].unsqueeze(0)
            # Check for NaNs in hand poses and replace with zeros if found
            if torch.isnan(lhp).any() or torch.isnan(rhp).any():
                print(f"Warning: NaN detected in hand poses at frame {i}, using neutral pose")
                lhp = torch.nan_to_num(lhp)
                rhp = torch.nan_to_num(rhp)
            try:
                output = self.smplx_model(
                    body_pose=bp,
                    right_hand_pose=rhp,
                    left_hand_pose=lhp,
                    global_orient=go,
                    betas=torch.zeros((1, 10)),
                    return_verts=True
                )
            except Exception as e:
                print(f"Error in SMPL-X model at frame {i}: {e}")
                output = self.smplx_model(
                    body_pose=bp,
                    right_hand_pose=torch.zeros_like(rhp),
                    left_hand_pose=torch.zeros_like(lhp),
                    global_orient=go,
                    betas=torch.zeros((1, 10)),
                    return_verts=True
                )
            if self.renderer and pyrender and trimesh:
                vertices = output.vertices.detach().cpu().numpy().squeeze()
                mesh = trimesh.Trimesh(vertices=vertices, faces=self.smplx_model.faces)
                scene = pyrender.Scene()
                mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
                scene.add(mesh_pyrender)
                scene.add(self.camera, pose=self.cam_pose)
                scene.add(self.light, pose=self.cam_pose)
                color, _ = self.renderer.render(scene)
                frames.append(color)
            else:
                print("Rendering not available. Returning pose parameters only.")
                return output
        if save_path:
            imageio.mimsave(save_path, frames, fps=fps)
        return frames

def convert_to_cpu(input_path, output_path):
    with open(input_path, "rb") as f:
        data = torch.load(f, map_location='cpu', weights_only=False)
    for k in data:
        if hasattr(data[k], 'cpu'):
            data[k] = data[k].cpu()
    torch.save(data, output_path)

def mirror_pose(pose):
    mirrored = pose.clone()
    mirrored[..., 1::3] *= -1  # Flip Y
    mirrored[..., 2::3] *= -1  # Flip Z
    return mirrored

if __name__ == "__main__":
    print("Word to SMPL-X Animation Generator (Cleaned)")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models")
    dataset_dir = os.path.join(current_dir, "word-level-dataset-cpu")  # Use the CPU-only dataset
    mapping_path = os.path.join(current_dir, "filtered_video_to_gloss.json")
    
    # Load mapping
    with open(mapping_path, "r") as f:
        gloss_map = json.load(f)
        
    # Invert mapping: word -> filename
    word_to_pkl = {v.lower(): k for k, v in gloss_map.items()}
    
    animator = WordToSMPLX(model_path=model_path)
    
    # Print all available words
    print("Available words:")
    print(", ".join(sorted(word_to_pkl.keys())))
    
    word = input("Enter a word (e.g., april, announce): ").strip().lower()
    
    try:
        if word not in word_to_pkl:
            raise ValueError(f"Word '{word}' not found in dataset.")
            
        pkl_file = os.path.join(dataset_dir, word_to_pkl[word])
        pose_data = animator.load_pose_sequence(pkl_file)
        print(f"Loaded pose data for '{word}' from {pkl_file}")
        
        # Debug: Print keys and types
        print("Pose data keys:", pose_data.keys())
        print("global_orient:", type(pose_data.get('global_orient')))
        print("body_pose:", type(pose_data.get('body_pose')))
        print("right_hand_pose:", type(pose_data.get('right_hand_pose')))
        
        # Save animation
        output_dir = os.path.join(current_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{word}_animation.mp4")
        
        animator.render_animation(pose_data, save_path=output_path, fps=15)
        print(f"Animation saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()