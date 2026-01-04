import os
import sys
import torch
import smplx
import numpy as np
import imageio
import json
from scipy.ndimage import gaussian_filter1d
import trimesh
from PIL import Image, ImageDraw

# Import pyrender for 3D rendering
try:
    import pyrender
    PYRENDER_AVAILABLE = True
except ImportError:
    PYRENDER_AVAILABLE = False
    pyrender = None

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
        # Locate SMPL-X files (support either models/smplx/*.npz or models/smplx/smplx/*.npz)
        primary_root = model_path  # expected: models
        primary_dir = os.path.join(primary_root, 'smplx')  # expected: models/smplx
        primary_file = os.path.join(primary_dir, f"SMPLX_{gender.upper()}.npz")

        nested_root = primary_dir  # fallback: models/smplx
        nested_dir = os.path.join(nested_root, 'smplx')  # fallback: models/smplx/smplx
        nested_file = os.path.join(nested_dir, f"SMPLX_{gender.upper()}.npz")

        if os.path.exists(primary_file):
            model_root = primary_root  # smplx.create will look in models/smplx
            model_file = primary_file
        elif os.path.exists(nested_file):
            model_root = nested_root  # smplx.create will look in models/smplx/smplx
            model_file = nested_file
        else:
            raise ValueError(
                f"Model file not found. Checked: {primary_file} and {nested_file}. "
                "Place SMPL-X .npz/.pkl files under models/smplx/."
            )
        
        self.smplx_model = smplx.create(
            model_path=model_root,
            model_type='smplx',
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
        
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        
        # Lazy initialization - only create renderer when actually rendering
        # This avoids slow/failed initialization at startup
        self.camera = None
        self.light = None
        self.cam_pose = None
        self.renderer = None
        self.renderer_initialized = False
        self._renderer_init_attempted = False

    @staticmethod
    def smooth_and_clamp_hand_pose(hand_pose_np, sigma=0.3, clamp_min=-2.0, clamp_max=2.0):
        """
        Smoothes and clamps hand pose parameters.
        
        Args:
            hand_pose_np: numpy array of shape [N, 45] containing hand pose parameters
            sigma: Gaussian filter sigma for smoothing
            clamp_min: Minimum value for clamping
            clamp_max: Maximum value for clamping
            
        Returns:
            Smoothed and clamped hand pose array
        """
        # Smoothing
        smoothed = np.copy(hand_pose_np)
        if smoothed.shape[0] > 1:  # Need at least 2 frames to smooth
            for i in range(smoothed.shape[1]):
                smoothed[:, i] = gaussian_filter1d(smoothed[:, i], sigma=sigma, mode='nearest')
        
        # Clamping
        smoothed = np.clip(smoothed, clamp_min, clamp_max)
        return smoothed

    def _init_pyrender_lazy(self):
        """Initialize pyrender only when actually needed (lazy initialization)."""
        if self._renderer_init_attempted or not PYRENDER_AVAILABLE:
            return
        
        self._renderer_init_attempted = True
        try:
            # Camera: Perspective camera with 36-degree vertical field of view (π/5 radians)
            self.camera = pyrender.PerspectiveCamera(yfov=np.pi / 5.0)
            
            # Light: Directional light with full white color and intensity 2.0
            self.light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
            
            # Camera pose: 4x4 transformation matrix
            # Position for frontal standing view with full body and hands visible
            self.cam_pose = np.eye(4)
            self.cam_pose[0, 3] = 0.0      # x: center (no horizontal offset)
            self.cam_pose[1, 3] = 0.2      # y: slightly above center for better hand view
            self.cam_pose[2, 3] = 2.5      # z: further away to avoid hand clipping
            
            # Offscreen renderer: renders to images (not screen display)
            self.renderer = pyrender.OffscreenRenderer(
                viewport_width=self.viewport_width,
                viewport_height=self.viewport_height
            )
            self.renderer_initialized = True
        except Exception as e:
            if not hasattr(self, '_pyrender_error_printed'):
                print(f"[INFO] Pyrender unavailable ({type(e).__name__}), using matplotlib fallback")
                self._pyrender_error_printed = True
            self.renderer_initialized = False
    
    def load_pose_sequence(self, pkl_path):
        """
        Load pose sequence from pickle file.
        Always loads to CPU to ensure compatibility.
        
        Args:
            pkl_path: Path to the pickle file containing pose data
            
        Returns:
            Dictionary containing pose data
        """
        with open(pkl_path, "rb") as f:
            data = torch.load(f, map_location='cpu', weights_only=False)
        return data

    def render_animation(self, pose_data, save_path=None, fps=15):
        """
        Render SMPL-X animation from pose data using pyrender.
        
        Args:
            pose_data: Dictionary containing 'smplx' key with pose parameters
            save_path: Optional path to save the video
            fps: Frames per second for the output video
            
        Returns:
            List of rendered frames
        """
        smplx_data = pose_data['smplx']
        
        # Pose Data Loading - stack numpy arrays to shape [N, D]
        if isinstance(smplx_data, np.ndarray) and isinstance(smplx_data[0], np.ndarray):
            smplx_params = np.stack(smplx_data)
            N = smplx_params.shape[0]
            
            # Pose Parameter Extraction
            # Global orient: 3 values controlling overall body rotation
            global_orient = torch.tensor(smplx_params[:, 0:3], dtype=torch.float32)
            # Body pose: 63 values (21 joints × 3 rotation values each)
            body_pose = torch.tensor(smplx_params[:, 3:66], dtype=torch.float32)
            # Left hand pose: 45 values for finger/hand rotation
            left_hand_pose_np = smplx_params[:, 66:111]
            # Right hand pose: 45 values for finger/hand rotation
            right_hand_pose_np = smplx_params[:, 111:156]
            
            # Hand Pose Smoothing - Gaussian filtering to reduce jitter
            left_hand_pose_np = self.smooth_and_clamp_hand_pose(left_hand_pose_np)
            right_hand_pose_np = self.smooth_and_clamp_hand_pose(right_hand_pose_np)
            
            left_hand_pose = torch.tensor(left_hand_pose_np, dtype=torch.float32)
            right_hand_pose = torch.tensor(right_hand_pose_np, dtype=torch.float32)
        else:
            raise ValueError("Unexpected structure in 'smplx' key.")

        print(f"[RENDERING] Rendering {N} frames...")
        frames = []
        
        # Frame Generation Loop
        for i in range(N):
            # Add 180-degree rotation around X-axis to flip model right-side up
            go = global_orient[i].unsqueeze(0).clone()
            go[0, 0] += np.pi
            
            bp = body_pose[i].unsqueeze(0)
            lhp = left_hand_pose[i].unsqueeze(0)
            rhp = right_hand_pose[i].unsqueeze(0)
            
            # Error Handling - Replace NaN values with neutral (zero) poses
            if torch.isnan(lhp).any() or torch.isnan(rhp).any():
                print(f"Warning: NaN in hand poses at frame {i}, using neutral pose")
                lhp = torch.zeros_like(lhp)
                rhp = torch.zeros_like(rhp)
            
            # Call SMPL-X model with pose parameters
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
                # Fallback: regenerate frame with neutral hand poses
                print(f"Warning: SMPL-X error at frame {i}, using neutral hands: {e}")
                output = self.smplx_model(
                    body_pose=bp,
                    right_hand_pose=torch.zeros_like(rhp),
                    left_hand_pose=torch.zeros_like(lhp),
                    global_orient=go,
                    betas=torch.zeros((1, 10)),
                    return_verts=True
                )
            
            # Get 3D vertex positions
            vertices = output.vertices.detach().cpu().numpy().squeeze()
            
            # Create trimesh from vertices and SMPL-X face connectivity
            mesh = trimesh.Trimesh(vertices=vertices, faces=self.smplx_model.faces)
            
            # Render frame using pyrender (primary) or trimesh (fallback)
            frame = self._render_pyrender_frame(mesh)
            if frame is not None:
                frames.append(frame)
            
            if (i + 1) % 10 == 0:
                print(f"  Rendered {i + 1}/{N} frames")
        
        print(f"✓ Rendering complete!")
        
        # Video Encoding - save to MP4 using imageio
        if save_path and frames:
            print(f"[SAVING] Saving video to {save_path}...")
            imageio.mimsave(save_path, frames, fps=fps)
            print(f"[SUCCESS] Video saved!")
            
        return frames
    
    def _render_pyrender_frame(self, mesh):
        """
        Render a single frame using pyrender.
        
        Args:
            mesh: trimesh.Trimesh object with vertices and faces
            
        Returns:
            np.ndarray: RGB image (640x480x3) or None if rendering fails
        """
        # Initialize pyrender on first render (lazy)
        if not self._renderer_init_attempted:
            self._init_pyrender_lazy()
        
        if not PYRENDER_AVAILABLE or not self.renderer_initialized or self.renderer is None:
            return self._render_trimesh_frame_fallback(mesh)
        
        try:
            # Create pyrender Mesh from trimesh
            pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
            
            # Create scene and add components
            scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0])  # White background
            scene.add(pyrender_mesh)
            scene.add(self.camera, pose=self.cam_pose)
            scene.add(self.light, pose=self.cam_pose)
            
            # Render to offscreen buffer
            color, depth = self.renderer.render(scene)
            
            # Return RGB image (discard alpha channel if present)
            if color.shape[2] == 4:
                return color[:, :, :3]
            return color
            
        except Exception as e:
            # Only print once to avoid spam
            if not hasattr(self, '_render_error_printed'):
                print(f"[INFO] Pyrender rendering unavailable, using matplotlib fallback")
                self._render_error_printed = True
            return self._render_trimesh_frame_fallback(mesh)
    
    def render_animation_to_bytes(self, pose_data, fps=15):
        """Render animation directly to bytes without saving to disk."""
        import io
        try:
            frames = self.render_animation(pose_data, save_path=None, fps=fps)
            buffer = io.BytesIO()
            imageio.mimsave(buffer, frames, fps=fps, format='mp4')
            buffer.seek(0)
            video_bytes = buffer.getvalue()
            
            # Clean up frames from memory
            del frames
            buffer.close()
            
            # Clean up renderer to prevent OpenGL context corruption
            if self.renderer is not None:
                try:
                    self.renderer.delete()
                except:
                    pass
                self.renderer = None
                self.renderer_initialized = False
                self._renderer_init_attempted = False
            
            return video_bytes
        except Exception as e:
            # Clean up on error
            if self.renderer is not None:
                try:
                    self.renderer.delete()
                except:
                    pass
                self.renderer = None
                self.renderer_initialized = False
                self._renderer_init_attempted = False
            raise
    
    def _render_trimesh_frame_fallback(self, mesh):
        """Render a frame using matplotlib 3D plotting as fallback."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from io import BytesIO
            
            # Create figure
            fig = plt.figure(figsize=(self.viewport_width/100, self.viewport_height/100), dpi=100)
            ax = fig.add_subplot(111, projection='3d')
            
            # Get vertices
            vertices = mesh.vertices
            faces = mesh.faces
            
            # Plot the mesh
            ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                           triangles=faces, color='lightblue', alpha=0.8, edgecolor='none')
            
            # Set viewing angle for frontal standing view
            ax.view_init(elev=0, azim=0)  # Front view, no tilt
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_box_aspect([1,1,1])
            
            # Remove axes for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.grid(False)
            ax.set_facecolor('white')
            fig.patch.set_facecolor('white')
            
            # Render to buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, facecolor='white')
            buf.seek(0)
            
            # Load as array
            img = Image.open(buf)
            img = img.resize((self.viewport_width, self.viewport_height))
            frame = np.array(img.convert('RGB'))
            
            plt.close(fig)
            buf.close()
            
            return frame
            
        except Exception as e:
            if not hasattr(self, '_fallback_error_printed'):
                print(f"[WARNING] Matplotlib fallback also failed: {e}")
                print("[INFO] Using simple placeholder")
                self._fallback_error_printed = True
            
            # Ultimate fallback: simple colored frame
            img = Image.new('RGB', (self.viewport_width, self.viewport_height), color=(240, 240, 240))
            draw = ImageDraw.Draw(img)
            text = "3D Rendering"
            draw.text((self.viewport_width//2 - 50, self.viewport_height//2), text, fill=(100, 100, 100))
            return np.array(img)
    
def convert_to_cpu(input_path, output_path):
    """
    Convert a pickle file from CUDA to CPU-compatible format.
    
    Args:
        input_path: Path to input pickle file
        output_path: Path to save CPU-compatible pickle file
    """
    with open(input_path, "rb") as f:
        data = torch.load(f, map_location='cpu', weights_only=False)
    for k in data:
        if hasattr(data[k], 'cpu'):
            data[k] = data[k].cpu()
    torch.save(data, output_path)


def mirror_pose(pose):
    """
    Mirror a pose by flipping Y and Z axes.
    
    Args:
        pose: Torch tensor containing pose parameters
        
    Returns:
        Mirrored pose tensor
    """
    mirrored = pose.clone()
    mirrored[..., 1::3] *= -1  # Flip Y
    mirrored[..., 2::3] *= -1  # Flip Z
    return mirrored


if __name__ == "__main__":
    print("Word to SMPL-X Animation Generator")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models")
    dataset_dir = os.path.join(current_dir, "word-level-dataset-cpu")
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
    
    word = input("\nEnter a word (e.g., april, announce): ").strip().lower()
    
    try:
        if word not in word_to_pkl:
            raise ValueError(f"Word '{word}' not found in dataset.")
            
        pkl_file = os.path.join(dataset_dir, word_to_pkl[word])
        pose_data = animator.load_pose_sequence(pkl_file)
        print(f"Loaded pose data for '{word}' from {pkl_file}")
        
        # Debug: Print keys and types
        print("Pose data keys:", pose_data.keys())
        print("Type of smplx data:", type(pose_data.get('smplx')))
        
        # Save animation
        output_dir = os.path.join(current_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{word}_animation.mp4")
        
        animator.render_animation(pose_data, save_path=output_path, fps=15)
        print(f"\n✅ Animation saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()