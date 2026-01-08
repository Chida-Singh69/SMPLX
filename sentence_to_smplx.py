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
import pickle
import io

# Import pyrender for 3D rendering
try:
    import pyrender
    PYRENDER_AVAILABLE = True
except ImportError:
    PYRENDER_AVAILABLE = False
    pyrender = None

class SentenceToSMPLX:
    """
    Renders SMPL-X animations from sentence-level pose data.
    Similar to WordToSMPLX but optimized for longer sequences.
    """
    def __init__(self, model_path="models", gender='neutral', viewport_width=640, viewport_height=480, device=None):
        # Device setup - use GPU if available, otherwise fallback to CPU
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"[INFO] Using device: {self.device}")
        
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
        ).to(self.device)
        
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        
        # Lazy initialization - only create renderer when actually rendering
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
        
        # Try multiple OpenGL platforms
        platforms_to_try = ['osmesa', 'egl', None]  # None = default platform
        errors = []
        
        for platform in platforms_to_try:
            try:
                # Set platform if specified
                if platform:
                    os.environ['PYOPENGL_PLATFORM'] = platform
                elif 'PYOPENGL_PLATFORM' in os.environ:
                    del os.environ['PYOPENGL_PLATFORM']
                
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
                print(f"[INFO] Pyrender initialized successfully using '{platform if platform else 'default'}' platform")
                return
                
            except Exception as e:
                errors.append(f"{platform if platform else 'default'}: {type(e).__name__}")
                continue
        
        # All platforms failed
        if not hasattr(self, '_pyrender_error_printed'):
            print(f"[INFO] Pyrender unavailable (tried: {', '.join(errors)}), using matplotlib fallback")
            self._pyrender_error_printed = True
        self.renderer_initialized = False
    
    def load_pose_sequence(self, pkl_path):
        """
        Load pose sequence from pickle file.
        Always loads to CPU to ensure compatibility.
        Handles nested CUDA tensors in How2Sign dataset.
        
        Args:
            pkl_path: Path to the pickle file containing pose data
            
        Returns:
            Dictionary containing pose data
        """
        # Custom unpickler to handle nested CUDA tensors
        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                return super().find_class(module, name)
        
        with open(pkl_path, "rb") as f:
            data = CPU_Unpickler(f).load()
            
        # Recursively move all tensors to CPU if they aren't already
        def to_cpu_recursive(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu()
            elif isinstance(obj, dict):
                return {k: to_cpu_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_cpu_recursive(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(to_cpu_recursive(item) for item in obj)
            return obj
            
        return to_cpu_recursive(data)

    def render_animation(self, pose_data, save_path=None, fps=15, max_frames=None):
        """
        Render SMPL-X animation from pose data using pyrender.
        
        Args:
            pose_data: Dictionary containing 'smplx' key with pose parameters
            save_path: Optional path to save the video
            fps: Frames per second for the output video
            max_frames: Optional maximum number of frames to render (for long sentences)
            
        Returns:
            List of rendered frames
        """
        smplx_data = pose_data['smplx']
        
        # Pose Data Loading - stack numpy arrays to shape [N, D]
        if isinstance(smplx_data, np.ndarray) and isinstance(smplx_data[0], np.ndarray):
            smplx_params = np.stack(smplx_data)
            N = smplx_params.shape[0]
            
            # Limit frames if requested (for previewing long sentences)
            if max_frames is not None and N > max_frames:
                print(f"[INFO] Limiting rendering to first {max_frames} of {N} frames")
                smplx_params = smplx_params[:max_frames]
                N = max_frames
            
            # Pose Parameter Extraction
            # Global orient: 3 values controlling overall body rotation
            global_orient = torch.tensor(smplx_params[:, 0:3], dtype=torch.float32, device=self.device)
            # Body pose: 63 values (21 joints × 3 rotation values each)
            body_pose = torch.tensor(smplx_params[:, 3:66], dtype=torch.float32, device=self.device)
            # Left hand pose: 45 values for finger/hand rotation
            left_hand_pose_np = smplx_params[:, 66:111]
            # Right hand pose: 45 values for finger/hand rotation
            right_hand_pose_np = smplx_params[:, 111:156]
            
            # Hand Pose Smoothing - Gaussian filtering to reduce jitter
            left_hand_pose_np = self.smooth_and_clamp_hand_pose(left_hand_pose_np)
            right_hand_pose_np = self.smooth_and_clamp_hand_pose(right_hand_pose_np)
            
            left_hand_pose = torch.tensor(left_hand_pose_np, dtype=torch.float32, device=self.device)
            right_hand_pose = torch.tensor(right_hand_pose_np, dtype=torch.float32, device=self.device)
        else:
            raise ValueError("Unexpected structure in 'smplx' key.")

        print(f"[RENDERING] Rendering {N} frames (sentence-level)...")
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
                    betas=torch.zeros((1, 10), device=self.device),
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
            
            # Progress updates more frequently for long sentences
            if (i + 1) % 20 == 0 or i == N - 1:
                print(f"  Rendered {i + 1}/{N} frames ({(i+1)/N*100:.1f}%)")
        
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
    
    def render_animation_to_bytes(self, pose_data, fps=15, max_frames=None):
        """Render animation directly to bytes without saving to disk."""
        import io
        try:
            frames = self.render_animation(pose_data, save_path=None, fps=fps, max_frames=max_frames)
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


if __name__ == "__main__":
    print("Sentence to SMPL-X Animation Generator")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models")
    dataset_dir = os.path.join(current_dir, "how2sign_pkls_cropTrue_shapeFalse")
    mapping_path = os.path.join(current_dir, "how2sign_mapping.json")
    
    # Load mapping
    with open(mapping_path, "r") as f:
        gloss_map = json.load(f)
    
    animator = SentenceToSMPLX(model_path=model_path)
    
    # Show sample sentences
    sample_sentences = list(gloss_map.items())[:10]
    print("Sample sentences:")
    for i, (pkl, sentence) in enumerate(sample_sentences):
        print(f"{i+1}. {sentence[:80]}...")
    
    # Example rendering
    pkl_file = list(gloss_map.keys())[0]
    sentence = gloss_map[pkl_file]
    print(f"\nRendering: {sentence[:80]}...")
    
    try:
        pkl_path = os.path.join(dataset_dir, pkl_file)
        pose_data = animator.load_pose_sequence(pkl_path)
        print(f"Loaded pose data from {pkl_file}")
        
        # Save animation
        output_dir = os.path.join(current_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "sentence_animation.mp4")
        
        animator.render_animation(pose_data, save_path=output_path, fps=15)
        print(f"\n✅ Animation saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
