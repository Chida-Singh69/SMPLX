import os
import torch
import pickle
import numpy as np
import re
from sentence_to_smplx import SentenceToSMPLX

class PoseAssembler:
    def __init__(self, poses_root):
        self.poses_root = poses_root

    def list_folders(self):
        """List all sequence folders in the poses directory."""
        if not os.path.exists(self.poses_root):
            return []
        folders = [d for d in os.listdir(self.poses_root) if os.path.isdir(os.path.join(self.poses_root, d))]
        return sorted(folders)

    def assemble_sequence(self, folder_name):
        """
        Assemble a sequence of SMPL-X parameters from a folder of per-frame pickles.
        """
        folder_path = os.path.join(self.poses_root, folder_name)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # List all .pkl files
        pkl_files = [f for f in os.listdir(folder_path) if f.endswith('_3D.pkl')]
        
        # Sort files numerically by frame index
        # Expecting format: name_frameindex_3D.pkl
        def get_frame_index(filename):
            match = re.search(r'_(\d+)_3D\.pkl$', filename)
            return int(match.group(1)) if match else 0

        pkl_files.sort(key=get_frame_index)

        all_params = []
        print(f"[ASSEMBLER] Loading {len(pkl_files)} frames from {folder_name}...")

        for i, f in enumerate(pkl_files):
            pkl_path = os.path.join(folder_path, f)
            try:
                with open(pkl_path, 'rb') as pf:
                    # Try standard pickle since my test showed it's likely standard pickle
                    try:
                        data = pickle.load(pf)
                    except:
                        # Fallback to torch.load
                        pf.seek(0)
                        data = torch.load(pf, map_location='cpu', weights_only=False)

                # Extract and concatenate the main pose parameters
                # Expected keys from my inspection: 
                # smplx_root_pose (3,), smplx_body_pose (63,), smplx_lhand_pose (45,), smplx_rhand_pose (45,)
                root = data.get('smplx_root_pose', np.zeros(3))
                body = data.get('smplx_body_pose', np.zeros(63))
                lhand = data.get('smplx_lhand_pose', np.zeros(45))
                rhand = data.get('smplx_rhand_pose', np.zeros(45))
                
                # Check shapes and flatten if necessary
                def ensure_flat(arr):
                    if hasattr(arr, 'cpu'): arr = arr.cpu().numpy()
                    return arr.flatten()

                combined = np.concatenate([
                    ensure_flat(root),
                    ensure_flat(body),
                    ensure_flat(lhand),
                    ensure_flat(rhand)
                ])
                
                all_params.append(combined)
            except Exception as e:
                print(f"  [WARNING] Error loading frame {f}: {e}")

        if not all_params:
            raise ValueError(f"No frames could be loaded from {folder_path}")

        # Stack into (N, 156) array
        smplx_params = np.vstack(all_params)
        
        # Create dictionary compatible with SentenceToSMPLX/WordToSMPLX
        pose_data = {
            'smplx': smplx_params,
            'fps': 15,
            'gender': 'neutral' # Default
        }
        
        return pose_data

def render_pose_folder(folder_name, poses_root="poses", output_path=None, gender='neutral'):
    """Helper to assemble and render a folder from the 'poses' directory."""
    assembler = PoseAssembler(poses_root)
    pose_data = assembler.assemble_sequence(folder_name)
    pose_data['gender'] = gender
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models")
    
    # Use SentenceToSMPLX for advanced rendering
    renderer = SentenceToSMPLX(model_path=model_path, gender=gender)
    
    if output_path is None:
        os.makedirs("output", exist_ok=True)
        output_path = os.path.join("output", f"pose_{folder_name}.mp4")
        
    renderer.render_animation(pose_data, save_path=output_path)
    print(f"[SUCCESS] Animation saved to {output_path}")
    return output_path

if __name__ == "__main__":
    # Small CLI for standalone use
    import sys
    poses_dir = os.path.join(os.getcwd(), "poses")
    assembler = PoseAssembler(poses_dir)
    folders = assembler.list_folders()
    
    if not folders:
        print(f"No pose folders found in {poses_dir}")
        sys.exit(0)
        
    print("Available pose folders:")
    for idx, f in enumerate(folders[:20]):
        print(f"{idx}: {f}")
    if len(folders) > 20: 
        print(f"... and {len(folders)-20} more.")
        
    try:
        choice = int(input("\nEnter folder index to render: "))
        selected = folders[choice]
        render_pose_folder(selected, poses_dir)
    except Exception as e:
        print(f"Error: {e}")
