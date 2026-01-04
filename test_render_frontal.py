#!/usr/bin/env python
"""Test frontal rendering with updated camera."""

import os
from word_to_smplx import WordToSMPLX

def main():
    print("[TEST] Testing frontal standing view rendering...")
    animator = WordToSMPLX(gender='neutral')
    
    # Load first pose file
    pose_dir = 'word-level-dataset-cpu-fixed'
    pose_files = [f for f in os.listdir(pose_dir) if f.endswith('.pkl')][:1]
    
    if pose_files:
        first_file = os.path.join(pose_dir, pose_files[0])
        print(f'[TEST] Loading: {first_file}')
        
        # Load and render
        pose_data = animator.load_pose_sequence(first_file)
        num_frames = len(pose_data['smplx'])
        print(f'[OK] Loaded {num_frames} frames')
        
        # Render with new camera view
        output_path = 'test_frontal_view.mp4'
        print(f'[TEST] Rendering to {output_path} with frontal camera view...')
        frames = animator.render_animation(pose_data, save_path=output_path, fps=15)
        print(f'[SUCCESS] Rendered {len(frames)} frames')
        
        if os.path.exists(output_path):
            size = os.path.getsize(output_path) / (1024 * 1024)
            print(f'[OK] Video file: {size:.2f} MB')
            print(f'[INFO] Check the video to see the frontal standing view')

if __name__ == '__main__':
    main()
