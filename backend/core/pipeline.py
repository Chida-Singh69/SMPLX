import os
import sys
import argparse
import subprocess
import re

def sanitize_filename(text):
    # Keep only alphanumeric and spaces, then replace spaces with underscores
    clean = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return clean.strip().replace(' ', '_').lower()

def main():
    parser = argparse.ArgumentParser(description="End-to-End Generate and Render Pipeline")
    parser.add_argument("--text", type=str, required=True, help="The sentence to translate to ASL")
    parser.add_argument("--version", type=str, choices=['v1', 'v2', 'v3'], default='v3', help="Model version to use")
    parser.add_argument("--model_dir", type=str, default=None, help="Path to checkpoints (overrides version default)")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames. Auto-calculated if not provided.")
    args = parser.parse_args()

    # Set defaults based on version
    if args.version == 'v1':
        script_name = "backend/models/mdm/train_diffusion.py"
        default_model_dir = "checkpoints/mdm_weights/sign_mdm_v1"
        out_dir_gen = "data/cache/generated"
    elif args.version == 'v2':
        script_name = "backend/models/mdm/train_diffusion_v2.py"
        default_model_dir = "checkpoints/mdm_weights/checkpoints_v2/sign_mdm_v2"
        out_dir_gen = "data/cache/generated_v2"
    else:
        script_name = "backend/models/mdm/train_diffusion_v3.py"
        default_model_dir = "checkpoints/mdm_weights/checkpoints_v3/sign_mdm_v3"
        out_dir_gen = "data/cache/generated_v3"
        
    model_dir = args.model_dir if args.model_dir else default_model_dir

    # Dynamic frame calculation (roughly 1 second per word + 1 second padding)
    if args.frames is None:
        word_count = len(args.text.split())
        frames = max(30, word_count * 15 + 15)
        print(f"[*] Auto-calculated length: {frames} frames ({frames/15.0:.1f} seconds) for {word_count} words.")
    else:
        frames = args.frames

    # 1. Setup paths
    safe_name = sanitize_filename(args.text)
    if not safe_name:
        safe_name = "output"
        
    os.makedirs(out_dir_gen, exist_ok=True)
    os.makedirs("data/mp4_outputs", exist_ok=True)
    
    final_pkl = os.path.join(out_dir_gen, f"{safe_name}.pkl")
    final_mp4 = os.path.join("data", "mp4_outputs", f"{safe_name}_{args.version}.mp4")

    print(f"\n=========================================================")
    print(f" PIPELINE START: '{args.text}' [{args.version.upper()}]")
    print(f"=========================================================")

    # 2. Run Generation
    print(f"\n[1/3] Generating 3D motion using {args.version.upper()} model...")
    gen_cmd = [
        sys.executable, script_name, "generate",
        "--model_dir", model_dir,
        "--text", args.text,
        "--num_frames", str(frames),
        "--output_dir", out_dir_gen
    ]
    try:
        subprocess.run(gen_cmd, check=True)
    except subprocess.CalledProcessError:
        print("[Error] Generation failed. Did you specify the correct --model_dir?")
        sys.exit(1)

    # 3. Rename the output file to the text prompt
    temp_pkl = os.path.join(out_dir_gen, "generated_motion.pkl")
    if os.path.exists(temp_pkl):
        if os.path.exists(final_pkl):
            os.remove(final_pkl) # Overwrite if exists
        os.rename(temp_pkl, final_pkl)
        print(f"[2/3] Saved motion data to: {final_pkl}")
    else:
        print(f"[Error] Could not find the expected output {temp_pkl}")
        sys.exit(1)

    # 4. Render Video (render_preview_video.py)
    print(f"\n[3/3] Rendering SMPL-X Avatar MP4...")
    ren_cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "render_preview_video.py"),
        "--input", final_pkl,
        "--out", final_mp4,
        "--fps", "15",
        "--text", args.text
    ]
    try:
        subprocess.run(ren_cmd, check=True)
        print(f"\n=========================================================")
        print(f" SUCCESS! Video saved to: {final_mp4}")
        print(f"=========================================================\n")
    except subprocess.CalledProcessError:
        print("[Error] Rendering failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
