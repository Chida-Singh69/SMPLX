import os
import json
import shutil

# --- CONFIGURATION ---
# Put the exact sentences you plan to type during your MVP demo here.
# Make sure these sentences ACTUALLY exist in your merged_how2sign_mapping.json
DEMO_SENTENCES = [
    "i am really sorry",
    "hello how are you",
    "i am a student"
]

ORIGINAL_MAPPING = "data/metadata/how2sign_mapping.json"
ORIGINAL_PKL_DIR = "data/raw_poses/how2sign_pkls_cropTrue_shapeFalse"

MVP_MAPPING_OUT = "data/metadata/mvp_demo_mapping.json"
MVP_PKL_DIR = "data/raw_poses/mvp_demo_pkls"
# ---------------------

def main():
    print("=== Building MVP Overfit Dataset ===")
    
    # 1. Load original mapping
    with open(ORIGINAL_MAPPING, 'r') as f:
        mapping = json.load(f)
        
    # Reverse mapping (text -> filename) to find the PKLs
    # Note: If multiple PKLs have the same text, it just grabs the first one it finds
    text_to_pkl = {}
    for pkl_file, text in mapping.items():
        clean_text = text.lower().strip().replace(".", "").replace(",", "")
        text_to_pkl[clean_text] = pkl_file

    os.makedirs(MVP_PKL_DIR, exist_ok=True)
    
    mvp_mapping = {}
    found_count = 0
    
    # 2. Find and copy the requested files
    for sentence in DEMO_SENTENCES:
        clean_sentence = sentence.lower().strip()
        
        if clean_sentence in text_to_pkl:
            pkl_filename = text_to_pkl[clean_sentence]
            src_path = os.path.join(ORIGINAL_PKL_DIR, pkl_filename)
            dst_path = os.path.join(MVP_PKL_DIR, pkl_filename)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                mvp_mapping[pkl_filename] = sentence
                found_count += 1
                print(f"[SUCCESS] Found data for: '{sentence}'")
            else:
                print(f"[ERROR] PKL file missing: {src_path}")
        else:
            print(f"[ERROR] Sentence not found in original dataset: '{sentence}'")
            print(f"        (Try checking the exact wording in your mapping JSON)")

    # 3. Save the new tiny mapping file
    with open(MVP_MAPPING_OUT, 'w') as f:
        json.dump(mvp_mapping, f, indent=4)
        
    print(f"\n=== Done! Copied {found_count}/{len(DEMO_SENTENCES)} sentences. ===")
    print(f"MVP Dataset located at: {MVP_PKL_DIR}")
    print(f"MVP Mapping located at: {MVP_MAPPING_OUT}")
    
    if found_count > 0:
        print("\nTo perfectly memorize these for your demo, run this training command:")
        print(f"python backend/models/mdm/train_diffusion_v3.py train \\")
        print(f"    --pkl_dir {MVP_PKL_DIR} \\")
        print(f"    --mapping {MVP_MAPPING_OUT} \\")
        print(f"    --save_dir checkpoints/mdm_weights/mvp_demo_v3 \\")
        print(f"    --epochs 50  <-- (50 epochs is enough for it to perfectly memorize 3 sentences)")

if __name__ == "__main__":
    main()
