import os
import pickle
from poses_to_animation import PoseAssembler

def convert_all():
    poses_dir = "poses"
    out_dir = "how2sign_pkls_full"
    
    print(f"Creating output directory: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    
    assembler = PoseAssembler(poses_dir)
    folders = assembler.list_folders()
    
    print(f"Found {len(folders)} pose folders to convert.")
    
    success_count = 0
    fail_count = 0
    
    for i, folder in enumerate(folders):
        try:
            # assemble_folder returns {"smplx": np.ndarray [N, 156], "fps": 15}
            data = assembler.assemble_folder(folder)
            
            # Save it as a single pkl file
            out_file = os.path.join(out_dir, folder + ".pkl")
            with open(out_file, "wb") as f:
                pickle.dump(data, f)
            
            success_count += 1
            if (i+1) % 500 == 0:
                print(f"Processed {i+1}/{len(folders)} folders...")
                
        except Exception as e:
            print(f"Failed to process {folder}: {e}")
            fail_count += 1

    print(f"\nConversion complete!")
    print(f"Successfully converted: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Your VAE training data is now in: {out_dir}/")
    print(f"\nTo train the VAE, run:")
    print(f"python vae_train.py --pkl-dir {out_dir} --pose-mode pose156 --epochs 200")

if __name__ == "__main__":
    convert_all()
