# University A100 Deployment Guide

This document is your exact blueprint for securely training the Generative AI (MDM) on your A100 cluster using the 30K dataset.

## Step 1: Uploading the Data

When your 30K dataset is ready, upload the repository to the University cluster. 

Create a folder for your data (e.g., `sign_avatars_30k/`).
Inside this folder, you must have two things:
1. **The 30,000 `.pkl` files** (using the exact same format as `how2sign-trial`).
2. **A JSON Mapping file** (e.g., `mapping.json`).

> [!IMPORTANT]
> The `mapping.json` is strictly required. It must be a dictionary where the keys are the exact filenames of the `.pkl` files, and the values are the English transcripts.
> **Example `mapping.json` Format:**
> ```json
> {
>   "video_001.pkl": "Hello how are you doing today",
>   "video_002.pkl": "I am driving to the store"
> }
> ```

---

## Step 2: The "Zero-Tolerance" Validation Check

Because A100 compute time is valuable and you cannot afford mid-training crashes, you **must** run the strict validation script I wrote before starting the training process.

Run this command on the cluster:
```bash
python validate_dataset.py --data_dir sign_avatars_30k --mapping sign_avatars_30k/mapping.json
```

**What this does:**
It will iterate over all 30,000 files in a few seconds. It mathematically inspects every single PyTorch tensor for `NaN` (Not-a-Number) corruption, checks that the shapes are exactly 182-dimensional or 169-dimensional, and verifies that no files are missing from the `mapping.json`.

- **If it says `[SUCCESS]`:** You are perfectly cleared to train.
- **If it says `[FAIL]`:** It will generate a `quarantine_list.txt` file. You must remove those specific corrupted files from your `mapping.json` before proceeding.

---

## Step 3: Launching the MDM Training

Once the data is validated, you launch the Motion Diffusion Model training using the following command.

```bash
python train_diffusion.py train --pkl_dir sign_avatars_30k --mapping sign_avatars_30k/mapping.json --epochs 500 --batch_size 128
```

### Understanding the Parameters:
- `--epochs 500`: The model needs to see the data multiple times to understand the grammar of ASL. 500 is a standard starting point for diffusion models.
- `--batch_size 128`: The A100 128GB has massive VRAM. Pushing the batch size to 128 or 256 maximizes GPU utilization and drastically speeds up training. If you get an "Out of Memory" (OOM) error, lower this to `64`.

> [!TIP]
> **Monitoring Progress:** The script will output checkpoints (e.g., `checkpoints/mdm_epoch_10.pt`) and will print the `Hand-Weighted Loss`. You want to see this loss steadily decreasing over the days it trains.

---

## Step 4: Generating Novel ASL (Inference)

After training finishes, you can test the AI by feeding it a brand new sentence it has never seen before:

```bash
python train_diffusion.py generate --checkpoint checkpoints/mdm_best.pt --text "This is a brand new sentence." --output_pkl result.pkl
```

This will output a `result.pkl` containing the smooth, generated `[N, 182]` SMPL-X sequence, which you can then pass to your Three.js or Pyrender pipeline.
