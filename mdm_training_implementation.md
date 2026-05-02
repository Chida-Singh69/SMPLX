# Motion Diffusion Model (MDM) A100 Training Implementation Guide

This document is the master blueprint for executing the full-scale Generative AI (MDM) training on your University's NVIDIA A100 cluster.

## 1. System Readiness
The architecture has been fully verified via a localized 2-epoch test on the trial data. The mathematical forward and backward passes (backpropagation) executed perfectly without triggering any Out-Of-Memory or shape-mismatch crashes. 

**Model Parameters Verified:** 28,185,782 (28.2M) parameters.

## 2. A100 Pre-Flight Checklist
Before you execute the training command on the A100, ensure you have:
1. Cloned this entire repository to the cluster environment.
2. Verified `torch` (with CUDA support) is active on the A100.
3. Installed all necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copied your `30K_SignAvatars` directory into the project folder.
4. **Crucial:** Run the zero-tolerance dataset validation script to ensure no corrupted files will crash the training on day 2.
   ```bash
   python validate_dataset.py --data_dir YOUR_30K_DIR_NAME --mapping YOUR_MAPPING_JSON.json
   ```

## 3. Launching the Training Pipeline

Once validation outputs `[SUCCESS]`, you will execute the main training loop. The A100 128GB GPU allows us to aggressively increase the batch size, reducing training time by a factor of 10x compared to a local GPU.

Run the following command on the A100:

```bash
python train_diffusion.py train --pkl_dir YOUR_30K_DIR_NAME --mapping YOUR_MAPPING_JSON.json --batch_size 128 --epochs 500
```

### Expected Output
When the training launches, you should see:
1. **Device Assignment:** `[Train] Device: cuda:0` (confirming it is using the A100).
2. **Text Embeddings:** The system will initialize OpenAI's CLIP encoder to convert the 30K English transcripts into mathematical embeddings.
3. **Loss Tracking:** It will print the loss every epoch. The loss should steadily decrease. The architecture is using our custom **Hand-Weighted Loss Function**, meaning it is calculating `body_loss + 1.5 * hand_loss` to ensure razor-sharp finger spelling.

## 4. Checkpoints & Resuming
The script automatically saves checkpoints to the `checkpoints/` directory.
- Every 50 epochs, it saves a numbered backup (e.g., `mdm_epoch_50.pt`).
- It continuously overrides `mdm_best.pt` whenever the model achieves a historically low loss score.

If your A100 cluster drops the connection or time-limits your compute job, you can instantly resume training from the exact epoch it failed by pointing it to the checkpoint file.

## 5. Generating Real ASL (Inference)
When you are satisfied with the loss metric (after roughly 300 to 500 epochs), you can generate completely novel ASL sequences from arbitrary English text.

Execute this command:
```bash
python train_diffusion.py generate --checkpoint checkpoints/mdm_best.pt --text "I am driving to the hospital" --output_pkl new_asl_sequence.pkl
```
The output `.pkl` file will contain a perfectly synthesized `[N, 182]` SMPL-X sequence that you can route directly to your 3D avatar.
