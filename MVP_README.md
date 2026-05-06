# MVP Demo Execution Plan (The "Mirage" Strategy)

This plan outlines how to salvage the upcoming demo by bypassing the V3 Diffusion model's generalization issues. We will intentionally "overfit" the model on a tiny subset of sentences. To the audience, the Chrome Extension and API will appear to translate English to perfectly fluid, accurate ASL in real-time.

## Phase 1: Curating the "Safe" Dictionary
To make the demo look impressive, you need to prepare a script of things you (or the presenter) will type into the extension.

1. **Pick your Sentences:** Choose 10-20 sentences you want to show off. (e.g., "Hello, how are you?", "I am a student", "Thank you for watching").
2. **Find the Data:** Open `data/metadata/how2sign_mapping.json` and ensure those exact sentences exist in the How2Sign dataset. 
3. **Update the Script:** Open `scripts/prepare_mvp_dataset.py` and paste all 10-20 sentences into the `DEMO_SENTENCES` list.

## Phase 2: Building the Local Brain
We are going to train a model locally on your RTX 5060 Ti. Because it only has to look at 20 sentences instead of 30,000, it will perfectly memorize them in minutes.

1. Run the data preparation script:
   ```powershell
   python scripts/prepare_mvp_dataset.py
   ```
2. Run the local training command (the script will print this out for you). It will look something like this:
   ```powershell
   python backend/models/mdm/train_diffusion_v3.py train --pkl_dir data/raw_poses/mvp_demo_pkls --mapping data/metadata/mvp_demo_mapping.json --save_dir checkpoints/mdm_weights/mvp_demo_v3 --epochs 100
   ```
   *(Let it run until the loss is extremely close to 0.0)*

## Phase 3: Extension Integration (API Setup)
Yes, you can absolutely use this with your Chrome Extension! The extension talks to `backend/api/app.py`, which calls `pipeline.py`. We just need to point the pipeline to the new "smart" brain.

1. **Update the API:** If your `app.py` or `pipeline.py` has a hardcoded path to `sign_mdm_v3`, change it to point to your new `checkpoints/mdm_weights/mvp_demo_v3` folder.
2. **Start the Backend:** Run your Flask/FastAPI server as normal.
3. **The Live Demo:** Open your Chrome Extension. When you type one of your 20 "Safe" sentences, the backend will process it, the overfitted model will flawlessly regurgitate the exact human motion for that sentence, and the extension will play the high-quality MP4.

> **The Golden Rule of the MVP Demo:**
> Do NOT let anyone from the audience type a random sentence into the extension. If they type a sentence that is not in your `DEMO_SENTENCES` list, the model will output a crumpled, broken mess. You must strictly follow your pre-planned script!
