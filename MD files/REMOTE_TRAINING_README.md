# Remote A100 Training (VS Code Remote-SSH) — SMPLX Diffusion

This repo is used to train a text-conditioned diffusion model that generates SMPL-X sign-language motion.

You are working from **Windows** and running training on a **remote Ubuntu 22.04 + A100** machine via **VS Code Remote-SSH**.

> Note: Do not share server IP/user details publicly. This file is meant for your personal workflow.

---

## 0) High-level workflow

1. Upload only the files needed for diffusion training (skip local venvs).
2. Create a clean Python venv **on the server** and install deps.
3. Verify GPU/CUDA.
4. Run training inside `tmux` so it continues after disconnect.
5. Copy checkpoints back to Windows.

---

## 1) Repo purpose and key files

### Training entrypoint
- `train_diffusion.py`
  - **Train:** `python train_diffusion.py train --pkl_dir <DIR> --mapping <JSON> ...`
  - **Generate:** `python train_diffusion.py generate --model_dir <DIR> --text "..." ...`

### Dataset loader
- `sign_language_dataset.py`
  - Expects PKLs containing `{'smplx': np.ndarray [T, 182]}` (or equivalent) per sequence.
  - Uses a **mapping JSON** of `pkl_filename -> English sentence`.
  - Computes/uses normalization stats; it may write `norm_stats.npz` inside the `--pkl_dir`.

### Dependency list
- `requirements.txt`
  - Includes training deps plus some app/demo deps (Streamlit/Flask/etc). Installing all is simplest.

### Typical mapping
- `merged_how2sign_mapping.json` (example mapping used with How2Sign PKLs)

---

## 2) Minimal folder layout (what you need)

### Local (Windows) — source repo
Example: `D:\Chida\Projects\SMPLX\`

Keep (minimum for training):
- `train_diffusion.py`
- `sign_language_dataset.py`
- `requirements.txt`
- `merged_how2sign_mapping.json` (or the mapping you plan to use)
- `how2sign_pkls_cropTrue_shapeFalse/` (or the folder you pass as `--pkl_dir`)

Skip (local-only / huge / reproducible):
- `venv_py311_gpu/`, `venv_py313/` (local venvs)
- `output/`, `generated/` (optional; you can regenerate)
- any other big folders you don’t need for training

### Remote (Ubuntu server)
Target structure:
- `~/SMPLX/`
  - `train_diffusion.py`
  - `sign_language_dataset.py`
  - `requirements.txt`
  - `merged_how2sign_mapping.json`
  - `how2sign_pkls_cropTrue_shapeFalse/`
  - `checkpoints/`
  - `.venv/`

---

## 3) SSH + VS Code Remote-SSH

### Verify SSH works (Windows PowerShell)
```powershell
ssh -i $env:USERPROFILE\.ssh\a100_user02 user02@14.143.127.114
```
If you see the Ubuntu welcome message, key auth is working.

### Connect VS Code
1. Install extension: **Remote - SSH** (Microsoft)
2. `Ctrl+Shift+P` → **Remote-SSH: Connect to Host...**
3. Connect to: `user02@14.143.127.114`
4. Choose **Linux** when prompted
5. In the remote window: **Open Folder** → open your home directory (`/home/user02`) or `~/SMPLX` after upload

---

## 4) Upload (copy) only what you need

You said you are **currently copying PKL files** for training.

### Create remote project dir
```powershell
ssh user02@14.143.127.114 "mkdir -p ~/SMPLX"
```

### Upload code + mapping + requirements
Run from your local repo folder on Windows:
```powershell
cd D:\Chida\Projects\SMPLX
scp requirements.txt train_diffusion.py sign_language_dataset.py merged_how2sign_mapping.json user02@14.143.127.114:~/SMPLX/
```

### Upload the dataset folder (PKLs)
```powershell
cd D:\Chida\Projects\SMPLX
scp -C -r how2sign_pkls_cropTrue_shapeFalse user02@14.143.127.114:~/SMPLX/
```

> Tip: If the dataset is extremely large, `scp` can be slow. Once SSH is stable, consider `rsync` (requires WSL or an rsync client on Windows).

---

## 5) Remote environment setup (Ubuntu)

Open the **remote** VS Code terminal (or SSH terminal) and run:

```bash
cd ~/SMPLX
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
pip install -r requirements.txt
```

### Verify GPU
```bash
nvidia-smi
```

### Verify PyTorch CUDA
```bash
python -c "import torch; print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

If `cuda: False`, fix PyTorch install before training (install a CUDA-enabled wheel).

---

## 6) Run training safely (tmux)

Use `tmux` so training continues even if VS Code or your internet disconnects.

### Start tmux session
```bash
tmux new -s train
```

### Run training (example)
```bash
cd ~/SMPLX
source .venv/bin/activate
python train_diffusion.py train \
  --pkl_dir how2sign_pkls_cropTrue_shapeFalse \
  --mapping merged_how2sign_mapping.json \
  --save_dir checkpoints/sign_mdm_v1 \
  --batch_size 64 \
  --epochs 200
```

### Detach / Reattach
- Detach: `Ctrl+b` then `d`
- Reattach later:
```bash
tmux attach -t train
```

---

## 7) Copy checkpoints back to Windows

Default save dir is `checkpoints/sign_mdm_v1`.

### Copy folder back (Windows PowerShell)
Run in the local destination folder where you want to store checkpoints:
```powershell
scp -r user02@14.143.127.114:~/SMPLX/checkpoints/sign_mdm_v1 .\sign_mdm_v1
```

### Faster option: tarball then download
On the server:
```bash
cd ~/SMPLX
tar -czf checkpoints.tgz checkpoints/
```

On Windows:
```powershell
scp user02@14.143.127.114:~/SMPLX/checkpoints.tgz .
# Extract (Windows has bsdtar as `tar` on most modern installs)
tar -xf .\checkpoints.tgz
```

---

## 8) Common gotchas

- **Permission denied (publickey)**: server is key-only; ensure your public key is in `~/.ssh/authorized_keys` on the server.
- **CUDA not detected**: you installed CPU-only PyTorch. Install a CUDA-enabled PyTorch wheel on Ubuntu.
- **Huge uploads**: prefer `rsync` once stable, and avoid copying local venvs and outputs.
- **Training stops after disconnect**: always use `tmux` (or `screen`).

---

## 9) Quick command summary

Windows:
```powershell
ssh -i $env:USERPROFILE\.ssh\a100_user02 user02@14.143.127.114
scp requirements.txt train_diffusion.py sign_language_dataset.py merged_how2sign_mapping.json user02@14.143.127.114:~/SMPLX/
scp -C -r how2sign_pkls_cropTrue_shapeFalse user02@14.143.127.114:~/SMPLX/
scp -r user02@14.143.127.114:~/SMPLX/checkpoints/sign_mdm_v1 .\sign_mdm_v1
```

Ubuntu:
```bash
cd ~/SMPLX
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

tmux new -s train
python train_diffusion.py train --pkl_dir how2sign_pkls_cropTrue_shapeFalse --mapping merged_how2sign_mapping.json
```
