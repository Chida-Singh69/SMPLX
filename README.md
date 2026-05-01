# 🤟 SMPL-X | ASL Animation Suite

An advanced 3D American Sign Language (ASL) generation system utilizing **SMPL-X parametric models**, **FAISS-powered semantic matching**, and **VAE latent space blending**.

---

## 🚀 Quick Start

### 1. Installation
Run the automated installation script for your OS:
```powershell
# Windows
.\install_sentence_system.bat
```

### 2. Model Weights
The VAE model weights and latent cache are included in the repository for easy access:
- `checkpoints/vae_h2s/vae_best.pt`
- `checkpoints/vae_h2s/norm_stats.npz`
- `checkpoints/vae_h2s/latent_cache.npz`

### 3. Launch the System
Start the backend server and the premium web interface:
```powershell
# Start Backend
python app.py

# Start Main UI (Premium Interface)
streamlit run streamlit_app.py
```

### 4. Direct API Usage (CURL Example)
You can trigger animations directly from the command line:
```powershell
# Create payload
Set-Content -Path payload.json -Value '{"text":"I have really worked hard on this","gender":"neutral","fps":15,"max_frames":60,"use_cache":false,"use_vae":true}' -NoNewline

# Send request
curl.exe -i -X POST "http://127.0.0.1:5000/api/render_text" -H "Content-Type: application/json" --data-binary '@payload.json'
```

---

## 🌟 Key Features

### 🧠 VAE Motion Prior (NEW)
- **Latent Blending**: Interpolates between top-k semantic matches in latent space to generate smooth, novel animations.
- **Improved Continuity**: Reduces jitter by blending motion encodings rather than simple concatenation.

### 📊 Semantic Sentence Matching
- **30K How2Sign Dataset**: Massive library of high-quality sentence-level signs.
- **FAISS Vector Search**: Ultra-fast semantic lookups using `sentence-transformers`.
- **Hybrid Strategy**: Uses full-sentence matching with a phrase-level chunking fallback for 85%+ coverage.

### 🎨 Premium Web Interface
- **Dynamic YouTube Translation**: Paste a URL, extract transcript, and generate ASL overlay.
- **Interactive Poses Explorer**: Inspect and assemble raw 3D pose data frame-by-frame.
- **Gender Customization**: Support for Neutral, Male, and Female SMPL-X body types.

---

## 🛠️ API Documentation

### **Text to Animation**
`POST /api/render_text`
- `text`: Input English sentence.
- `use_vae`: (bool) Enable latent blending.
- `gender`: neutral | male | female
- `fps`: Default 15.

### **YouTube to ASL**
`POST /asl_from_youtube_sentences`
- `url`: YouTube Video URL.
- `max_sentences`: Max lines to process.
- `use_vae`: (bool) Enable latent blending.

---

## 📁 Project Structure

```text
SMPLX/
├── app.py                      # Flask API (Core logic & VAE Inference)
├── vae_model.py                # SignLanguageVAE architecture
├── sentence_matcher.py         # FAISS & Semantic lookup engine
├── sentence_to_smplx.py        # 3D SMPL-X rendering pipeline
├── streamlit_app.py            # Main Premium Web UI
├── checkpoints/                # Model weights (vae_best.pt, etc.)
├── models/                     # SMPL-X parametric files
└── output/                     # Generated MP4 animations
```

---

## 🎓 Academic Project

**Dayananda Sagar College of Engineering**  
Computer Science and Design (B.E.)  
**Team:** Akriti Khetan, Bhoomika K S, Chidananda Singh A  
**Guide:** Prof. Nayana U Shinde

---

## ⚠️ Important Notes
- **First Load**: The first request takes ~2 minutes to build the FAISS index in memory.
- **OpenGL**: Requires a GPU with OpenGL support for headless rendering (Pyrender).
- **Dataset**: Built upon the How2Sign dataset (30,000+ annotated sentences).
