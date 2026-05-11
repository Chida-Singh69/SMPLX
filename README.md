# SilentVoice: SMPL-X ASL Animation Suite

SilentVoice is an advanced 3D American Sign Language (ASL) synthesis system designed to bridge the digital accessibility gap for the Deaf and Hard-of-Hearing community. The system utilizes SMPL-X parametric body models, FAISS-powered semantic matching, and VAE latent space blending to generate linguistically accurate and visually realistic sign language animations from English text.


## Demo

<img width="1600" height="752" alt="Image" src="https://github.com/user-attachments/assets/272fa1da-3ed5-4906-b9a6-246b3985339a" />

<img width="1600" height="764" alt="Image" src="https://github.com/user-attachments/assets/23869a90-7339-4bc6-9241-fd6cdff46b32" />

<img width="1600" height="766" alt="Image" src="https://github.com/user-attachments/assets/ea2e227d-5254-446e-8260-0bc013ab3891" />

<img width="1600" height="750" alt="Image" src="https://github.com/user-attachments/assets/858dc938-e287-46b4-a5b9-6452f26ae3fd" />

<img width="1419" height="1041" alt="Image" src="https://github.com/user-attachments/assets/61ef22df-70d6-4129-98a2-03f350d7340f" />



## Core Technology
- **Semantic Sentence Matching**: Utilizes Sentence-BERT and FAISS for ultra-fast semantic lookups over a 31K+ sentence corpus (How2Sign).
- **VAE Latent Blending**: Performs weighted interpolation of top-K semantic matches in a Variational Autoencoder latent space, enabling the generation of smooth, novel animations while maintaining anatomical plausibility.
- **SMPL-X Parametric Models**: Renders animations on 55-joint anatomically accurate avatars with support for neutral, male, and female body types.
- **Confidence Cascade**: An adaptive three-tier retrieval strategy (full-sentence matching, VAE blending, and phrase-level chunking) achieving over 92% coverage.

## Performance Metrics
- **Coverage Rate**: 92%
- **Semantic Accuracy**: 88%
- **Animation Quality**: 91%
- **Retrieval Latency**: <5ms

## Key Features
- **Text-to-ASL Translation**: Real-time conversion of English sentences into 3D animations.
- **YouTube Integration**: Extract transcripts from YouTube URLs to generate synchronized ASL overlays.
- **Chrome Extension**: A Manifest V3 extension for on-the-fly translation of web content.
- **Gender Customization**: Modular appearance system with vertex-mask garment rendering.

## Quick Start

### 1. Installation
Run the automated installation script:
```powershell
.\install_sentence_system.bat
```

### 2. Model Weights
Required weights should be placed in:
- `checkpoints/vae_h2s/vae_best.pt`
- `checkpoints/vae_h2s/norm_stats.npz`
- `checkpoints/vae_h2s/latent_cache.npz`

### 3. Execution
Start the backend server and the web interface:
```powershell
# Backend
python app.py

# Frontend
cd frontend_vite
npm run dev
```