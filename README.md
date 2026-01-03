# ASL Overlay for Streaming Media

3D American Sign Language animation generator using SMPL-X body models.

## 🚀 Quick Start

### Option 1: Web UI (Streamlit)

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501

### Option 2: REST API (Flask)

```bash
python app.py
```

Server at http://localhost:5000

## 📋 Features

- ✅ Generate 3D ASL animations from text
- ✅ Extract YouTube transcripts and convert to ASL
- ✅ 2000+ word vocabulary
- ✅ Realistic SMPL-X body model
- ✅ Smooth motion blending
- ✅ MP4 video export

## 🛠️ Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📖 Usage

### Streamlit Interface

1. Select words from dropdown
2. Click "Generate Animation"
3. Preview and download video

### Flask API

```bash
# YouTube translation
curl -X POST http://localhost:5000/asl_from_youtube \
  -H "Content-Type: application/json" \
  -d '{"url": "YOUTUBE_URL"}'

# Get video
curl http://localhost:5000/output/VIDEO_FILENAME.mp4 --output video.mp4
```

### Python Script

```python
from word_to_smplx import WordToSMPLX

animator = WordToSMPLX(model_path="models")
pose_data = animator.load_pose_sequence("word-level-dataset-cpu/00873.pkl")
animator.render_animation(pose_data, save_path="output/hello.mp4", fps=15)
```

## 📁 Project Structure

```
├── app.py                    # Flask REST API
├── streamlit_app.py          # Streamlit web interface
├── word_to_smplx.py         # Core animation engine
├── models/smplx/            # SMPL-X body models
├── word-level-dataset-cpu/  # 2000+ ASL pose sequences
├── filtered_video_to_gloss.json  # Word → filename mapping
├── output/                  # Generated videos
└── requirements.txt         # Python dependencies
```

## 🎓 Academic Project

**Institution:** Dayananda Sagar College of Engineering  
**Course:** Computer Science and Design (B.E.)  
**Team:** Akriti Khetan, Bhoomika K S, Chidananda Singh A  
**Guide:** Prof. Nayana U Shinde

See detailed documentation:

- `BACKEND_DESIGN.md` - System architecture
- `IMPLEMENTATION_GUIDE.md` - Development guide
- `API_REFERENCE.md` - API documentation
- `RUN_GUIDE.md` - Detailed running instructions

## 🔧 Technologies

- **3D Model:** SMPL-X (parametric body model)
- **Rendering:** Pyrender, Trimesh
- **ML/Processing:** PyTorch, NumPy, SciPy
- **Web:** Flask, Streamlit
- **Video:** imageio, FFmpeg

## 📊 Performance

- Single word: ~2-5 seconds
- Multiple words: ~10-30 seconds
- 2000+ word vocabulary
- 15 FPS output (configurable)

## 🤝 Contributing

This is an academic project. For enhancements, see `BACKEND_DESIGN.md` for planned features.

## 📄 License

Academic project - Dayananda Sagar College of Engineering

## 🆘 Support

For issues:

1. Check `RUN_GUIDE.md` for troubleshooting
2. Verify all dependencies installed
3. Ensure models and dataset exist
4. Check console output for errors
