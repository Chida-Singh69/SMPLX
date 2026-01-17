# 🤟 YouTube to ASL - Sentence Translation System

3D American Sign Language animation generator using SMPL-X body models with semantic sentence matching.

## 🚀 Quick Start

### 1. Install
```bash
# Windows
install_sentence_system.bat

# Linux/Mac  
bash install_sentence_system.sh
```

### 2. Run Server
```bash
python app.py
```

### 3. Launch Web UI
```bash
# Sentence-level (recommended - 30K dataset)
streamlit run streamlit_youtube_sentences.py

# Word-level (legacy - 104 words)
streamlit run streamlit_app.py
```

## 📊 Features

### Sentence-Level Translation (NEW)
- ✅ **30K How2Sign sentences** with semantic matching
- ✅ **FAISS vector search** for similarity
- ✅ **Phrase chunking fallback** for better coverage
- ✅ **Confidence scoring** (High/Medium/Low)
- ✅ **70-85% transcript coverage**

### Word-Level Translation (Legacy)
- ✅ **104 words** with direct lookup
- ✅ **5-15% transcript coverage**
- ✅ **Fast rendering**

## � API Endpoints

### Sentence Translation
```bash
POST http://localhost:5000/asl_from_youtube_sentences
{
  "url": "https://youtube.com/watch?v=VIDEO_ID",
  "max_sentences": 5
}
```

### Word Translation (Legacy)
```bash
POST http://localhost:5000/asl_from_youtube
{
  "url": "https://youtube.com/watch?v=VIDEO_ID"
}
```

## 📁 Project Structure

```
SMPLX/
├── app.py                              # Flask API server
├── sentence_matcher.py                 # Semantic matching (FAISS)
├── sentence_to_smplx.py               # 3D renderer (sentences)
├── word_to_smplx.py                   # 3D renderer (words)
├── streamlit_youtube_sentences.py     # Web UI (sentence-level)
├── streamlit_app.py                   # Web UI (word-level)
├── test_sentence_translation.py       # Testing
│
├── how2sign_mapping.json              # 30K sentence mappings
├── how2sign_pkls_cropTrue_shapeFalse/ # Sentence pose data
├── filtered_video_to_gloss.json       # 104 word mappings
├── word-level-dataset-cpu-fixed/      # Word pose data
└── output/                            # Generated videos
```

## ⚙️ Tech Stack

- **Semantic Matching**: sentence-transformers, FAISS
- **3D Model**: SMPL-X parametric body model
- **Rendering**: Pyrender, OpenGL
- **Backend**: Flask, PyTorch
- **Frontend**: Streamlit
- **Dataset**: 30K How2Sign + 104 words

## 📚 Documentation

- **QUICKSTART.md** - 5-minute guide
- **GETTING_STARTED.md** - Detailed setup
- **SENTENCE_TRANSLATION_README.md** - Technical docs

## 🎓 Academic Project

**Dayananda Sagar College of Engineering**  
Computer Science and Design (B.E.)  
Team: Akriti Khetan, Bhoomika K S, Chidananda Singh A  
Guide: Prof. Nayana U Shinde

## ⚠️ Notes

- **First request**: 2-5 min (builds FAISS index)
- **Semantic matching**: Not true translation, finds similar sentences
- **Best for**: Conversational/educational content
- **Lower accuracy**: Abstract/motivational content
