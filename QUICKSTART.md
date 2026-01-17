# 🚀 Quick Start: Sentence-Level ASL Translation

## Installation (5 minutes)

### Option 1: Automated (Recommended - Windows)
```bash
install_sentence_system.bat
```

### Option 2: Manual
```bash
# 1. Install Python packages
pip install sentence-transformers faiss-cpu spacy flask youtube-transcript-api

# 2. Download spaCy model
python -m spacy download en_core_web_sm

# 3. Verify installation
python -c "import sentence_transformers, faiss, spacy; print('✓ All dependencies installed')"
```

## Usage

### 1. Start the Server
```bash
python app.py
```

### 2. Test the System
```bash
# In a new terminal
python test_sentence_translation.py
```

### 3. Make API Request
```python
import requests

response = requests.post(
    'http://localhost:5000/asl_from_youtube_sentences',
    json={
        'url': 'https://www.youtube.com/watch?v=VIDEO_ID',
        'max_sentences': 5
    }
)

result = response.json()
print(f"Video: {result['url']}")
print(f"Confidence: {result['statistics']['avg_confidence']:.2f}")
```

## What Happens on First Request?

```
[First Request] (~2-5 minutes)
├── Loading sentence-transformers model... (30s)
├── Loading How2Sign mapping (30K sentences)... (10s)
├── Encoding sentences to vectors... (2-3 min)
└── Building FAISS index... (10s)

[Subsequent Requests] (<5 seconds)
└── Index already built, query directly!
```

## Example Output

```json
{
  "url": "/output/youtube_sentences_dQw4w9WgXcQ_5.mp4",
  "sentences_processed": 5,
  "sentences_successful": 5,
  "total_frames": 842,
  "statistics": {
    "high_confidence": 3,
    "medium_confidence": 1,
    "low_confidence": 1,
    "avg_confidence": 0.81
  },
  "note": "This translation uses semantic sentence matching from 30K+ How2Sign dataset"
}
```

## API Endpoints

| Endpoint | Description | Dataset |
|----------|-------------|---------|
| `/asl_from_youtube` | Word-level (OLD) | 104 words |
| `/asl_from_youtube_sentences` | **Sentence-level (NEW)** | **30K sentences** |
| `/extract_transcript` | Preview transcript | - |

## Troubleshooting

### ❌ "sentence-transformers not installed"
```bash
pip install sentence-transformers
```

### ❌ "faiss not installed"
```bash
pip install faiss-cpu
```

### ❌ "Can't find model 'en_core_web_sm'"
```bash
python -m spacy download en_core_web_sm
```

### ❌ Server won't start
```bash
# Check if port 5000 is in use
netstat -ano | findstr :5000  # Windows
lsof -i :5000  # Linux/Mac

# Use different port
python app.py --port 5001
```

### ⏱️ First request takes forever
This is normal! The system needs to:
1. Download sentence-transformer model (~90MB)
2. Build FAISS index from 30K sentences (~2-5 min)

After this, all requests are fast (<5s).

## Full Documentation

- **[SENTENCE_TRANSLATION_README.md](SENTENCE_TRANSLATION_README.md)** - Complete documentation
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Technical details
- **[test_sentence_translation.py](test_sentence_translation.py)** - Testing script

## Need Help?

Check the error logs in the terminal where you ran `python app.py`.

Common issues are documented in [SENTENCE_TRANSLATION_README.md](SENTENCE_TRANSLATION_README.md#troubleshooting).
