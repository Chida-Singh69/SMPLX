# 🎬 Getting Started - Copy & Paste Commands

## Step 1: Install Dependencies (Choose One)

### Option A: Automated Install (Windows)
```powershell
.\install_sentence_system.bat
```

### Option B: Automated Install (Linux/Mac)
```bash
chmod +x install_sentence_system.sh
./install_sentence_system.sh
```

### Option C: Manual Install
```bash
pip install sentence-transformers faiss-cpu spacy flask youtube-transcript-api
python -m spacy download en_core_web_sm
```

---

## Step 2: Verify Installation
```bash
python -c "import sentence_transformers, faiss, spacy; print('✓ All dependencies ready!')"
```

---

## Step 3: Start the Server
```bash
python app.py
```

You should see:
```
[INFO] Using device: cpu
Loaded 104 words with available pose data (out of 2000 total)
[SentenceMatcher] Initialized (lazy-loading enabled)
 * Running on http://127.0.0.1:5000
```

---

## Step 4: Test the System (New Terminal)

### Quick Test
```bash
python test_sentence_translation.py
```

### Or Manual cURL Test
```bash
curl -X POST http://localhost:5000/asl_from_youtube_sentences ^
  -H "Content-Type: application/json" ^
  -d "{\"url\": \"https://www.youtube.com/watch?v=dQw4w9WgXcQ\", \"max_sentences\": 3}"
```

**Note**: First request takes 2-5 minutes to build the index. This is normal!

---

## Step 5: Check the Output
```bash
# List generated videos
ls output/

# Play the video
# (Open the video file in your media player)
```

---

## 🎯 Example Python Usage

```python
import requests
import json

# Make request
response = requests.post(
    'http://localhost:5000/asl_from_youtube_sentences',
    json={
        'url': 'https://www.youtube.com/watch?v=YOUR_VIDEO_ID',
        'max_sentences': 5
    }
)

# Parse result
result = response.json()

# Print summary
print(f"✓ Video generated: {result['url']}")
print(f"  Sentences processed: {result['sentences_processed']}")
print(f"  Average confidence: {result['statistics']['avg_confidence']:.2f}")
print(f"  Total frames: {result['total_frames']}")

# Download video
video_url = f"http://localhost:5000{result['url']}"
video_data = requests.get(video_url).content

with open('my_asl_video.mp4', 'wb') as f:
    f.write(video_data)

print("✓ Video saved to my_asl_video.mp4")
```

---

## 🔍 Troubleshooting

### Issue: "Import Error: sentence_transformers"
```bash
pip install sentence-transformers
```

### Issue: "Import Error: faiss"
```bash
pip install faiss-cpu
```

### Issue: "Can't find model 'en_core_web_sm'"
```bash
python -m spacy download en_core_web_sm
```

### Issue: "Connection refused"
Make sure Flask server is running:
```bash
python app.py
```

### Issue: "First request takes forever"
This is **normal**! The system is:
1. Downloading sentence-transformer model (~90MB)
2. Building FAISS index from 30K sentences (~2-5 min)

After the first request, everything is fast!

---

## 📊 What to Expect

### First Request Timeline
```
[0:00] Request sent
[0:05] Loading sentence-transformers model...
[0:30] Loading 30K sentence mappings...
[0:40] Encoding sentences to vectors...
[3:30] Building FAISS index...
[3:40] ✓ Index built! Matching sentences...
[3:45] ✓ Video generated!
```

### Subsequent Requests
```
[0:00] Request sent
[0:02] ✓ Video generated! (index already built)
```

---

## 🎉 Success Indicators

You know it's working when you see:
```
[SentenceMatcher] Loading dependencies...
  ✓ sentence-transformers loaded
  ✓ faiss loaded
  ✓ spacy loaded
[SentenceMatcher] Loading mappings from how2sign_mapping.json...
  ✓ Loaded 30997 sentences with available .pkl files
[SentenceMatcher] Building semantic search index...
  Loading sentence transformer model...
  Encoding 30997 sentences...
Batches: 100%|████████████████| 243/243 [02:34<00:00]
  Building FAISS index...
  ✓ Index built with 30997 sentences
```

---

## 📖 Need More Help?

- **Quick Start**: `QUICKSTART.md`
- **Full Documentation**: `SENTENCE_TRANSLATION_README.md`
- **Technical Details**: `IMPLEMENTATION_SUMMARY.md`
- **Delivery Summary**: `DELIVERY_SUMMARY.md`

---

## 🚀 Ready to Go!

Now you have a **semantic sentence-based ASL translation system** using 30K+ real sign language sentences! 

Try it with different YouTube videos and see the natural, continuous sign language animations! 🎬✨
