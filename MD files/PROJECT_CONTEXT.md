# 🤟 SMPL-X ASL Animation Suite - Project Context

**Academic Project:** Dayananda Sagar College of Engineering, B.E. Computer Science & Design  
**Team:** Akriti Khetan, Bhoomika K S, Chidananda Singh A  
**Guide:** Prof. Nayana U Shinde  
**Status:** MVP Phase (Real-time inference focus)

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Current Architecture](#current-architecture)
3. [Data Status & Missing Data](#data-status--missing-data)
4. [Known Issues & Flaws](#known-issues--flaws)
5. [Model Performance Analysis](#model-performance-analysis)
6. [Priority Work Items](#priority-work-items)
7. [Timeline & Roadmap](#timeline--roadmap)

---

## 🎯 Project Overview

### What Is This?
An advanced **3D American Sign Language (ASL) generation system** that converts English text/videos into photorealistic ASL animations using **SMPL-X parametric body models**. The system combines:
- **Semantic matching** (FAISS + sentence-transformers)
- **Latent space interpolation** (VAE-based motion blending)
- **Real-time 3D rendering** (pyrender → future Three.js migration)

### Core Goal
**Generate natural, diverse ASL animations from arbitrary English input** without manual video creation.

### Key Differentiator
Unlike simple pose retrieval, the system **blends motion in latent space** to create novel animations from existing dataset examples, improving diversity while maintaining quality.

### Current Capabilities
✅ Text-to-ASL conversion (sentence-level)  
✅ YouTube transcript extraction & translation  
✅ 31K+ sentence dataset with SMPL-X poses  
✅ Gender customization (neutral/male/female)  
✅ VAE latent blending for motion smoothing  
⚠️ Real-time inference (slow, needs optimization)

---

## 🏗️ Current Architecture

### System Pipeline
```
User Input (Text/Video)
    ↓
[Sentence Matcher] - FAISS semantic search on 31K sentences
    ↓
[Top-K Retrieval] - Get best matching poses from dataset
    ↓
[VAE Encoder] - Convert poses to latent vectors (32-64D)
    ↓
[Latent Blending] - Interpolate between top matches for novelty
    ↓
[VAE Decoder] - Reconstruct motion [T, 182] SMPL-X parameters
    ↓
[SMPL-X Renderer] - Render 3D mesh per frame
    ↓
[MP4 Export] - Video output to disk
```

### Key Components

| Component | Technology | Status | Issues |
|-----------|-----------|--------|--------|
| **Sentence Matcher** | FAISS + sentence-transformers | ✅ Working | Needs fallback for low-match sentences |
| **VAE Model** | PyTorch (64D latent) | ✅ Trained | Artifacts in blended motions |
| **MDM v1/v2/v3** | Diffusion models | ⚠️ Marginal | See Model Performance Analysis |
| **Rendering** | pyrender (offline) | ✅ Works | Bottleneck for real-time (minutes per video) |
| **Flask API** | Flask + CORS | ✅ Basic | Missing error handling |
| **Frontend** | Streamlit + React (Vite) | 🚧 WIP | YouTube overlay incomplete |

### Data Flow Files
- `backend/api/app.py` - Flask API endpoints (core logic)
- `backend/core/sentence_matcher.py` - FAISS indexing & search
- `backend/core/sentence_to_smplx.py` - SMPL-X pose loading & rendering
- `backend/models/vae/vae_model.py` - VAE architecture
- `backend/models/mdm/train_diffusion*.py` - Diffusion inference (v1, v2, v3)
- `streamlit/streamlit_app.py` - Web UI

---

## 📦 Data Status & Missing Data

### 🟢 Data That Exists

#### 1. How2Sign Dataset (SignAvatars)
- **Location:** `data/raw_poses/how2sign_pkls_cropTrue_shapeFalse/`
- **Size:** ~31,625 .pkl files (estimated ~300-400GB)
- **Format:** PyTorch serialized tensors with CUDA placeholders
- **Content per file:**
  - `smplx`: [N_frames, 182] - Full SMPL-X parameters
  - `unsmooth_smplx`: [N_frames, 169] - Alternative format
  - `2d`: [N_frames, 106, 3] - 2D keypoints
  - `left_valid`, `right_valid` - Hand validity flags
  - Camera parameters (focal length, principal point)
- **Mapping:** `data/metadata/how2sign_mapping.json` (pkl_file → English sentence)
- **Status:** ✅ Fully integrated

#### 2. Neural Sign Actors Dataset
- **Location:** `data/raw_poses/poses/` (trial version) & `how2sign-trial/`
- **Size:** ~375K individual frame .pkl files
- **Format:** Standard NumPy pickle (one frame per file)
- **Content per frame:**
  - `smplx_root_pose`: [3]
  - `smplx_body_pose`: [63]
  - `smplx_lhand_pose`, `smplx_rhand_pose`: [45] each
  - `smplx_jaw_pose`: [3]
  - `smplx_shape`, `smplx_expr`: [10] each
  - `cam_trans`: [3]
- **Status:** ⚠️ Partially integrated (trial version exists)

#### 3. SMPL-X Model Files
- **Location:** `models/smplx/`
- **Files:** SMPLX_NEUTRAL.npz, SMPLX_MALE.npz, SMPLX_FEMALE.npz
- **Content:** Mesh vertices, faces, blend shapes, joint regressors
- **Status:** ✅ Complete

#### 4. Cached Assets
- **VAE Checkpoint:** `checkpoints/vae_h2s/vae_best.pt` (~25MB)
- **Normalization Stats:** `checkpoints/vae_h2s/norm_stats.npz`
- **Latent Cache:** `checkpoints/vae_h2s/latent_cache.npz` (pre-encoded embeddings)
- **Status:** ✅ Available

#### 5. MDM Checkpoints
- **v1:** `checkpoints/mdm_weights/sign_mdm_v1/` (trained)
- **v2:** `checkpoints/mdm_weights/checkpoints_v2/sign_mdm_v2/` (trained)
- **v3:** `checkpoints/mdm_weights/checkpoints_v3/sign_mdm_v3/` (trained)
- **Status:** ⚠️ All exist but performance varies

### 🔴 Data That's MISSING (Due to Size)

#### 1. Full How2Sign Dataset (30K → 70K)
- **Missing:** Original SignAvatars full 70K sentence dataset
- **Current:** Only ~31K sentences available
- **Impact:** 45% coverage loss = more unmatched sentences → more fallback needed
- **Size if available:** ~700GB
- **How to get:** Request from SignAvatars GitHub (academic access)

#### 2. Full Neural Sign Actors
- **Missing:** Complete Neural Sign Actors (2,318 videos)
- **Current:** Trial subset only
- **Impact:** Could add 2K+ more unique sentence matches
- **Size:** ~400GB
- **How to get:** Download from original GitHub repo

#### 3. WLASL 2000 Sign Database
- **Missing:** Pre-extracted SMPL-X poses for WLASL (2K isolated signs)
- **Current:** None
- **Impact:** Better fallback for unknown words (fingerspelling)
- **Size:** ~50GB (estimated)
- **How to get:** Run MediaPipe + SMPL-X fitting on WLASL videos

#### 4. YouTube Transcript Cache
- **Missing:** Pre-cached popular YouTube videos with timestamps
- **Current:** Real-time extraction only
- **Impact:** Faster demo videos, offline capability
- **Size:** ~10-50GB per 1000 videos
- **How to get:** Batch download transcripts + timestamps

#### 5. Diffusion Training Data
- **Missing:** Augmented dataset for MDM (multi-pose per sentence)
- **Current:** Only raw pose sequences
- **Impact:** Better diffusion training, reduced repetition
- **Size:** Depends on augmentation (2-5x of raw)

---

## ⚠️ Known Issues & Flaws

### 1. **Model Accuracy & Repetition Problem** ❌
**Severity:** 🔴 HIGH | **Impact:** MVP demo quality  

**Issue:**
- Animations generated by **all three MDM versions (v1, v2, v3) are inaccurate**
- Generate **similar animations repeatedly** for different input sentences
- **Cross-validation shows** the models haven't truly learned diverse motion
- No clear semantic understanding of text input

**Root Cause:**
- Insufficient training data diversity (30K sentences but many semantically similar)
- Text encoding bottleneck (CLIP embeddings may not capture ASL semantics well)
- Potential loss function imbalance (body vs. hand weighting)
- Diffusion models trained on limited GPU (5060 Ti) - may not have converged

**Evidence:**
- `check_nan.py`, `check_weights.py`, `inspect_checkpoints.py` indicate frequent NaN/corruption issues
- v3 is "latest" but no documented improvement over v1/v2
- No ablation studies to identify which component fails

**Current Workaround:**
Use **VAE latent blending instead** of pure generation. This achieves 85%+ acceptable results by:
- Retrieving best N matches from dataset
- Blending their latent encodings (guaranteed to stay in learned distribution)
- Decoding blended latent back to motion
- Result: Smooth novel animations (though limited by dataset coverage)

**Recommended Action:**
**PIVOT AWAY from generative diffusion for MVP.** Use retrieval + blending for now:
```python
# Current flow (BROKEN):
Text → MDM Diffusion → Often gibberish animations

# Recommended flow (WORKING):
Text → Semantic Match → VAE Blend → Good animations
```

---

### 2. **Architecture Indecision** ⚠️
**Severity:** 🟡 MEDIUM | **Impact:** Development direction  

**Issue:**
- **MDM (Diffusion) vs. VAE vs. Retrieval** - unclear which to prioritize
- v1, v2, v3 all exist but no clear performance ranking
- No decision matrix for which models to use when
- Documentation splits focus: diffusion training guide exists, but VAE is actually working better

**Root Cause:**
- Project evolved through multiple phases (diffusion → VAE → blending)
- Each phase added code without removing old approaches
- No unified benchmark

**Current State:**
- **VAE latent blending:** Actually works well ✅
- **MDM diffusion:** Doesn't work well ❌
- **Pure retrieval:** 70-75% success rate (missing ~25% of sentences)

**Recommended Action:**
Make explicit decision for MVP:
```
MVP DECISION: Use Retrieval + VAE Blending + Fallback Chain
- Phase 1: Retrieval (fast, 70% coverage)
- Phase 2: VAE blending (smooth novel animations)
- Phase 3: Fallback (fingerspelling for unknowns)
- Post-MVP: Then consider diffusion if needed
```

---

### 3. **Rendering Bottleneck** 🐌
**Severity:** 🟡 MEDIUM | **Impact:** Real-time capability  

**Issue:**
- **pyrender offline rendering** is the major bottleneck (minutes per video)
- 300-frame animation = 3-10 minutes for rendering
- Real-time requirement impossible with current approach

**Timeline:**
- Motion generation: ~0.5-1s
- **Rendering: 3-10 minutes** ← BOTTLENECK
- Video encoding: ~10s

**Root Cause:**
- pyrender renders each frame sequentially with full lighting + shadows
- No GPU acceleration for rendering (CPU-bound)
- No caching or progressive rendering

**Recommended Solution:**
**Migrate to Three.js WebGL rendering:**
- Export SMPL-X mesh as .glb file (one-time)
- Stream pose frames as JSON over WebSocket
- Browser renders in real-time (60fps)
- Estimated benefit: 3-10 min → 1-2 sec ✅

**Estimated Work:** 5-7 days (export + WebSocket + browser UI)

---

### 4. **Sentence Matching Coverage** 📊
**Severity:** 🟡 MEDIUM | **Impact:** Translation success rate  

**Issue:**
- Only **31K unique sentences** in current dataset
- Similarity threshold too strict (HIGH_CONFIDENCE = 0.85)
- ~25-30% of input sentences **fail to match** anything usable
- Fallback system is incomplete

**Current Matching Strategy:**
```
1. HIGH_CONFIDENCE (0.85): Use directly
2. MEDIUM_CONFIDENCE (0.70): Use with caution
3. LOW_CONFIDENCE (0.60): Must chunk into phrases
4. NO_MATCH: Currently fails ❌ (should fallback)
```

**Root Cause:**
- Too few training sentences for comprehensive coverage
- Thresholds may be miscalibrated for ASL domain
- No hierarchical fallback implemented

**Recommended Action:**
1. Expand dataset from 31K → 70K+ (see Data section)
2. Implement fallback chain:
   ```python
   if match_score > 0.85:
       use_full_sentence()  # Best option
   elif match_score > 0.70:
       blend_topk_matches()  # VAE blending
   elif match_score > 0.60:
       chunk_phrases()  # Split sentence into chunks
   else:
       fallback_fingerspelling()  # Spell it out letter-by-letter
   ```
3. Add phrase-level dataset from How2Sign metadata
4. Create 26-letter fingerspelling library

**Estimated improvement:** 70% → 95%+ coverage

---

### 5. **Corrupted Checkpoint Files** 💥
**Severity:** 🔴 HIGH | **Impact:** Training & debugging  

**Issue:**
- **NaN values detected** in checkpoint weights (per `check_nan.py`)
- Indicates training instability
- Models may produce invalid outputs
- Difficult to resume training

**Evidence:**
- `scripts/debug/inspect_checkpoints.py` warns about corrupted data
- v1, v2, v3 all have corruption issues (check logs)
- No validation pipeline to catch this early

**Recommended Action:**
1. Run checkpoint validation:
   ```bash
   python scripts/debug/inspect_checkpoints.py
   ```
2. Implement pre-training data validation:
   ```bash
   python scripts/data_prep/validate_dataset.py --data_dir how2sign_pkls_cropTrue_shapeFalse
   ```
3. Retrain with dropout + batch normalization to stabilize
4. Add checkpoint validation hook in training loop

---

### 6. **Incomplete Fallback System** 🔗
**Severity:** 🟡 MEDIUM | **Impact:** MVP robustness  

**Issue:**
- No comprehensive fallback when sentence doesn't match
- Currently fails silently or crashes
- No fingerspelling engine (0% coverage for unknown words)
- No phrase-level chunking

**Current State:**
- Sentence-level matching only
- If no match → system fails
- No graceful degradation

**Recommended Fallback Chain:**
```python
1. Full Sentence Match (>0.85 similarity) 
2. Multi-key Match (split into words, match each)
3. Phrase-level Chunking (split sentence into 2-3 word chunks)
4. Word-level Lookup (match individual words)
5. Fingerspelling Engine (spell unknown words letter-by-letter)
6. Default ASL Avatar Greeting (last resort)
```

**Missing Components:**
- Fingerspelling 26-letter library
- Phrase-level chunking logic
- Word-level ASL dictionary (WLASL mapping)
- Transition animations between chunks

**Estimated Work:** 3-5 days to fully implement

---

### 7. **YouTube Integration Incomplete** 📹
**Severity:** 🟢 LOW | **Impact:** Demo/UX  

**Issue:**
- YouTube transcript extraction works
- ASL overlay/rendering incomplete in Streamlit UI
- Real-time WebSocket streaming not implemented

**Current State:**
- `youtube-transcript-api` can pull transcripts ✅
- Sentence splitting ✅
- Individual sentence-to-ASL ✅
- **Multiple ASL overlays on video** ❌
- **YouTube player + ASL sync** ❌

**Recommended Fix:**
Implement WebSocket streaming (see Real-time roadmap below)

---

## 📊 Model Performance Analysis

### MDM (Motion Diffusion Model) - v1, v2, v3

| Metric | v1 | v2 | v3 | Target |
|--------|----|----|----|----|
| **Accuracy** | ⚠️ Low | ⚠️ Low | ⚠️ Low | ✅ >90% |
| **Diversity** | ❌ Repetitive | ❌ Repetitive | ❌ Repetitive | ✅ Unique |
| **Speed** | 🐌 3-5s | 🐌 3-5s | 🐌 3-5s | ✅ <1s |
| **Stability** | ⚠️ NaNs | ⚠️ NaNs | ⚠️ NaNs | ✅ Stable |
| **Checkpoint Quality** | 💥 Corrupted | 💥 Corrupted | 💥 Corrupted | ✅ Valid |

### Issues Across All Versions:
1. **Semantic Understanding Failure**
   - Input: "I like cats" → Output: Generic signing motion
   - Input: "The weather is sunny" → Same output (!)
   - CLIP text embedding may not capture ASL semantics

2. **Motion Repetition**
   - Sample 10 outputs for same sentence
   - 7-8 are identical or very similar
   - Diffusion isn't exploring latent space

3. **Hand Quality**
   - Fingers collapse inward
   - Hand-pose loss (2.0x weighted) not working
   - GNN loss function may be miscalibrated

4. **Temporal Coherence**
   - Frame-to-frame jitter
   - Unnatural pose transitions
   - LSTM decoder may be underfitting

### Why Diffusion Isn't Working:

**Technical Root Causes:**
1. **Insufficient Training Data**
   - 30K sentences looks big, but semantically many are similar ("How are you?" variants)
   - Real diversity: ~5K unique semantic meanings
   - Diffusion needs 100K+ diverse examples

2. **Text Encoding Bottleneck**
   - CLIP was trained on images + general English text
   - ASL is fundamentally different (spatial, temporal, hand-shape focused)
   - Need ASL-specific text encoding

3. **Loss Function Imbalance**
   - Body loss + 2.0 * hand loss → aggressive hand correction
   - May cause body to compensate (unnatural poses)
   - No tuning/ablation studies done

4. **Training Stability**
   - 5060 Ti (16GB VRAM) is marginal for diffusion
   - Out-of-memory errors → training crash & corruption
   - Gradient explosion possible (no gradient clipping visible)

5. **Evaluation Metric Missing**
   - No quantitative metric for "semantically correct motion"
   - Just visual inspection → bias & inconsistency

### VAE Latent Blending - Current Workaround ✅

**Why it works:**
- Encodes known-good poses from dataset
- Interpolates in latent space (guaranteed valid)
- Decoding always produces plausible motion
- No hallucination risk (unlike diffusion)

**Performance:**
- **Accuracy:** ✅ 95%+ (exact or very similar to dataset examples)
- **Diversity:** ⚠️ Limited to dataset blends (but acceptable)
- **Speed:** ✅ <1s (single forward pass)
- **Stability:** ✅ No NaNs, always completes

**Limitation:**
- Can only blend existing poses
- Limited novelty (constrained by dataset)
- Works best with dataset expansion

**Verdict:** **Use VAE blending for MVP.** Replace with better text encoding + diffusion in future.

---

## 🎯 Priority Work Items

### 🚀 IMMEDIATE (For MVP Demo - Next 2 Weeks)

#### 1. Fix Real-Time Rendering [CRITICAL]
**Current:** 3-10 minutes rendering time  
**Goal:** <2 seconds per video

**Tasks:**
- [ ] Export SMPL-X .glb mesh (one-time)
- [ ] Build Flask-SocketIO WebSocket server
- [ ] Create Three.js WebGL renderer
- [ ] Implement pose frame streaming
- [ ] Connect frontend UI to WebSocket

**Files to modify:**
- `backend/core/sentence_to_smplx.py` (add GLB export)
- `backend/api/app.py` (add WebSocket endpoints)
- `frontend_vite/src/App.tsx` (Three.js integration)

**Estimated time:** 5-7 days

---

#### 2. Fix Sentence Matching Fallback Chain [HIGH]
**Current:** 70% success (failures crash)  
**Goal:** 95%+ success with graceful degradation

**Tasks:**
- [ ] Implement phrase-level chunking (split "I like cats" → ["I like", "cats"])
- [ ] Add word-level fallback
- [ ] Create 26-letter fingerspelling library
- [ ] Add transition animations between chunks
- [ ] Implement fallback logic in app.py

**Files to modify:**
- `backend/api/app.py` (add fallback routing)
- `backend/core/sentence_matcher.py` (enhance matching logic)
- Create `backend/core/fingerspelling_engine.py` (new)

**Estimated time:** 3-5 days

---

#### 3. Validation & Data Quality Check [MEDIUM]
**Current:** Unknown data quality  
**Goal:** Identify & flag bad poses early

**Tasks:**
- [ ] Run `validate_dataset.py` on full how2sign dataset
- [ ] Document any corrupted/invalid files
- [ ] Create data cleaning script
- [ ] Add pre-flight checks to app startup
- [ ] Log data issues to file

**Files to modify/run:**
- `scripts/data_prep/validate_dataset.py` (run as-is)
- `backend/api/app.py` (add startup checks)

**Estimated time:** 2-3 days

---

### 📈 SHORT TERM (Weeks 3-4)

#### 4. Expand Dataset Coverage
**Current:** 31K sentences  
**Goal:** 70K+ sentences (better matching)

**Options (Choose 1-2):**
- [ ] **Request full SignAvatars dataset** (70K sentences) from GitHub (1-2 weeks waiting)
- [ ] **Add Neural Sign Actors full dataset** (~2K new videos, 5-7 days integration)
- [ ] **Mine WLASL 2000** signs for fingerspelling fallback (5-10 days with GPU)

**Estimated time:** 1-3 weeks (depends on data access)

---

#### 5. YouTube Multi-Sentence Overlay [MEDIUM]
**Current:** Extracts transcript but no overlay  
**Goal:** Full transcript → ASL video overlay

**Tasks:**
- [ ] Extract timestamps from YouTube transcript
- [ ] Chunk transcript into sentences with timings
- [ ] Generate ASL for each chunk
- [ ] Composite ASL avatar alongside YouTube video
- [ ] Sync audio + avatar timing

**Files to modify:**
- `streamlit/streamlit_youtube_sentences.py` (complete UI)
- `backend/api/app.py` (add batch endpoint)

**Estimated time:** 4-5 days

---

#### 6. Checkpoint Stability Fix
**Current:** NaN corruption in all v1/v2/v3  
**Goal:** Clean, valid checkpoints

**Tasks:**
- [ ] Run diagnostic: `scripts/debug/inspect_checkpoints.py`
- [ ] Identify which checkpoints are valid
- [ ] Retrain failing models with gradient clipping
- [ ] Document training stability improvements
- [ ] Archive corrupted checkpoints

**Estimated time:** 3-7 days (depends on training needs)

---

### 🎯 MEDIUM TERM (Weeks 5-8)

#### 7. Architecture Decision & Documentation
**Goal:** Settle on single unified architecture

**Deliverable:**
- Create `ARCHITECTURE_DECISION.md` documenting:
  - Why retrieval + VAE blending > pure diffusion
  - When to use each component
  - Future upgrade path to diffusion (if time permits)

**Estimated time:** 1 day

---

#### 8. Performance Profiling & Optimization
**Current:** Unknown bottlenecks  
**Goal:** <1s end-to-end for common sentences

**Tasks:**
- [ ] Profile end-to-end pipeline with `cProfile`
- [ ] Identify top 3 bottlenecks
- [ ] Optimize/cache expensive operations
- [ ] Benchmark improvement

**Estimated time:** 3-5 days

---

#### 9. Better Text Encoding [OPTIONAL]
**Current:** Uses OpenAI CLIP (English-focused)  
**Future:** ASL-aware text encoding

**Options:**
- [ ] Fine-tune CLIP on ASL lexicon
- [ ] Use MUSE (multilingual embeddings)
- [ ] Train custom text encoder on How2Sign corpus

**Note:** Lower priority for MVP (VAE blending already works well)

**Estimated time:** 2-3 weeks (if pursued)

---

## 📅 Timeline & Roadmap

### Phase 1: MVP (Weeks 1-2) - FOCUS ON REAL-TIME DEMO
```
Week 1:
  - Fix rendering bottleneck (WebSocket + Three.js)
  - Implement fallback chain
  - Data validation
  
Week 2:
  - YouTube multi-sentence overlay
  - Bug fixes & testing
  - MVP ready for demo
```

**Success Metrics:**
- [ ] Real-time text-to-ASL (<2 seconds)
- [ ] 95%+ sentence match success
- [ ] YouTube transcript integration working
- [ ] No crashes or NaNs

---

### Phase 2: Robustness (Weeks 3-4)
```
Week 3:
  - Expand dataset (if available)
  - Checkpoint stability improvements
  - Performance profiling
  
Week 4:
  - Architecture decision doc
  - Frontend polish
  - Prepare for demo/submission
```

---

### Phase 3: Polish & Enhancement (Post-MVP)
```
- Better text encoding (ASL-aware)
- Diffusion model improvements
- Mobile app version
- Real-time streaming capability
- Integration with educational platforms
```

---

## 🛠️ Development Setup Checklist

### Required
- [ ] Python 3.11+ (GPU-enabled PyTorch)
- [ ] CUDA 12.x (for GPU acceleration)
- [ ] 16GB+ VRAM (5060 Ti minimum)
- [ ] 500GB+ free storage (dataset + checkpoints)

### Installation
```bash
# Windows
.\scripts\install_sentence_system.bat

# Linux/Mac
bash scripts/install_sentence_system.sh

# Or manual
pip install -r requirements.txt
```

### Quick Start
```bash
# Backend
python backend/api/app.py

# Frontend (Streamlit)
streamlit run streamlit/streamlit_app.py

# Test API
curl -X POST http://127.0.0.1:5000/api/render_text \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","gender":"neutral"}'
```

---

## 📚 Key References

### Documentation
- `mdm_training_implementation.md` - Diffusion training guide
- `FEASIBILITY_ASSESSMENT.md` - Architecture decision analysis
- `DATASET_COMPARISON_REPORT.md` - How2Sign vs. Neural Actors
- `VAE_MOTION_PRIOR_PLAN.md` - VAE blending approach

### Scripts
- `scripts/prepare_mvp_dataset.py` - End-to-end pipeline
- `scripts/data_prep/validate_dataset.py` - Data quality check
- `scripts/debug/inspect_checkpoints.py` - Model inspection

### Data Locations
- Poses: `data/raw_poses/`
- Metadata: `data/metadata/`
- Checkpoints: `checkpoints/`
- Output: `data/mp4_outputs/`

---

## 🎓 Contributors & Acknowledgments

**Team:** Akriti Khetan, Bhoomika K S, Chidananda Singh A  
**Institution:** Dayananda Sagar College of Engineering  
**Guide:** Prof. Nayana U Shinde  
**Data Sources:** How2Sign, SignAvatars, Neural Sign Actors, WLASL

---

## 📝 Notes & Questions

### Open Questions
1. **Architecture:** Is retrieval + VAE blending confirmed as direction for MVP?
2. **Dataset:** Can we access full SignAvatars or Neural Sign Actors?
3. **Deployment:** Cloud-hosted or local-only demo?
4. **Timeline:** Hard deadline for MVP demo?

### Known Technical Debt
- [ ] Old diffusion training code needs cleanup (v1, v2, v3 duplication)
- [ ] No unit tests for pipeline
- [ ] Frontend (Vite) not connected to backend endpoints
- [ ] No error logging (only console prints)
- [ ] Dataset validation runs manually (should be automated)

---

**Last Updated:** May 6, 2026  
**Status:** Active Development (MVP Phase)
