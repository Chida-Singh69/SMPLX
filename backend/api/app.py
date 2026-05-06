import os
import json
import re
import string
import hashlib
import pickle
import io
import numpy as np
import torch

from flask import Flask, request, jsonify, send_from_directory, send_file
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

# Add project root to sys.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.core.word_to_smplx import WordToSMPLX
from backend.core.sentence_to_smplx import SentenceToSMPLX
from backend.core.sentence_matcher import SentenceMatcher
from backend.models.vae.vae_model import SignLanguageVAE


def _torch_load_compat(obj, map_location=None):
    """Compatibility wrapper for PyTorch 2.6+ (weights_only default True)."""
    try:
        return torch.load(obj, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(obj, map_location=map_location)
from backend.core.pose_dataset import load_stats

try:
    from flask_cors import CORS
    _cors_available = True
except ImportError:
    _cors_available = False
    print("[WARN] flask-cors not installed. CORS headers will not be set.")
    print("[WARN] Install with: pip install flask-cors")

app = Flask(__name__, static_folder='static', template_folder='templates')
if _cors_available:
    CORS(app)

# --- Setup paths and load resources once ---
current_dir = os.path.dirname(os.path.abspath(__file__))
mapping_path = os.path.join(current_dir, "..", "..", "data", "metadata", "filtered_video_to_gloss.json")
dataset_dir = os.path.join(current_dir, "..", "..", "data", "raw_poses", "word-level-dataset-cpu-fixed")
how2sign_mapping_path = os.path.join(current_dir, "..", "..", "data", "metadata", "how2sign_mapping.json")
how2sign_dataset_dir = os.path.join(current_dir, "..", "..", "data", "raw_poses", "how2sign_pkls_cropTrue_shapeFalse")
if not os.path.exists(how2sign_dataset_dir):
    how2sign_dataset_dir = os.path.join(current_dir, "..", "..", "data", "raw_poses", "how2sign-trial")
output_dir = os.path.join(current_dir, "..", "..", "data", "mp4_outputs")
os.makedirs(output_dir, exist_ok=True)

text_cache_dir = os.path.join(output_dir, "text_cache")
os.makedirs(text_cache_dir, exist_ok=True)

with open(mapping_path, "r") as f:
    gloss_map = json.load(f)

# Create reverse mapping and filter to only words with existing pickle files
word_to_pkl = {}
for pkl_file, word in gloss_map.items():
    full_path = os.path.join(dataset_dir, pkl_file)
    if os.path.exists(full_path):
        word_to_pkl[word.lower()] = pkl_file

dataset_words = set(word_to_pkl.keys())
print(f"Loaded {len(dataset_words)} words with available pose data (out of {len(gloss_map)} total)")

# Initialize animators lazily by gender cache to save memory
animators_cache = {}

def get_animators(gender='neutral'):
    gender = gender.lower()
    if gender not in animators_cache:
        print(f"[INFO] Lazily initializing SMPLX models for gender: {gender.upper()}")
        animators_cache[gender] = {
            'word': WordToSMPLX(model_path=os.path.join(current_dir, "..", "..", "models"), gender=gender),
            'sentence': SentenceToSMPLX(model_path=os.path.join(current_dir, "..", "..", "models"), gender=gender)
        }
    return animators_cache[gender]['word'], animators_cache[gender]['sentence']

# Initialize sentence matcher (lazy-loaded on first use)
sentence_matcher = None

def get_sentence_matcher():
    """Lazy-load sentence matcher on first use."""
    global sentence_matcher
    if sentence_matcher is None:
        print("[INFO] Initializing sentence matcher...")
        sentence_matcher = SentenceMatcher(how2sign_mapping_path, how2sign_dataset_dir)
    return sentence_matcher


# --- VAE Resources (Lazy Loaded) ---
vae_resources = {
    'model': None,
    'stats': None,
    'cache': None,
    'config': None,
    'initialized': False,
    'available': False
}

def get_vae_resources():
    """Lazy-load VAE model, stats, and latent cache."""
    if vae_resources['initialized']:
        return vae_resources

    vae_dir = os.path.join(current_dir, "..", "..", "checkpoints", "vae_weights", "vae_h2s")
    ckpt_path = os.path.join(vae_dir, "vae_best.pt")
    stats_path = os.path.join(vae_dir, "norm_stats.npz")
    cache_path = os.path.join(vae_dir, "latent_cache.npz")

    if not all(os.path.exists(p) for p in [ckpt_path, stats_path, cache_path]):
        print("[INFO] VAE model files not found. VAE blending disabled.")
        vae_resources['initialized'] = True
        vae_resources['available'] = False
        return vae_resources

    try:
        print("[INFO] Loading VAE model components...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint
        ckpt = _torch_load_compat(ckpt_path, map_location=device)
        cfg = ckpt["config"]
        
        model = SignLanguageVAE(
            seq_len=cfg["seq_len"],
            pose_dim=ckpt["pose_dim"],
            latent_dim=cfg["latent_dim"],
            hidden_dim=cfg["hidden_dim"],
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        
        # Load stats and cache
        stats = load_stats(stats_path)
        cache = np.load(cache_path)
        
        vae_resources.update({
            'model': model,
            'stats': stats,
            'cache': cache,
            'config': cfg,
            'initialized': True,
            'available': True
        })
        print(f"[OK] VAE initialized successfully on {device}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize VAE: {e}")
        vae_resources['initialized'] = True
        vae_resources['available'] = False

    return vae_resources


def translate_with_vae(text, gender='neutral', top_k=5, rerank=False):
    """Translate sentence using VAE latent blending of top-k matches."""
    vae = get_vae_resources()
    if not vae['available']:
        return None

    matcher = get_sentence_matcher()
    matches = matcher.search(text, top_k=top_k, rerank=rerank)
    
    if not matches:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z_list = []
    weights = []
    
    cache = vae['cache']
    for m in matches:
        key = m['pkl_file']
        if key in cache:
            z_list.append(cache[key])
            weights.append(m['similarity'])
    
    if not z_list:
        return None

    # Weighted blend in latent space
    z = np.stack(z_list, axis=0)
    w = np.asarray(weights, dtype=np.float32)
    w = w / (w.sum() + 1e-8)
    z_blend = (z * w[:, None]).sum(axis=0).astype(np.float32)
    
    # Decode
    with torch.no_grad():
        z_tensor = torch.from_numpy(z_blend).unsqueeze(0).to(device)
        pred_norm = vae['model'].decode(z_tensor).squeeze(0).cpu().numpy()
    
    # Denormalize
    stats = vae['stats']
    pred = (pred_norm * stats.std) + stats.mean
    
    # Post-process root
    if vae['config'].get("root_relative", True) and pred.shape[1] == 182:
        pred[:, 179:182] = 0.0

    return {
        'pose_sequence': pred,
        'strategy': 'vae_blend',
        'confidence': float(np.mean(weights)),
        'matches': matches
    }


def chunk_transcript_by_timestamps(transcript_list, max_gap=0.8, max_chunk_words=25):
    """
    Group YouTube transcript entries into chunks using timestamp gaps.
    
    Instead of splitting on punctuation (which YouTube captions rarely have),
    this uses the natural pauses in speech to find sentence boundaries.
    
    Args:
        transcript_list: Raw YouTube transcript (list of FetchedTranscriptSnippet)
        max_gap: Seconds of silence that triggers a chunk boundary
        max_chunk_words: Hard cap — force a split if chunk gets too long
    
    Returns:
        List of dicts: [{text, start_time, end_time}, ...]
    """
    chunks = []
    current_words = []
    chunk_start = 0.0
    prev_end = 0.0

    for entry in transcript_list:
        start = entry.start
        duration = entry.duration
        text = entry.text.strip()
        
        if not text:
            continue

        gap = start - prev_end
        word_count = len(' '.join(current_words).split()) if current_words else 0

        # Split on pause OR word limit
        if current_words and (gap > max_gap or word_count >= max_chunk_words):
            chunks.append({
                'text': ' '.join(current_words),
                'start_time': chunk_start,
                'end_time': prev_end
            })
            current_words = []
            chunk_start = start

        if not current_words:
            chunk_start = start

        current_words.append(text)
        prev_end = start + duration

    # Flush remaining
    if current_words:
        chunks.append({
            'text': ' '.join(current_words),
            'start_time': chunk_start,
            'end_time': prev_end
        })

    return chunks
def build_sentence_timeline(chunks, start_offset=0):
    """
    Builds a word-level cumulative timeline for a sentence, proportional to word length.
    chunks: list of dicts: [{'text': 'hello world', 'frames': 30}, ...]
    start_offset: starting frame index
    """
    timeline = []
    curr = float(start_offset)
    prefix = ""
    for idx, chunk in enumerate(chunks):
        fc = chunk['frames']
        if fc <= 0: continue
        words = str(chunk['text']).split()
        if not words:
            timeline.append({'start_frame': int(round(curr)), 'end_frame': int(round(curr + fc)) - 1, 'text': prefix})
            curr += fc
            if idx < len(chunks) - 1:
                curr += 6
            continue
            
        total_chars = sum(len(w) for w in words)
        for i, w in enumerate(words):
            frames = fc * (len(w) / total_chars) if total_chars > 0 else fc / len(words)
            start = int(round(curr))
            curr += frames
            end = int(round(curr)) - 1
            
            if i == len(words) - 1 and idx < len(chunks) - 1:
                end += 6
                curr += 6
                
            prefix = (prefix + " " + w).strip()
            timeline.append({'start_frame': start, 'end_frame': max(start, end), 'text': prefix})
            
    return timeline, curr


def blend_adjacent_chunks(pose_sequences, blend_frames=6):
    """
    Smooth transitions between chunk boundaries using linear interpolation.
    
    Instead of hard-cutting between adjacent pose arrays (which causes visible
    jumps in the animation), this inserts short interpolated transitions.
    
    Args:
        pose_sequences: List of [T_i, D] numpy arrays (D is typically 182)
        blend_frames: Number of interpolation frames between chunks
    
    Returns:
        Single [T_total, D] array with smooth transitions
    """
    if not pose_sequences:
        return np.empty((0, 182))
    if len(pose_sequences) == 1:
        return pose_sequences[0]
    
    result = [pose_sequences[0]]
    
    for i in range(1, len(pose_sequences)):
        end_pose = pose_sequences[i - 1][-1]   # Last frame of previous chunk
        start_pose = pose_sequences[i][0]       # First frame of next chunk
        
        # Linear interpolation between boundary frames
        transition = np.zeros((blend_frames, end_pose.shape[0]))
        for f in range(blend_frames):
            alpha = (f + 1) / (blend_frames + 1)
            transition[f] = (1 - alpha) * end_pose + alpha * start_pose
        
        result.append(transition)
        result.append(pose_sequences[i])
    
    return np.vstack(result)

def _extract_smplx_params_array(pose_data):
    """Return a numpy array of shape [T, D] from pose_data loaded by SentenceToSMPLX/WordToSMPLX."""
    smplx_params = pose_data.get('smplx')

    # Some loaders return {'smplx': {'smooth_smplx': ..., ...}}
    if isinstance(smplx_params, dict):
        if 'smooth_smplx' in smplx_params:
            smplx_params = smplx_params['smooth_smplx']
        else:
            first_key = list(smplx_params.keys())[0]
            smplx_params = smplx_params[first_key]

    if torch.is_tensor(smplx_params):
        smplx_params = smplx_params.detach().cpu().numpy()

    if isinstance(smplx_params, list):
        smplx_params = np.stack(smplx_params)

    smplx_params = np.asarray(smplx_params)
    if smplx_params.ndim != 2:
        raise ValueError(f"Unexpected smplx params shape: {smplx_params.shape}")
    return smplx_params


def _peek_num_frames_from_pkl(pkl_path: str):
    """Best-effort: return number of frames in a How2Sign pose pkl without fully processing tensors."""

    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: _torch_load_compat(io.BytesIO(b), map_location='cpu')
            return super().find_class(module, name)

    with open(pkl_path, 'rb') as f:
        data = CPU_Unpickler(f).load()

    if not isinstance(data, dict) or 'smplx' not in data:
        return None

    smplx = data.get('smplx')
    if isinstance(smplx, dict) and smplx:
        smplx = smplx.get('smooth_smplx', next(iter(smplx.values())))

    try:
        if isinstance(smplx, list) or isinstance(smplx, tuple):
            return len(smplx)
        if hasattr(smplx, 'shape') and len(smplx.shape) >= 1:
            return int(smplx.shape[0])
    except Exception:
        return None
    return None

# --- Helper: Extract YouTube video ID ---
def extract_video_id(url):
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    elif len(url) == 11:
        return url
    else:
        raise ValueError("Invalid YouTube URL or video ID.")

def transcript_to_words(transcript_list):
    # transcript_list: FetchedTranscript iterable with FetchedTranscriptSnippet objects
    words = []
    for entry in transcript_list:
        text = entry.text if hasattr(entry, 'text') else str(entry)
        for w in text.lower().split():
            w_clean = w.strip(string.punctuation)
            if w_clean in dataset_words and w_clean not in words:
                words.append(w_clean)
    return words

def extract_full_transcript_text(transcript_list):
    """Extract full transcript text for display"""
    return ' '.join([entry.text if hasattr(entry, 'text') else str(entry) for entry in transcript_list])

def create_word_mapping(transcript_list, dataset_words):
    """Create detailed word-by-word mapping with status"""
    word_map = []
    for entry in transcript_list:
        text = entry.text if hasattr(entry, 'text') else str(entry)
        for raw_word in text.split():
            word_clean = raw_word.lower().strip(string.punctuation)
            if word_clean:  # Skip empty strings
                status = 'available' if word_clean in dataset_words else 'missing'
                word_map.append({
                    'original': raw_word,
                    'clean': word_clean,
                    'status': status
                })
    return word_map

# --- API Endpoints (Core) ---

@app.route('/')
def home():
    return jsonify({"status": "ok", "message": "SMPL-X ASL API running. Use the Vite frontend on port 5173."})

@app.route('/api/available_words')
def available_words():
    return jsonify(sorted(list(dataset_words)))

@app.route('/api/list_poses')
def list_poses():
    try:
        from poses_to_animation import PoseAssembler
        poses_dir = os.path.join(current_dir, "..", "..", "data", "raw_poses", "poses")
        assembler = PoseAssembler(poses_dir)
        return jsonify(assembler.list_folders())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/render_poses', methods=['POST'])
def render_pose_api():
    try:
        from poses_to_animation import render_pose_folder
        data = request.get_json()
        folder_name = data.get('folder')
        gender = data.get('gender', 'neutral')
        
        if not folder_name:
            return jsonify({"error": "No folder specified"}), 400
            
        poses_dir = os.path.join(current_dir, "..", "..", "data", "raw_poses", "poses")
        output_filename = f"pose_{folder_name}_{gender}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # Assemble and render
        render_pose_folder(
            folder_name, 
            poses_root=poses_dir,
            output_path=output_path,
            gender=gender
        )
        
        return jsonify({
            "status": "success",
            "url": f"/output/{output_filename}",
            "filename": output_filename
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/list_sentences')
def list_sentences():
    try:
        if not os.path.exists(how2sign_mapping_path):
            return jsonify([])
        with open(how2sign_mapping_path, "r", encoding='utf-8') as f:
            mapping = json.load(f)
        
        # Prepare list for frontend
        sentences = []
        for pkl, text in mapping.items():
            if os.path.exists(os.path.join(how2sign_dataset_dir, pkl)):
                sentences.append({"pkl": pkl, "text": text})
        
        return jsonify(sentences)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/render_sentence', methods=['POST'])
def render_sentence_api():
    try:
        data = request.get_json()
        pkl_file = data.get('pkl')
        gender = data.get('gender', 'neutral')
        
        if not pkl_file:
            return jsonify({"error": "No pickle file specified"}), 400
            
        pkl_path = os.path.join(how2sign_dataset_dir, pkl_file)
        if not os.path.exists(pkl_path):
            return jsonify({"error": "Pickle file not found"}), 404
            
        _, sentence_animator = get_animators(gender)
        
        output_filename = f"h2s_{pkl_file.replace('.pkl', '')}_{gender}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        pose_data = sentence_animator.load_pose_sequence(pkl_path)
        
        with open(how2sign_mapping_path, "r", encoding='utf-8') as f:
            mapping = json.load(f)
        text = mapping.get(pkl_file, "")
        
        params = _extract_smplx_params_array(pose_data)
        subtitle_timeline, _ = build_sentence_timeline([{'text': text, 'frames': params.shape[0]}], 0)
        
        sentence_animator.render_animation(pose_data, save_path=output_path, subtitle_timeline=subtitle_timeline)
        
        return jsonify({
            "status": "success",
            "url": f"/output/{output_filename}",
            "filename": output_filename
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/api/render_text_mp4', methods=['POST'])
def render_text_mp4():
    """Render a single input sentence to MP4 and return it directly (video/mp4).

    Request JSON:
      {"text": "I go to school.", "gender": "neutral", "fps": 15, "max_frames": 180, "use_cache": true}
    """
    try:
        data = request.get_json(force=True) or {}
        text = (data.get('text') or '').strip()
        gender = (data.get('gender') or 'neutral').lower()
        fps = int(data.get('fps') or 15)
        max_frames = data.get('max_frames', 180)
        use_cache = bool(data.get('use_cache', True))
        rerank = bool(data.get('rerank', False))

        if not text:
            return jsonify({'error': 'Missing text'}), 400
        if fps <= 0 or fps > 60:
            return jsonify({'error': 'fps must be in 1..60'}), 400

        if max_frames is not None:
            max_frames = int(max_frames)
            if max_frames <= 0:
                return jsonify({'error': 'max_frames must be positive or null'}), 400

        # Deterministic cache key (MP4 response)
        key_material = f"v2_mp4|{gender}|{fps}|{max_frames}|{text.lower()}".encode('utf-8')
        key = hashlib.sha1(key_material).hexdigest()[:16]
        cache_path = os.path.join(text_cache_dir, f"text_{key}.mp4")

        if use_cache and os.path.exists(cache_path):
            return send_file(cache_path, mimetype='video/mp4')

        matcher = get_sentence_matcher()
        _, sentence_animator = get_animators(gender)

        result = matcher.translate_sentence(text, verbose=False, rerank=rerank)
        if result.get('strategy') == 'failed' or not result.get('matches'):
            return jsonify({'error': 'No matches found', 'details': result}), 404

        # Heuristic: if chunking finds <2 usable chunks, prefer a single full-sentence match
        if result.get('strategy') == 'chunked' and len(result.get('matches', [])) < 2:
            best_full = matcher.search(text, top_k=1, rerank=rerank)
            if best_full:
                result = {
                    'strategy': 'fallback',
                    'matches': [best_full[0]],
                    'confidence': float(best_full[0].get('similarity', 0.0)),
                    'input_sentence': text,
                    'warning': 'Chunking produced too few matches; using best full-sentence match'
                }

        pose_sequences = []
        valid_chunks = []
        if result['strategy'] == 'chunked':
            for chunk_match in result['matches']:
                match = chunk_match['match']
                pose_data = sentence_animator.load_pose_sequence(match['pkl_path'])
                params = _extract_smplx_params_array(pose_data)
                pose_sequences.append(params)
                valid_chunks.append({'text': match.get('sentence', chunk_match.get('input_chunk', '')), 'frames': params.shape[0]})
        else:
            match = result['matches'][0]
            pose_data = sentence_animator.load_pose_sequence(match['pkl_path'])
            params = _extract_smplx_params_array(pose_data)
            pose_sequences.append(params)
            valid_chunks.append({'text': text, 'frames': params.shape[0]})

        all_params = blend_adjacent_chunks(pose_sequences, blend_frames=6)
        pose_data_out = {'smplx': all_params, 'gender': gender, 'fps': fps}
        
        subtitle_timeline, _ = build_sentence_timeline(valid_chunks, 0)

        sentence_animator.render_animation(
            pose_data_out,
            save_path=cache_path,
            fps=fps,
            max_frames=max_frames,
            subtitle_timeline=subtitle_timeline
        )

        return send_file(cache_path, mimetype='video/mp4')

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/render_text', methods=['POST'])
def render_text_json():
    """Render a single input sentence, return JSON with match metadata + cached video URL."""
    try:
        data = request.get_json(force=True, silent=True)
        if data is None:
            raw = request.get_data(cache=False, as_text=True) or ''
            return jsonify({
                'error': 'Invalid JSON body',
                'hint': 'Send valid JSON with double-quoted keys/strings, e.g. {"text":"I love this"}',
                'received_prefix': raw[:200],
            }), 400
        data = data or {}
        text = (data.get('text') or '').strip()
        gender = (data.get('gender') or 'neutral').lower()
        fps = int(data.get('fps') or 15)
        max_frames = data.get('max_frames', 180)
        use_cache = bool(data.get('use_cache', True))
        min_frames = data.get('min_frames', None)
        candidate_k = int(data.get('candidate_k') or 5)
        rerank = bool(data.get('rerank', False))
        use_vae = bool(data.get('use_vae', False))

        if not text:
            return jsonify({'error': 'Missing text'}), 400
        if fps <= 0 or fps > 60:
            return jsonify({'error': 'fps must be in 1..60'}), 400

        if max_frames is not None:
            max_frames = int(max_frames)
            if max_frames <= 0:
                return jsonify({'error': 'max_frames must be positive or null'}), 400

        if min_frames is not None:
            min_frames = int(min_frames)
            if min_frames <= 0:
                return jsonify({'error': 'min_frames must be positive or null'}), 400

        if candidate_k < 1 or candidate_k > 25:
            return jsonify({'error': 'candidate_k must be in 1..25'}), 400

        key_material = f"v2_mp4|{gender}|{fps}|{max_frames}|{use_vae}|{text.lower()}".encode('utf-8')
        key = hashlib.sha1(key_material).hexdigest()[:16]
        cache_filename = f"text_{key}.mp4"
        cache_path = os.path.join(text_cache_dir, cache_filename)

        matcher = get_sentence_matcher()
        debug = None
        
        # --- Strategy Selection ---
        result = None
        vae_res = None
        
        if use_vae:
            vae_res = translate_with_vae(text, gender=gender, top_k=candidate_k, rerank=rerank)
            if vae_res:
                result = {
                    'strategy': 'vae_blend',
                    'matches': vae_res['matches'],
                    'confidence': vae_res['confidence'],
                    'input_sentence': text,
                    'warning': 'VAE latent blending used'
                }
            else:
                print("[INFO] VAE failed or not available, falling back to standard matching")

        if result is None:
            result = matcher.translate_sentence(text, verbose=False, rerank=rerank)
        if result.get('strategy') == 'failed' or not result.get('matches'):
            return jsonify({'error': 'No matches found', 'details': result}), 404

        # Optional: if the best match is very short, try to pick a longer near-best candidate.
        if min_frames is not None and result.get('strategy') in ('full', 'fallback'):
            best = result['matches'][0]
            best_frames = _peek_num_frames_from_pkl(best['pkl_path'])
            if best_frames is not None and best_frames < min_frames:
                candidates = matcher.search(text, top_k=candidate_k, rerank=rerank)
                # Require at least medium confidence; also keep candidates reasonably close to best.
                best_sim = float(best.get('similarity', 0.0) or 0.0)
                min_sim = max(matcher.MEDIUM_CONFIDENCE, best_sim - 0.10)
                chosen = None
                chosen_frames = None
                for cand in candidates:
                    sim = float(cand.get('similarity', 0.0) or 0.0)
                    if sim < min_sim:
                        continue
                    n = _peek_num_frames_from_pkl(cand['pkl_path'])
                    if n is None:
                        continue
                    if n >= min_frames and (chosen is None or n > chosen_frames):
                        chosen = cand
                        chosen_frames = n
                debug = {
                    'min_frames': min_frames,
                    'candidate_k': candidate_k,
                    'best': {
                        'pkl_file': best.get('pkl_file'),
                        'similarity': float(best.get('similarity', 0.0) or 0.0),
                        'frames': best_frames,
                    },
                    'chosen': None if chosen is None else {
                        'pkl_file': chosen.get('pkl_file'),
                        'similarity': float(chosen.get('similarity', 0.0) or 0.0),
                        'frames': chosen_frames,
                    },
                    'min_sim': float(min_sim),
                }
                if chosen is not None:
                    result = {
                        'strategy': 'full',
                        'matches': [chosen],
                        'confidence': float(chosen.get('similarity', 0.0) or 0.0),
                        'input_sentence': text,
                        'warning': f"Chose longer near-best match ({chosen_frames} frames) over short best ({best_frames} frames)"
                    }

        if result.get('strategy') == 'chunked' and len(result.get('matches', [])) < 2:
            best_full = matcher.search(text, top_k=1, rerank=rerank)
            if best_full:
                result = {
                    'strategy': 'fallback',
                    'matches': [best_full[0]],
                    'confidence': float(best_full[0].get('similarity', 0.0)),
                    'input_sentence': text,
                    'warning': 'Chunking produced too few matches; using best full-sentence match'
                }

        if not (use_cache and os.path.exists(cache_path)):
            _, sentence_animator = get_animators(gender)
            pose_sequences = []
            valid_chunks = []
            
            if result['strategy'] == 'vae_blend' and vae_res:
                params = vae_res['pose_sequence']
                pose_sequences.append(params)
                valid_chunks.append({'text': text, 'frames': params.shape[0]})
            elif result['strategy'] == 'chunked':
                for chunk_match in result['matches']:
                    match = chunk_match['match']
                    pose_data = sentence_animator.load_pose_sequence(match['pkl_path'])
                    params = _extract_smplx_params_array(pose_data)
                    pose_sequences.append(params)
                    valid_chunks.append({'text': match.get('sentence', chunk_match.get('input_chunk', '')), 'frames': params.shape[0]})
            else:
                match = result['matches'][0]
                pose_data = sentence_animator.load_pose_sequence(match['pkl_path'])
                params = _extract_smplx_params_array(pose_data)
                pose_sequences.append(params)
                valid_chunks.append({'text': text, 'frames': params.shape[0]})

            all_params = blend_adjacent_chunks(pose_sequences, blend_frames=6)
            pose_data_out = {'smplx': all_params, 'gender': gender, 'fps': fps}
            
            subtitle_timeline, _ = build_sentence_timeline(valid_chunks, 0)
            
            sentence_animator.render_animation(
                pose_data_out,
                save_path=cache_path,
                fps=fps,
                max_frames=max_frames,
                subtitle_timeline=subtitle_timeline
            )

        # Summarize match info for debugging
        if result['strategy'] == 'chunked':
            match_summary = [
                {
                    'chunk': cm.get('input_chunk', ''),
                    'sentence': cm.get('match', {}).get('sentence', ''),
                    'similarity': cm.get('match', {}).get('similarity', 0.0),
                    'pkl_file': cm.get('match', {}).get('pkl_file', '')
                }
                for cm in result.get('matches', [])
            ]
        else:
            m = result['matches'][0]
            match_summary = {
                'sentence': m.get('sentence', ''),
                'similarity': m.get('similarity', 0.0),
                'pkl_file': m.get('pkl_file', '')
            }

        return jsonify({
            'status': 'success',
            'text': text,
            'gender': gender,
            'fps': fps,
            'max_frames': max_frames,
            'strategy': result.get('strategy'),
            'confidence': float(result.get('confidence', 0.0) or 0.0),
            'match': match_summary,
            'warning': result.get('warning'),
            'debug': debug,
            'url': request.host_url.rstrip('/') + f"/output/text_cache/{cache_filename}",
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/output/text_cache/<path:filename>')
def download_text_cache(filename):
    return send_from_directory(text_cache_dir, filename)

# --- Endpoint: Extract transcript only (preview) ---
@app.route('/extract_transcript', methods=['POST'])
def extract_transcript():
    """Extract and analyze transcript without generating video"""
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({'error': 'Missing YouTube URL'}), 400
    
    try:
        video_id = extract_video_id(url)
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(video_id)
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        return jsonify({'error': f'No transcript available for this video: {str(e)}'}), 404
    except Exception as e:
        return jsonify({'error': f'Error fetching transcript: {str(e)}'}), 500
    
    # Extract full transcript text and create word mapping
    full_transcript = extract_full_transcript_text(transcript_list)
    word_mapping = create_word_mapping(transcript_list, dataset_words)
    
    # Count statistics
    available_words = [w for w in word_mapping if w['status'] == 'available']
    missing_words = [w for w in word_mapping if w['status'] == 'missing']
    unique_available = list(dict.fromkeys([w['clean'] for w in available_words]))
    
    return jsonify({
        'transcript': full_transcript,
        'word_mapping': word_mapping,
        'total_words': len(word_mapping),
        'available_count': len(available_words),
        'missing_count': len(missing_words),
        'unique_available': unique_available,
        'video_id': video_id
    })

# --- Endpoint: Get transcript from YouTube ---
@app.route('/asl_from_youtube', methods=['POST'])
def asl_from_youtube():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({'error': 'Missing YouTube URL'}), 400
    try:
        video_id = extract_video_id(url)
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(video_id)
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        return jsonify({'error': f'No transcript available for this video: {str(e)}'}), 404
    except Exception as e:
        return jsonify({'error': f'Error fetching transcript: {str(e)}'}), 500

    # Extract full transcript text and create word mapping
    full_transcript = extract_full_transcript_text(transcript_list)
    word_mapping = create_word_mapping(transcript_list, dataset_words)
    
    words = transcript_to_words(transcript_list)
    if not words:
        return jsonify({
            'error': 'No recognizable ASL words found in transcript.',
            'transcript': full_transcript,
            'word_mapping': word_mapping
        }), 400

    video_filename = f"{'_'.join(words[:5])}_asl.mp4"  # Limit filename length
    video_path = os.path.join(output_dir, video_filename)
    if os.path.exists(video_path):
        return jsonify({'url': f"/output/{video_filename}", 'words': words})

    # Retrieve requested gender and initialize model lazily
    gender = data.get('gender', 'neutral').lower()
    animator, _ = get_animators(gender)

    # Load and concatenate pose data with comprehensive error handling
    pose_data_sequences = []
    successful_words = []
    skipped_words = []
    
    for word in words:
        try:
            pkl_file = os.path.join(dataset_dir, word_to_pkl[word])
            
            # Suppress stdout/stderr during loading to hide torch warnings
            import sys
            import io
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            
            try:
                pose_data_dict = animator.load_pose_sequence(pkl_file)
                smplx_params_np = np.stack(pose_data_dict['smplx'])
                pose_data_sequences.append(smplx_params_np)
                successful_words.append(word)
            finally:
                sys.stderr = old_stderr
                
        except RuntimeError as e:
            # CUDA deserialization error - skip silently
            if 'cuda' in str(e).lower():
                skipped_words.append(word)
                continue
            else:
                # Other runtime errors - log and skip
                print(f"RuntimeError for '{word}': {str(e)[:100]}")
                skipped_words.append(word)
                continue
        except Exception as e:
            # Any other error - log and skip
            print(f"Error loading '{word}': {type(e).__name__}: {str(e)[:100]}")
            skipped_words.append(word)
            continue
    
    if not pose_data_sequences:
        return jsonify({
            'error': 'No pose data could be loaded from transcript',
            'attempted_words': words,
            'skipped_words': skipped_words
        }), 400
    
    # Create valid_chunks for word-level subtitles
    valid_chunks = [{'text': w, 'frames': p.shape[0]} for w, p in zip(successful_words, pose_data_sequences)]
    subtitle_timeline, _ = build_sentence_timeline(valid_chunks, 0)
    
    # Concatenate all sequences
    all_params = blend_adjacent_chunks(pose_data_sequences, blend_frames=6)
    
    # Create proper pose_data structure
    pose_data = {
        'smplx': all_params,
        'gender': gender,
        'fps': 15
    }
    
    animator.render_animation(pose_data, save_path=video_path, fps=15, subtitle_timeline=subtitle_timeline)
    
    response_data = {
        'url': f"/output/{video_filename}",
        'words': successful_words,
        'total_recognized': len(words),
        'total_processed': len(successful_words),
        'transcript': full_transcript,
        'word_mapping': word_mapping
    }
    
    if skipped_words:
        response_data['skipped_words'] = skipped_words
        response_data['skipped_count'] = len(skipped_words)
    
    return jsonify(response_data)

# --- Endpoint: Stream ASL video (no disk save) ---
@app.route('/asl_stream', methods=['POST'])
def asl_stream():
    from flask import Response
    from io import StringIO
    
    try:
        data = request.get_json()
        words = data.get('words', [])
        
        if not words:
            return jsonify({'error': 'No words provided'}), 400
        
        # Validate words
        invalid_words = [w for w in words if w.lower() not in dataset_words]
        if invalid_words:
            return jsonify({'error': f'Invalid words: {", ".join(invalid_words)}'}), 400
        
        # Retrieve requested gender and initialize model lazily
        gender = data.get('gender', 'neutral').lower()
        animator, _ = get_animators(gender)
        
        # Load and concatenate pose data
        pose_data_sequences = []
        successful_words = []
        
        for word in words:
            try:
                pkl_file = os.path.join(dataset_dir, word_to_pkl[word.lower()])
                
                # Suppress warnings
                import sys
                old_stderr = sys.stderr
                sys.stderr = StringIO()
                
                try:
                    pose_data_dict = animator.load_pose_sequence(pkl_file)
                    smplx_params_np = np.stack(pose_data_dict['smplx'])
                    pose_data_sequences.append(smplx_params_np)
                    successful_words.append(word)
                finally:
                    sys.stderr = old_stderr
                    
            except Exception as e:
                print(f"Error loading '{word}': {type(e).__name__}: {str(e)[:100]}")
                continue
        
        if not pose_data_sequences:
            return jsonify({'error': 'No pose data could be loaded'}), 400
        
        # Concatenate all sequences
        all_params = np.vstack(pose_data_sequences)
        
        # Create pose_data structure
        pose_data = {
            'smplx': all_params,
            'gender': gender,
            'fps': 15
        }
        
        # Render to bytes (in-memory)
        print(f"[STREAM] Rendering {len(successful_words)} word(s): {', '.join(successful_words)}")
        print(f"[STREAM] Total frames: {all_params.shape[0]}")
        
        video_bytes = animator.render_animation_to_bytes(pose_data, fps=15)
        
        print(f"[STREAM] Video generated: {len(video_bytes)} bytes")
        return Response(video_bytes, mimetype='video/mp4')
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] Stream endpoint failed: {error_trace}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# --- Endpoint: ASL from YouTube using sentence-level matching (30K dataset) ---
@app.route('/asl_from_youtube_sentences', methods=['POST'])
def asl_from_youtube_sentences():
    """
    Generate ASL video from YouTube transcript using semantic sentence matching.
    Uses the 30K How2Sign sentence dataset for more accurate translations.
    """
    data = request.get_json()
    url = data.get('url')
    max_sentences = data.get('max_sentences', 10)  # Limit to avoid long processing
    include_subtitles = data.get('include_subtitles', True)
    
    if not url:
        return jsonify({'error': 'Missing YouTube URL'}), 400
    
    try:
        # Extract video ID and fetch transcript
        video_id = extract_video_id(url)
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(video_id)
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        return jsonify({'error': f'No transcript available for this video: {str(e)}'}), 404
    except Exception as e:
        return jsonify({'error': f'Error fetching transcript: {str(e)}'}), 500
    
    # Extract full transcript
    full_transcript = extract_full_transcript_text(transcript_list)
    
    # Split transcript into sentences using timestamp-aware chunking
    transcript_chunks = chunk_transcript_by_timestamps(transcript_list)
    sentences = [c['text'] for c in transcript_chunks]
    
    # Fallback: if timestamp chunking produced nothing (e.g. no timing data),
    # fall back to simple punctuation splitting
    if not sentences:
        import re
        sentences = re.split(r'[.!?]+', full_transcript)
        sentences = [s.strip() for s in sentences if s.strip()]
    
    # Limit number of sentences
    if len(sentences) > max_sentences:
        print(f"[INFO] Limiting to first {max_sentences} of {len(sentences)} sentences")
        sentences = sentences[:max_sentences]
        truncated = True
    else:
        truncated = False
    
    print(f"[INFO] Processing {len(sentences)} sentences with semantic matching")
    
    # Initialize sentence matcher
    matcher = get_sentence_matcher()
    
    # Retrieve requested gender and initialized model
    gender = data.get('gender', 'neutral').lower()
    _, sentence_animator = get_animators(gender)
    
    # Match each sentence
    use_vae = bool(data.get('use_vae', False))
    translation_results = []
    pose_sequences = []
    
    for idx, sentence in enumerate(sentences):
        print(f"\n[{idx+1}/{len(sentences)}] Processing: {sentence[:80]}...")
        
        try:
            # Try VAE if requested
            vae_res = None
            if use_vae:
                vae_res = translate_with_vae(sentence, gender=gender, top_k=5)
            
            if vae_res:
                result = {
                    'strategy': 'vae_blend',
                    'matches': vae_res['matches'],
                    'confidence': vae_res['confidence'],
                    'input_sentence': sentence
                }
            else:
                result = matcher.translate_sentence(sentence, verbose=True)
            
            # Build enhanced result with all fields Streamlit expects
            frame_count = 0
            matched_text = ""
            alternatives = []
            
            # Load pose data based on strategy
            if result['strategy'] == 'vae_blend' and vae_res:
                matched_text = f"VAE Blend ({len(vae_res['matches'])} matches)"
                smplx_params = vae_res['pose_sequence']
                frame_count = smplx_params.shape[0]
                pose_sequences.append(smplx_params)
                alternatives = [{'text': m['sentence'], 'confidence': m['similarity']} for m in result['matches']]

            elif result['strategy'] == 'full':
                # Single sentence match
                match = result['matches'][0]
                matched_text = match['sentence']  # Changed from 'text' to 'sentence'
                alternatives = [{'text': m['sentence'], 'confidence': m['similarity']} for m in result['matches'][1:4]]
                
                pose_data = sentence_animator.load_pose_sequence(match['pkl_path'])
                
                # Extract smooth_smplx if available, otherwise use raw smplx
                if 'smplx' in pose_data and isinstance(pose_data['smplx'], dict):
                    if 'smooth_smplx' in pose_data['smplx']:
                        smplx_params = pose_data['smplx']['smooth_smplx']
                    else:
                        # Fallback to first available key
                        first_key = list(pose_data['smplx'].keys())[0]
                        smplx_params = pose_data['smplx'][first_key]
                else:
                    smplx_params = pose_data['smplx']
                
                # Convert to numpy if tensor
                if torch.is_tensor(smplx_params):
                    smplx_params = smplx_params.cpu().numpy()
                
                frame_count = smplx_params.shape[0]
                pose_sequences.append(smplx_params)
                
            elif result['strategy'] == 'chunked':
                # Multiple phrase matches - concatenate
                matched_chunks = []
                for chunk_match in result['matches']:
                    match = chunk_match['match']
                    matched_chunks.append(match['sentence'])  # Changed from 'text' to 'sentence'
                    
                    pose_data = sentence_animator.load_pose_sequence(match['pkl_path'])
                    
                    # Extract smooth_smplx if available
                    if 'smplx' in pose_data and isinstance(pose_data['smplx'], dict):
                        if 'smooth_smplx' in pose_data['smplx']:
                            smplx_params = pose_data['smplx']['smooth_smplx']
                        else:
                            first_key = list(pose_data['smplx'].keys())[0]
                            smplx_params = pose_data['smplx'][first_key]
                    else:
                        smplx_params = pose_data['smplx']
                    
                    if torch.is_tensor(smplx_params):
                        smplx_params = smplx_params.cpu().numpy()
                    
                    frame_count += smplx_params.shape[0]
                    pose_sequences.append(smplx_params)
                
                matched_text = " + ".join(matched_chunks)
            
            elif result['strategy'] == 'fallback':
                # Low quality match - still use it
                match = result['matches'][0]
                matched_text = match['sentence']  # Changed from 'text' to 'sentence'
                alternatives = [{'text': m['sentence'], 'confidence': m['similarity']} for m in result['matches'][1:4]]
                
                pose_data = sentence_animator.load_pose_sequence(match['pkl_path'])
                
                if 'smplx' in pose_data and isinstance(pose_data['smplx'], dict):
                    if 'smooth_smplx' in pose_data['smplx']:
                        smplx_params = pose_data['smplx']['smooth_smplx']
                    else:
                        first_key = list(pose_data['smplx'].keys())[0]
                        smplx_params = pose_data['smplx'][first_key]
                else:
                    smplx_params = pose_data['smplx']
                
                if torch.is_tensor(smplx_params):
                    smplx_params = smplx_params.cpu().numpy()
                
                frame_count = smplx_params.shape[0]
                pose_sequences.append(smplx_params)
            
            # Add enhanced result
            translation_results.append({
                'original': sentence,
                'match': matched_text,
                'confidence': result.get('confidence', 0.0),
                'strategy': result['strategy'],
                'frames': frame_count,
                'alternatives': alternatives
            })
                
        except Exception as e:
            import traceback
            print(f"[ERROR] Failed to process sentence: {str(e)}")
            traceback.print_exc()
            translation_results.append({
                'strategy': 'error',
                'original': sentence,
                'match': '',
                'confidence': 0.0,
                'frames': 0,
                'alternatives': [],
                'error': str(e)
            })
            continue
    
    if not pose_sequences:
        return jsonify({
            'error': 'No sentences could be matched to ASL animations',
            'transcript': full_transcript,
            'sentences_attempted': len(sentences),
            'translation_results': translation_results
        }), 400
    
    # Concatenate all pose sequences with smooth transitions between chunks
    print(f"\n[INFO] Blending {len(pose_sequences)} pose sequences with transitions...")
    all_params = blend_adjacent_chunks(pose_sequences, blend_frames=6)
    print(f"[INFO] Total frames: {all_params.shape[0]}")

    # Build frame-aligned subtitle timeline from successful sentence matches.
    subtitle_timeline = []
    if include_subtitles:
        current_frame = 0
        for result in translation_results:
            if result.get('strategy') == 'error':
                continue
            
            # Since a sentence could have been generated from chunks, we treat the sentence as one chunk here
            # Or if it's chunked, the 'frames' is total frames.
            fc = int(result.get('frames', 0) or 0)
            if fc > 0:
                sentence_timeline, current_frame = build_sentence_timeline([{'text': result.get('original', ''), 'frames': fc}], current_frame)
                subtitle_timeline.extend(sentence_timeline)
    
    # Create pose_data structure
    pose_data = {
        'smplx': all_params,
        'gender': gender,
        'fps': 15
    }
    
    # Generate output filename
    video_filename = f"youtube_sentences_{video_id}_{len(translation_results)}.mp4"
    video_path = os.path.join(output_dir, video_filename)
    
    # Render animation
    print(f"[INFO] Rendering animation to {video_filename}...")
    sentence_animator.render_animation(
        pose_data,
        save_path=video_path,
        fps=15,
        subtitle_timeline=subtitle_timeline
    )
    
    # Calculate statistics
    successful = [r for r in translation_results if r['strategy'] != 'error']
    high_conf = [r for r in successful if r.get('confidence', 0) >= SentenceMatcher.HIGH_CONFIDENCE]
    medium_conf = [r for r in successful if SentenceMatcher.MEDIUM_CONFIDENCE <= r.get('confidence', 0) < SentenceMatcher.HIGH_CONFIDENCE]
    low_conf = [r for r in successful if r.get('confidence', 0) < SentenceMatcher.MEDIUM_CONFIDENCE]
    
    full_strategies = [r['strategy'] for r in successful]
    strategy_counts = {
        'full': full_strategies.count('full'),
        'chunked': full_strategies.count('chunked'),
        'fallback': full_strategies.count('fallback')
    }
    
    video_duration = all_params.shape[0] / 15.0  # fps = 15
    coverage = (len(successful) / len(sentences) * 100) if sentences else 0
    
    return jsonify({
        'url': f"/output/{video_filename}",
        'video_id': video_id,
        'transcript': full_transcript,
        'sentences': translation_results,
        'statistics': {
            'sentences_processed': len(sentences),
            'sentences_successful': len(successful),
            'sentences_failed': len(translation_results) - len(successful),
            'coverage_percentage': coverage,
            'total_frames': int(all_params.shape[0]),
            'video_duration_seconds': video_duration,
            'confidence_breakdown': {
                'high': len(high_conf),
                'medium': len(medium_conf),
                'low': len(low_conf)
            },
            'strategy_breakdown': strategy_counts,
            'avg_confidence': float(np.mean([r.get('confidence', 0) for r in successful])) if successful else 0.0,
            'vae_used': use_vae
        },
        'truncated': truncated,
        'subtitles_enabled': bool(include_subtitles),
        'note': f"This translation uses semantic sentence matching from 30K+ How2Sign dataset{' with VAE blending' if use_vae else ''}"
    })

# --- Serve generated videos ---
@app.route('/output/<path:filename>')
def download_file(filename):
    return send_from_directory(output_dir, filename)


@app.route('/health')
def health():
    return "ok"


if __name__ == '__main__':
    app.run(port=5000, debug=False, use_reloader=False)