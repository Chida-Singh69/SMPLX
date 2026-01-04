import os
import json
import re
import string
import numpy as np
import torch

from flask import Flask, request, jsonify, send_from_directory
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from word_to_smplx import WordToSMPLX

app = Flask(__name__)

# --- Setup paths and load resources once ---
current_dir = os.path.dirname(os.path.abspath(__file__))
mapping_path = os.path.join(current_dir, "filtered_video_to_gloss.json")
dataset_dir = os.path.join(current_dir, "word-level-dataset-cpu-fixed")
output_dir = os.path.join(current_dir, "output")
os.makedirs(output_dir, exist_ok=True)

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

animator = WordToSMPLX(model_path=os.path.join(current_dir, "models"))

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

    words = transcript_to_words(transcript_list)
    if not words:
        return jsonify({'error': 'No recognizable ASL words found in transcript.'}), 400

    video_filename = f"{'_'.join(words[:5])}_asl.mp4"  # Limit filename length
    video_path = os.path.join(output_dir, video_filename)
    if os.path.exists(video_path):
        return jsonify({'url': f"/output/{video_filename}", 'words': words})

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
    
    # Concatenate all sequences
    all_params = np.vstack(pose_data_sequences)
    
    # Create proper pose_data structure
    pose_data = {
        'smplx': all_params,
        'gender': 'neutral',
        'fps': 15
    }
    
    animator.render_animation(pose_data, save_path=video_path, fps=15)
    
    response_data = {
        'url': f"/output/{video_filename}",
        'words': successful_words,
        'total_recognized': len(words),
        'total_processed': len(successful_words)
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
            'gender': 'neutral',
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

# --- Serve generated videos ---
@app.route('/output/<path:filename>')
def download_file(filename):
    return send_from_directory(output_dir, filename)

@app.route('/')
def home():
    return "SMPLX ASL Backend is running. Use the /asl_from_youtube endpoint."

if __name__ == '__main__':
    app.run(port=5000, debug=True)