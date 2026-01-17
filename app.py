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
from sentence_to_smplx import SentenceToSMPLX
from sentence_matcher import SentenceMatcher

app = Flask(__name__)

# --- Setup paths and load resources once ---
current_dir = os.path.dirname(os.path.abspath(__file__))
mapping_path = os.path.join(current_dir, "filtered_video_to_gloss.json")
dataset_dir = os.path.join(current_dir, "word-level-dataset-cpu-fixed")
how2sign_mapping_path = os.path.join(current_dir, "how2sign_mapping.json")
how2sign_dataset_dir = os.path.join(current_dir, "how2sign_pkls_cropTrue_shapeFalse")
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

# Initialize animators
animator = WordToSMPLX(model_path=os.path.join(current_dir, "models"))
sentence_animator = SentenceToSMPLX(model_path=os.path.join(current_dir, "models"))

# Initialize sentence matcher (lazy-loaded on first use)
sentence_matcher = None

def get_sentence_matcher():
    """Lazy-load sentence matcher on first use."""
    global sentence_matcher
    if sentence_matcher is None:
        print("[INFO] Initializing sentence matcher...")
        sentence_matcher = SentenceMatcher(how2sign_mapping_path, how2sign_dataset_dir)
    return sentence_matcher

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
    
    # Split transcript into sentences (simple approach)
    # TODO: Could use more sophisticated sentence splitting
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
    
    # Match each sentence
    translation_results = []
    pose_sequences = []
    
    for idx, sentence in enumerate(sentences):
        print(f"\n[{idx+1}/{len(sentences)}] Processing: {sentence[:80]}...")
        
        try:
            result = matcher.translate_sentence(sentence, verbose=True)
            
            # Build enhanced result with all fields Streamlit expects
            frame_count = 0
            matched_text = ""
            alternatives = []
            
            # Load pose data based on strategy
            if result['strategy'] == 'full':
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
    
    # Concatenate all pose sequences
    print(f"\n[INFO] Concatenating {len(pose_sequences)} pose sequences...")
    all_params = np.vstack(pose_sequences)
    print(f"[INFO] Total frames: {all_params.shape[0]}")
    
    # Create pose_data structure
    pose_data = {
        'smplx': all_params,
        'gender': 'neutral',
        'fps': 15
    }
    
    # Generate output filename
    video_filename = f"youtube_sentences_{video_id}_{len(translation_results)}.mp4"
    video_path = os.path.join(output_dir, video_filename)
    
    # Render animation
    print(f"[INFO] Rendering animation to {video_filename}...")
    sentence_animator.render_animation(pose_data, save_path=video_path, fps=15)
    
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
            'avg_confidence': float(np.mean([r.get('confidence', 0) for r in successful])) if successful else 0.0
        },
        'truncated': truncated,
        'note': 'This translation uses semantic sentence matching from 30K+ How2Sign dataset'
    })

# --- Serve generated videos ---
@app.route('/output/<path:filename>')
def download_file(filename):
    return send_from_directory(output_dir, filename)

@app.route('/')
def home():
    return "SMPLX ASL Backend is running. Use the /asl_from_youtube endpoint."

if __name__ == '__main__':
    app.run(port=5000, debug=False, use_reloader=False)