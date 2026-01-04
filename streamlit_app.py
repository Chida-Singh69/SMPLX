import streamlit as st
import os
import json
import subprocess
from word_to_smplx import WordToSMPLX
import imageio
import numpy as np
import tempfile
import requests

# --- Pose Blending Function ---
def blend_pose_sequences(seq_a, seq_b, n_blend=5):
    # seq_a, seq_b: [N, D] numpy arrays of SMPL-X parameters
    if n_blend == 0 or len(seq_a) < n_blend or len(seq_b) < n_blend:
        return np.vstack([seq_a, seq_b])
    
    blended_part = []
    for i in range(n_blend):
        alpha = (i + 1) / (n_blend + 1)  # Alpha from near 0 to near 1
        # Linear interpolation for all 156 parameters
        current_blend = (1 - alpha) * seq_a[-n_blend + i, :] + alpha * seq_b[i, :]
        blended_part.append(current_blend)
    
    if not blended_part: # Should not happen if n_blend > 0 and sequences are long enough
        return np.vstack([seq_a, seq_b])
        
    blended_part_np = np.array(blended_part)
    
    # Concatenate: part of A, blended part, part of B
    return np.vstack([seq_a[:-n_blend, :], blended_part_np, seq_b[n_blend:, :]])

st.set_page_config(page_title="SMPL-X Animation Demo", layout="centered")
st.title("🤟 ASL Overlay - Sign Language Animation")

# --- Configuration and Setup ---
FLASK_API_URL = "http://localhost:5000"  # Flask backend URL
@st.cache_resource # Cache the animator resource
def get_animator(model_base_path):
    return WordToSMPLX(model_path=model_base_path)

current_dir = os.path.dirname(os.path.abspath(__file__))
mapping_path = os.path.join(current_dir, "filtered_video_to_gloss.json")
dataset_dir = os.path.join(current_dir, "word-level-dataset-cpu-fixed")
output_dir = os.path.join(current_dir, "output")
models_base_dir = os.path.join(current_dir, "models") # Path to "models" directory
os.makedirs(output_dir, exist_ok=True)

with open(mapping_path, "r") as f:
    gloss_map = json.load(f)

# Filter to only words with existing pickle files
word_to_pkl = {}
for pkl_file, word in gloss_map.items():
    full_path = os.path.join(dataset_dir, pkl_file)
    if os.path.exists(full_path):
        word_to_pkl[word.lower()] = pkl_file

all_words = sorted(word_to_pkl.keys())

animator = get_animator(models_base_dir)

# --- Tabs for Different Modes ---
tab1, tab2 = st.tabs(["📺 YouTube Video", "📝 Word Selection"])

# ============================================
# TAB 1: YOUTUBE VIDEO TRANSLATION
# ============================================
with tab1:
    st.markdown("### Translate YouTube Video to ASL")
    st.markdown("Paste a YouTube video URL below. The system will extract the transcript and generate ASL animation.")
    
    youtube_url = st.text_input(
        "YouTube Video URL:",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Enter a valid YouTube video URL with available captions/transcript"
    )
    
    if st.button("🎬 Translate Video", key="youtube_btn", type="primary"):
        if not youtube_url:
            st.warning("Please enter a YouTube URL.")
        elif "youtube.com" not in youtube_url and "youtu.be" not in youtube_url:
            st.error("Please enter a valid YouTube URL.")
        else:
            with st.spinner("🔄 Extracting transcript and generating ASL animation... This may take 30-60 seconds."):
                try:
                    # Call Flask API endpoint
                    response = requests.post(
                        f"{FLASK_API_URL}/asl_from_youtube",
                        json={"url": youtube_url},
                        timeout=120
                    )
                    
                    # Debug: show raw response
                    if response.status_code != 200:
                        st.error(f"Flask returned status {response.status_code}")
                        st.code(response.text)
                    
                    if response.status_code == 200:
                        result = response.json()
                        video_url = result.get("url")  # Returns "/output/filename.mp4"
                        words_found = result.get("words", [])
                        total_recognized = result.get("total_recognized", len(words_found))
                        total_processed = result.get("total_processed", len(words_found))
                        skipped_words = result.get("skipped_words", [])
                        skipped_count = result.get("skipped_count", 0)
                        
                        if video_url:
                            # Extract filename from URL
                            video_filename = video_url.split("/")[-1]
                            video_path = os.path.join(output_dir, video_filename)
                            
                            if os.path.exists(video_path):
                                st.success(f"✅ ASL animation generated successfully!")
                                
                                # Show processing statistics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Recognized", total_recognized)
                                with col2:
                                    st.metric("Animated", total_processed)
                                with col3:
                                    if skipped_count > 0:
                                        st.metric("Skipped", skipped_count, delta=f"-{skipped_count}", delta_color="inverse")
                                    else:
                                        st.metric("Skipped", 0)
                                
                                if words_found:
                                    st.info(f"**Animated words:** {', '.join(words_found[:15])}{'...' if len(words_found) > 15 else ''}")
                                
                                if skipped_words:
                                    with st.expander(f"⚠️ {len(skipped_words)} words skipped (CUDA compatibility issues)"):
                                        st.caption(', '.join(skipped_words))
                                
                                st.video(video_path)
                                
                                # Download button
                                with open(video_path, "rb") as file:
                                    st.download_button(
                                        label="⬇️ Download ASL Video",
                                        data=file,
                                        file_name=video_filename,
                                        mime="video/mp4"
                                    )
                            else:
                                st.error("Video file was generated but not found on disk.")
                        else:
                            st.warning("No recognizable words found in the video transcript.")
                    else:
                        error_msg = response.json().get("error", "Unknown error occurred")
                        st.error(f"❌ Error: {error_msg}")
                        
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. The video might be too long or the server is busy.")
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Cannot connect to Flask backend. Make sure Flask is running on port 5000.")
                    st.info("Run this command: `.\.venv\Scripts\python app.py`")
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
    
    st.markdown("---")
    st.markdown("""
    **How it works:**
    1. Paste a YouTube video URL
    2. System extracts the video transcript (captions)
    3. Converts text to ASL glosses
    4. Generates 3D sign language animation
    5. Download or watch the ASL overlay video
    """)

# ============================================
# TAB 2: WORD SELECTION (Original functionality)
# ============================================
with tab2:
    st.markdown("### Select Word(s) for Animation")
    selected_words = st.multiselect(
        "Choose one or more words from the dataset:", 
        all_words,
        help="Animations will be played in the order of selection if multiple words are chosen."
    )

    if 'video_path_to_display' not in st.session_state:
        st.session_state.video_path_to_display = None
    if 'video_header' not in st.session_state:
        st.session_state.video_header = ""

    if st.button("✨ Generate Animation", type="primary", key="word_btn"):
        if not selected_words:
            st.warning("Please select at least one word to animate.")
            st.session_state.video_path_to_display = None
        else:
            st.session_state.video_path_to_display = None
            with st.spinner(f"Generating animation for {', '.join(selected_words)}... This might take a moment."):
                pose_data_sequences = []
                for word in selected_words:
                    pkl_file = os.path.join(dataset_dir, word_to_pkl[word])
                    try:
                        pose_data_dict = animator.load_pose_sequence(pkl_file)
                        smplx_params_np = np.stack(pose_data_dict['smplx'])
                        pose_data_sequences.append(smplx_params_np)
                    except Exception as e:
                        st.error(f"Error loading pose data for '{word}': {e}")
                        pose_data_sequences = []
                        break
                if pose_data_sequences:
                    if len(selected_words) == 1:
                        video_filename = f"{selected_words[0]}_animation.mp4"
                        video_path = os.path.join(output_dir, video_filename)
                        if not os.path.exists(video_path):
                            pose_data = animator.load_pose_sequence(os.path.join(dataset_dir, word_to_pkl[selected_words[0]]))
                            animator.render_animation(pose_data, save_path=video_path, fps=15)
                        st.session_state.video_path_to_display = video_path
                        st.session_state.video_header = f"Animation for: {selected_words[0]}"
                    else:
                        # Concatenate videos in memory using imageio and tempfile
                        all_frames = []
                        for word in selected_words:
                            video_filename = f"{word}_animation.mp4"
                            video_path = os.path.join(output_dir, video_filename)
                            if not os.path.exists(video_path):
                                pose_data = animator.load_pose_sequence(os.path.join(dataset_dir, word_to_pkl[word]))
                                animator.render_animation(pose_data, save_path=video_path, fps=15)
                            if os.path.exists(video_path):
                                reader = imageio.get_reader(video_path)
                                all_frames.extend([frame for frame in reader])
                                reader.close()
                            else:
                                st.error(f"Video for '{word}' could not be found or rendered.")
                        if all_frames:
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
                                imageio.mimsave(tmpfile.name, all_frames, fps=15)
                                st.session_state.video_path_to_display = tmpfile.name
                                st.session_state.video_header = f"Combined Animation: {', '.join(selected_words)}"
                        else:
                            st.session_state.video_path_to_display = None
                            st.session_state.video_header = ""

    # Video display logic
    video_path = st.session_state.video_path_to_display
    if video_path and os.path.exists(video_path):
        st.markdown(f"### {st.session_state.video_header}")
        st.video(video_path)
        with open(video_path, "rb") as file:
            st.download_button(
                label="Download Video",
                data=file,
                file_name=os.path.basename(video_path),
                mime="video/mp4"
            )
        if st.button("Clear Output"):
            st.session_state.video_path_to_display = None
            st.session_state.video_header = ""
            st.experimental_rerun()
    else:
        if video_path:  # Only show error if a path was set
            st.error(f"Video file not found or could not be opened: {video_path}")

    st.markdown("---")
    st.markdown("**Instructions:**\
    1. Select one or more words from the list.\
    2. Click '✨ Generate Animation'.\
    3. Watch the animation. If multiple words are selected, they will be concatenated and played as a single video.\
    4. You can download the generated video or clear the output.") 