import streamlit as st
import os
import json
import subprocess
from word_to_smplx import WordToSMPLX
import imageio
import numpy as np
import tempfile
import requests

st.set_page_config(page_title="SMPL-X Animation Demo", layout="centered")
st.title("🤟 ASL Overlay - Sign Language Animation")

# --- Configuration and Setup ---
FLASK_API_URL = "http://localhost:5000"  # Flask backend URL

@st.cache_resource(hash_funcs={WordToSMPLX: id})  # Cache with version tracking
def get_animator(model_base_path, version="v2_pyrender"):
    """Load animator with version tracking to force cache refresh when code changes."""
    return WordToSMPLX(model_path=model_base_path)

current_dir = os.path.dirname(os.path.abspath(__file__))
mapping_path = os.path.join(current_dir, "filtered_video_to_gloss.json")
dataset_dir = os.path.join(current_dir, "word-level-dataset-cpu-fixed")
output_dir = os.path.join(current_dir, "output")
models_base_dir = os.path.join(current_dir, "models")
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
animator = get_animator(models_base_dir, version="v2_pyrender")

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
                    
                    if response.status_code == 200:
                        result = response.json()
                        video_url = result.get("url")
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
# TAB 2: WORD SELECTION
# ============================================
with tab2:
    st.markdown("### Select Word(s) for Animation")
    selected_words = st.multiselect(
        "Choose one or more words from the dataset:", 
        all_words,
        help="Animations will be played in the order of selection if multiple words are chosen."
    )

    if st.button("✨ Generate Animation", type="primary", key="word_btn"):
        if not selected_words:
            st.warning("Please select at least one word to animate.")
        else:
            with st.spinner(f"Generating animation for {', '.join(selected_words)}... This might take a moment."):
                try:
                    video_paths = []
                    
                    # Generate individual videos
                    for word in selected_words:
                        video_filename = f"{word}_animation.mp4"
                        video_path = os.path.join(output_dir, video_filename)
                        
                        # Only render if video doesn't exist
                        if not os.path.exists(video_path):
                            pkl_file = os.path.join(dataset_dir, word_to_pkl[word])
                            pose_data = animator.load_pose_sequence(pkl_file)
                            result = animator.render_animation(pose_data, save_path=video_path, fps=15)
                            
                            if result is None:
                                st.warning(f"⚠️ OpenGL rendering unavailable for '{word}'. Pose data loaded successfully but video generation skipped.")
                                continue
                        
                        if os.path.exists(video_path):
                            video_paths.append(video_path)
                    
                    if not video_paths:
                        st.error("No videos could be generated. OpenGL rendering may be unavailable.")
                    elif len(video_paths) == 1:
                        # Single video - display directly
                        st.success(f"✅ Animation generated for: {selected_words[0]}")
                        st.video(video_paths[0])
                        
                        with open(video_paths[0], "rb") as file:
                            st.download_button(
                                label="⬇️ Download Video",
                                data=file,
                                file_name=os.path.basename(video_paths[0]),
                                mime="video/mp4"
                            )
                    else:
                        # Multiple videos - concatenate
                        combined_filename = f"combined_{'_'.join(selected_words[:3])}.mp4"
                        combined_path = os.path.join(output_dir, combined_filename)
                        
                        all_frames = []
                        for path in video_paths:
                            reader = imageio.get_reader(path)
                            all_frames.extend([frame for frame in reader])
                            reader.close()
                        
                        imageio.mimsave(combined_path, all_frames, fps=15)
                        
                        st.success(f"✅ Combined animation generated for: {', '.join(selected_words)}")
                        st.video(combined_path)
                        
                        with open(combined_path, "rb") as file:
                            st.download_button(
                                label="⬇️ Download Combined Video",
                                data=file,
                                file_name=combined_filename,
                                mime="video/mp4"
                            )
                            
                except Exception as e:
                    st.error(f"❌ Error generating animation: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    st.markdown("---")
    st.markdown("""
    **Instructions:**
    1. Select one or more words from the list
    2. Click '✨ Generate Animation'
    3. Watch the animation. If multiple words are selected, they will be concatenated
    4. Download the generated video if needed
    """)