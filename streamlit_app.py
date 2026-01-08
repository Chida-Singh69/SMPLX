import streamlit as st
import os
import json
import requests

st.set_page_config(page_title="SMPL-X Animation Demo", layout="centered")
st.title("🤟 ASL Overlay - Sign Language Animation")

# --- Configuration and Setup ---
FLASK_API_URL = "http://localhost:5000"  # Flask backend URL

current_dir = os.path.dirname(os.path.abspath(__file__))
mapping_path = os.path.join(current_dir, "filtered_video_to_gloss.json")
dataset_dir = os.path.join(current_dir, "word-level-dataset-cpu-fixed")
output_dir = os.path.join(current_dir, "output")
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

# Load sentence-level dataset (How2Sign)
sentence_mapping_path = os.path.join(current_dir, "how2sign_mapping.json")
sentence_dataset_dir = os.path.join(current_dir, "how2sign_pkls_cropTrue_shapeFalse")

sentence_to_pkl = {}
if os.path.exists(sentence_mapping_path):
    with open(sentence_mapping_path, "r", encoding='utf-8') as f:
        sentence_gloss_map = json.load(f)
    
    # Create searchable mapping
    for pkl_file, sentence in sentence_gloss_map.items():
        full_path = os.path.join(sentence_dataset_dir, pkl_file)
        if os.path.exists(full_path):
            # Truncate long sentences for display
            display_text = sentence[:100] + "..." if len(sentence) > 100 else sentence
            sentence_to_pkl[display_text] = {
                "pkl": pkl_file,
                "full_text": sentence
            }

# --- Tabs for Different Modes ---
tab1, tab2, tab3 = st.tabs(["📺 YouTube Video", "📝 Word Selection", "📖 Sentence Animations"])

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
    
    # Step 1: Extract transcript
    if st.button("📥 Extract Transcript", key="extract_btn", type="secondary"):
        if not youtube_url:
            st.warning("Please enter a YouTube URL.")
        elif "youtube.com" not in youtube_url and "youtu.be" not in youtube_url:
            st.error("Please enter a valid YouTube URL.")
        else:
            with st.spinner("📥 Extracting transcript from YouTube..."):
                try:
                    response = requests.post(
                        f"{FLASK_API_URL}/extract_transcript",
                        json={"url": youtube_url},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Store in session state
                        st.session_state['transcript_data'] = result
                        st.session_state['youtube_url'] = youtube_url
                        st.session_state['transcript_extracted'] = True
                        st.success("✅ Transcript extracted successfully!")
                        st.rerun()
                    else:
                        error_msg = response.json().get("error", "Unknown error")
                        st.error(f"❌ {error_msg}")
                        
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out.")
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Cannot connect to Flask backend on port 5000.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Display extracted transcript if available
    if 'transcript_data' in st.session_state:
        transcript_data = st.session_state['transcript_data']
        
        st.markdown("---")
        st.markdown("### 📝 Extracted Transcript")
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Words", transcript_data['total_words'])
        with col2:
            st.metric("✅ Available", transcript_data['available_count'])
        with col3:
            st.metric("❌ Missing", transcript_data['missing_count'])
        
        # Show full transcript in expandable section
        with st.expander("📄 Full Transcript Text", expanded=False):
            st.text_area("Transcript", transcript_data['transcript'], height=150, disabled=True)
        
        # Show word-by-word breakdown
        with st.expander("🔍 Word-by-Word Analysis", expanded=True):
            st.markdown("**Legend:** 🟢 Available in dataset | 🔴 Not in dataset")
            
            word_mapping = transcript_data['word_mapping']
            html_words = []
            for word_info in word_mapping:
                if word_info['status'] == 'available':
                    html_words.append(
                        f'<span style="background-color: #d4edda; color: #155724; padding: 2px 6px; '
                        f'margin: 2px; border-radius: 3px; display: inline-block;">🟢 {word_info["original"]}</span>'
                    )
                else:
                    html_words.append(
                        f'<span style="background-color: #f8d7da; color: #721c24; padding: 2px 6px; '
                        f'margin: 2px; border-radius: 3px; display: inline-block;">🔴 {word_info["original"]}</span>'
                    )
            
            st.markdown(' '.join(html_words), unsafe_allow_html=True)
        
        # Show available words list
        if transcript_data['unique_available']:
            st.info(f"**Words to be animated:** {', '.join(transcript_data['unique_available'][:20])}" + 
                   ('...' if len(transcript_data['unique_available']) > 20 else ''))
        
        # Step 2: Generate video
        st.markdown("---")
        if st.button("🎬 Generate ASL Animation", key="youtube_btn", type="primary"):
            with st.spinner("🔄 Generating ASL animation... This may take 30-60 seconds."):
                try:
                    # Call Flask API endpoint
                    response = requests.post(
                        f"{FLASK_API_URL}/asl_from_youtube",
                        json={"url": st.session_state['youtube_url']},
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        video_url = result.get("url")
                        words_found = result.get("words", [])
                        
                        if video_url:
                            # Extract filename from URL
                            video_filename = video_url.split("/")[-1]
                            video_path = os.path.join(output_dir, video_filename)
                            
                            if os.path.exists(video_path):
                                st.success(f"✅ ASL video generated successfully!")
                                
                                st.markdown("### 🎥 Generated ASL Animation")
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
                    st.error("⏱️ Request timed out. The video might be too long.")
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Cannot connect to Flask backend. Make sure Flask is running on port 5000.")
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
                    # Call Flask API to generate video
                    response = requests.post(
                        f"{FLASK_API_URL}/asl_stream",
                        json={"words": selected_words},
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        st.success(f"✅ Animation generated for: {', '.join(selected_words)}")
                        
                        # Display video inline
                        st.video(response.content)
                        
                        # Download button
                        video_filename = f"{'_'.join(selected_words[:3])}_animation.mp4"
                        st.download_button(
                            label="⬇️ Download Video",
                            data=response.content,
                            file_name=video_filename,
                            mime="video/mp4"
                        )
                    else:
                        error_data = response.json()
                        st.error(f"❌ Error: {error_data.get('error', 'Unknown error')}")
                        
                except requests.exceptions.Timeout:
                    st.error("❌ Request timed out. The server may be busy.")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Could not connect to Flask API. Make sure it's running on port 5000.")
                except Exception as e:
                    st.error(f"❌ Error generating animation: {str(e)}")
    
    st.markdown("---")
    st.markdown("""
    **Instructions:**
    1. Select one or more words from the list
    2. Click '✨ Generate Animation'
    3. Watch the animation. If multiple words are selected, they will be concatenated
    4. Download the generated video if needed
    """)

# ============================================
# TAB 3: SENTENCE ANIMATIONS (How2Sign)
# ============================================
with tab3:
    st.markdown("### Sentence-Level ASL Animations")
    st.markdown("Browse and render full sentence signs from the How2Sign dataset.")
    
    if not sentence_to_pkl:
        st.error("❌ How2Sign dataset not found. Please ensure `how2sign_mapping.json` and pickle files are available.")
    else:
        st.success(f"✅ Loaded {len(sentence_to_pkl):,} sentences from How2Sign dataset")
        
        # Search functionality
        search_term = st.text_input("🔍 Search sentences:", placeholder="Type to search sentences...")
        
        if search_term:
            filtered_sentences = [s for s in sentence_to_pkl.keys() if search_term.lower() in s.lower()]
            st.write(f"Found {len(filtered_sentences)} matching sentences")
        else:
            filtered_sentences = list(sentence_to_pkl.keys())[:100]  # Show first 100 by default
            st.info("💡 Showing first 100 sentences. Use search to find specific sentences.")
        
        # Sentence selection
        selected_sentence = st.selectbox(
            "Select a sentence to animate:",
            [""] + filtered_sentences,
            help="Choose a sentence to generate its ASL animation"
        )
        
        if selected_sentence:
            sentence_info = sentence_to_pkl[selected_sentence]
            
            # Display sentence details
            st.markdown("---")
            st.markdown("**Selected Sentence:**")
            st.info(sentence_info['full_text'])
            
            with st.expander("📄 Details"):
                st.markdown(f"**Pickle File:** `{sentence_info['pkl']}`")
                pkl_path = os.path.join(sentence_dataset_dir, sentence_info['pkl'])
                st.markdown(f"**File Path:** `{pkl_path}`")
                st.markdown(f"**File Exists:** {'✅ Yes' if os.path.exists(pkl_path) else '❌ No'}")
            
            # Render options
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                render_full = st.checkbox("Render full animation (may be slow for long sentences)", value=False)
            with col2:
                max_frames = None if render_full else 150  # Limit to ~10 seconds at 15 fps
            with col3:
                # Check if GPU is available
                import torch
                gpu_available = torch.cuda.is_available()
                if gpu_available:
                    use_gpu = st.checkbox("🚀 Use GPU", value=True, help="Use GPU for faster rendering")
                    device = 'cuda' if use_gpu else 'cpu'
                else:
                    st.info("💻 CPU only")
                    device = 'cpu'
            
            if st.button("🎬 Generate Sentence Animation", type="primary", key="sentence_btn"):
                pkl_path = os.path.join(sentence_dataset_dir, sentence_info['pkl'])
                
                if not os.path.exists(pkl_path):
                    st.error("❌ Pickle file not found!")
                else:
                    device_label = "GPU (CUDA)" if device == 'cuda' else "CPU"
                    with st.spinner(f"Generating animation on {device_label}... {'Full sequence' if render_full else 'Preview (first ~10 seconds)'}"):
                        try:
                            from sentence_to_smplx import SentenceToSMPLX
                            
                            # Initialize animator with device selection
                            animator = SentenceToSMPLX(
                                model_path=os.path.join(current_dir, "models"),
                                viewport_width=640,
                                viewport_height=480,
                                device=device
                            )
                            
                            # Load pose data
                            pose_data = animator.load_pose_sequence(pkl_path)
                            
                            # Render to video
                            output_filename = f"sentence_{sentence_info['pkl'].replace('.pkl', '.mp4')}"
                            output_path = os.path.join(output_dir, output_filename)
                            
                            animator.render_animation(
                                pose_data,
                                save_path=output_path,
                                fps=15,
                                max_frames=max_frames
                            )
                            
                            st.success("✅ Animation generated successfully!")
                            
                            # Display video
                            st.markdown("### 🎥 Generated Animation")
                            st.video(output_path)
                            
                            # Download button
                            with open(output_path, "rb") as file:
                                st.download_button(
                                    label="⬇️ Download Video",
                                    data=file,
                                    file_name=output_filename,
                                    mime="video/mp4"
                                )
                            
                        except Exception as e:
                            st.error(f"❌ Error generating animation: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
        
        st.markdown("---")
        st.markdown("""
        **About Sentence-Level Animations:**
        - Contains 30,997+ full sentence signs from How2Sign dataset
        - Each animation shows continuous signing for a complete sentence
        - Animations can be long (10-60+ seconds per sentence)
        - Use preview mode to render only the first ~10 seconds
        - Search to find specific sentences or topics
        """)
        
        # Show some example sentences
        with st.expander("📚 Example Sentences"):
            sample_sentences = list(sentence_to_pkl.items())[:10]
            for display, info in sample_sentences:
                st.markdown(f"- {info['full_text'][:120]}{'...' if len(info['full_text']) > 120 else ''}")