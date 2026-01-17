import streamlit as st
import requests
import json
import os

st.set_page_config(page_title="YouTube ASL Sentence Translator", layout="wide")
st.title("🤟 YouTube to ASL - Sentence-Level Translation")
st.markdown("*Powered by 30,997 How2Sign sentences with semantic matching*")

# --- Configuration ---
FLASK_API_URL = "http://localhost:5000"
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
max_sentences = st.sidebar.slider("Max sentences to translate", 1, 20, 5)
top_k = st.sidebar.slider("Number of matches to consider", 1, 10, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 💡 How it works
1. Extracts transcript from YouTube
2. Finds semantically similar sentences
3. Uses phrase chunking as fallback
4. Renders 3D animation with SMPL-X

### 📊 Confidence Levels
- 🟢 **High** ≥ 0.85 - Excellent match
- 🟡 **Medium** 0.70-0.85 - Good match  
- 🔴 **Low** < 0.70 - Fallback used

**Note:** First translation takes 2-5 minutes to build the semantic index.
""")

# Main interface
st.markdown("### Enter YouTube URL")
youtube_url = st.text_input(
    "YouTube URL:",
    placeholder="https://www.youtube.com/watch?v=...",
    help="Paste any YouTube video URL with captions"
)

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    translate_btn = st.button("🎬 Translate to ASL", type="primary", use_container_width=True)
with col2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.rerun()

st.markdown("---")

# Translation logic
if translate_btn and youtube_url:
    with st.spinner("🔄 Translating... (First time: 2-5 min to build index, then fast!)"):
        try:
            # Call Flask API
            response = requests.post(
                f"{FLASK_API_URL}/asl_from_youtube_sentences",
                json={
                    "url": youtube_url,
                    "max_sentences": max_sentences,
                    "top_k": top_k
                },
                timeout=600  # 10 minutes for first request
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Display success
                st.success("✅ Translation complete!")
                
                # Display video - construct absolute path
                video_url = result.get('url', '')
                if video_url:
                    # Extract filename from URL like "/output/video.mp4"
                    video_filename = os.path.basename(video_url)
                    # Build absolute path
                    video_path = os.path.join(output_dir, video_filename)
                    
                    if os.path.exists(video_path):
                        st.markdown("### 🎥 Generated ASL Animation")
                        st.video(video_path)
                        
                        # Download button
                        with open(video_path, 'rb') as f:
                            st.download_button(
                                label="📥 Download Video",
                                data=f,
                                file_name=video_filename,
                                mime="video/mp4"
                            )
                    else:
                        st.warning(f"⚠️ Video file not found at: {video_path}")
                
                # Display statistics
                stats = result.get('statistics', {})
                st.markdown("### 📊 Translation Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Sentences", stats.get('sentences_processed', 0))
                with col2:
                    st.metric("Coverage", f"{stats.get('coverage_percentage', 0):.1f}%")
                with col3:
                    st.metric("Total Frames", stats.get('total_frames', 0))
                with col4:
                    st.metric("Duration", f"{stats.get('video_duration_seconds', 0):.1f}s")
                
                # Confidence breakdown
                if 'confidence_breakdown' in stats:
                    st.markdown("#### Confidence Breakdown")
                    conf = stats['confidence_breakdown']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🟢 High", conf.get('high', 0), 
                                 help="Excellent semantic matches (≥0.85)")
                    with col2:
                        st.metric("🟡 Medium", conf.get('medium', 0),
                                 help="Good matches (0.70-0.85)")
                    with col3:
                        st.metric("🔴 Low", conf.get('low', 0),
                                 help="Phrase chunking used (<0.70)")
                
                # Strategy breakdown
                if 'strategy_breakdown' in stats:
                    st.markdown("#### Translation Strategies Used")
                    strat = stats['strategy_breakdown']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Full Match", strat.get('full', 0),
                                 help="Entire sentence matched")
                    with col2:
                        st.metric("Chunked", strat.get('chunked', 0),
                                 help="Split into phrases")
                    with col3:
                        st.metric("Fallback", strat.get('fallback', 0),
                                 help="Best-effort match")
                
                # Detailed sentence breakdown
                st.markdown("### 📝 Sentence-by-Sentence Details")
                sentences = result.get('sentences', [])
                
                for i, sent in enumerate(sentences, 1):
                    with st.expander(f"Sentence {i}: {sent['original'][:100]}..."):
                        cols = st.columns([2, 1, 1, 1])
                        
                        with cols[0]:
                            st.markdown(f"**Original:** {sent['original']}")
                            st.markdown(f"**Matched:** {sent['match'][:100]}...")
                        
                        with cols[1]:
                            confidence = sent.get('confidence', 0)
                            if confidence >= 0.85:
                                st.markdown(f"🟢 **{confidence:.2f}**")
                            elif confidence >= 0.70:
                                st.markdown(f"🟡 **{confidence:.2f}**")
                            else:
                                st.markdown(f"🔴 **{confidence:.2f}**")
                        
                        with cols[2]:
                            st.markdown(f"**Strategy:** {sent.get('strategy', 'N/A')}")
                        
                        with cols[3]:
                            st.markdown(f"**Frames:** {sent.get('frames', 0)}")
                        
                        # Show alternatives if available
                        if 'alternatives' in sent and sent['alternatives']:
                            st.markdown("**Alternative matches:**")
                            for alt in sent['alternatives'][:3]:
                                st.markdown(f"- {alt['text'][:80]}... (confidence: {alt['confidence']:.2f})")
            
            else:
                error_msg = response.json().get('error', 'Unknown error')
                st.error(f"❌ Translation failed: {error_msg}")
                
                # Show debug info
                with st.expander("🔍 Debug Information"):
                    st.json(response.json())
        
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The first request can take 2-5 minutes to build the index. Please try again.")
        
        except requests.exceptions.ConnectionError:
            st.error("🔌 Cannot connect to Flask server. Make sure it's running with: `python app.py`")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            with st.expander("🔍 Debug Information"):
                st.exception(e)

elif translate_btn and not youtube_url:
    st.warning("⚠️ Please enter a YouTube URL")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>
        Using sentence-transformers (all-MiniLM-L6-v2) + FAISS for semantic matching<br>
        Dataset: 30,997 How2Sign sentences | Model: SMPL-X 3D body model
    </small>
</div>
""", unsafe_allow_html=True)
