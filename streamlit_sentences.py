import streamlit as st
import os
import json

st.set_page_config(page_title="How2Sign Sentence Viewer", layout="wide")
st.title("🤟 How2Sign Dataset - Sentence-Level ASL Viewer")

# --- Configuration and Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
mapping_path = os.path.join(current_dir, "how2sign_mapping.json")
dataset_dir = os.path.join(current_dir, "how2sign_pkls_cropTrue_shapeFalse")

# Load mapping
@st.cache_data
def load_mapping():
    with open(mapping_path, "r", encoding='utf-8') as f:
        return json.load(f)

gloss_map = load_mapping()

# Create sentence to pkl mapping
sentence_to_pkl = {}
for pkl_file, sentence in gloss_map.items():
    # Truncate long sentences for display
    display_text = sentence[:100] + "..." if len(sentence) > 100 else sentence
    sentence_to_pkl[display_text] = {
        "pkl": pkl_file,
        "full_text": sentence
    }

all_sentences = sorted(sentence_to_pkl.keys())

st.markdown(f"""
### Dataset Statistics
- **Total Sentences**: {len(all_sentences):,}
- **Pickle Files**: {len(gloss_map):,}
- **Dataset**: How2Sign (sentence-level continuous signing)
""")

st.markdown("---")

# Search functionality
search_term = st.text_input("🔍 Search sentences:", placeholder="Type to search...")

if search_term:
    filtered_sentences = [s for s in all_sentences if search_term.lower() in s.lower()]
    st.write(f"Found {len(filtered_sentences)} matching sentences")
else:
    filtered_sentences = all_sentences

# Display mode selection
display_mode = st.radio("Display Mode:", ["List View", "Table View"], horizontal=True)

if display_mode == "List View":
    # Pagination
    items_per_page = 20
    total_pages = (len(filtered_sentences) + items_per_page - 1) // items_per_page
    
    page = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1)
    
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(filtered_sentences))
    
    st.write(f"Showing {start_idx + 1}-{end_idx} of {len(filtered_sentences)} sentences")
    
    for sentence in filtered_sentences[start_idx:end_idx]:
        info = sentence_to_pkl[sentence]
        with st.expander(f"📝 {sentence}"):
            st.markdown(f"**Full Text**: {info['full_text']}")
            st.markdown(f"**Pickle File**: `{info['pkl']}`")
            st.code(f"Dataset Path: {os.path.join(dataset_dir, info['pkl'])}", language="text")
            
            # Check if file exists
            full_path = os.path.join(dataset_dir, info['pkl'])
            if os.path.exists(full_path):
                st.success("✅ Pickle file exists")
            else:
                st.error("❌ Pickle file not found")

else:  # Table View
    # Show limited table
    st.dataframe({
        "Sentence": [sentence_to_pkl[s]["full_text"][:80] + "..." for s in filtered_sentences[:100]],
        "Pickle File": [sentence_to_pkl[s]["pkl"] for s in filtered_sentences[:100]]
    }, use_container_width=True)
    
    if len(filtered_sentences) > 100:
        st.info(f"Showing first 100 of {len(filtered_sentences)} results. Use search to narrow down.")

st.markdown("---")
st.markdown("""
### Usage Notes:
- This dataset contains **full sentence-level** ASL signs from the How2Sign dataset
- Each entry represents a continuous signing sequence for a complete sentence
- The pickle files contain SMPL-X pose parameters for 3D visualization
- To visualize these signs, you would need to integrate with the rendering pipeline
""")
