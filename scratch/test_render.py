import os
import sys
sys.path.append('backend/api')
import app
import json

app.how2sign_dataset_dir = "data/raw_poses/how2sign_pkls_cropTrue_shapeFalse"
app.output_dir = "data/output"

with app.app.test_request_context('/api/render_sentence', json={'pkl': 'how2sign_0000.pkl', 'gender': 'neutral'}):
    # we need to find a valid pkl file first
    mapping_path = "data/metadata/how2sign_mapping.json"
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    first_pkl = list(mapping.keys())[0]
    
    response = app.app.test_client().post('/api/render_sentence', json={'pkl': first_pkl, 'gender': 'neutral'})
    print(response.get_data(as_text=True))
