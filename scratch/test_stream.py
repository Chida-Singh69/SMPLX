import os
import sys
sys.path.append('backend/api')
import app
import json

app.how2sign_dataset_dir = "data/raw_poses/how2sign_pkls_cropTrue_shapeFalse"
app.output_dir = "data/output"

with app.app.test_request_context('/api/stream_youtube_chunks', json={'sentences': ['hello world'], 'gender': 'neutral'}):
    response = app.app.test_client().post('/api/stream_youtube_chunks', json={'sentences': ['hello world'], 'gender': 'neutral'})
    # Iterate through the generator
    for chunk in response.iter_encoded():
        print(chunk)
