import os
import sys
import json
sys.path.append('backend/api')
import app

app.how2sign_dataset_dir = "data/raw_poses/how2sign_pkls_cropTrue_shapeFalse"
app.output_dir = "data/output"

with app.app.test_request_context('/asl_stream', json={'words': ['ALL'], 'gender': 'neutral'}):
    response = app.app.test_client().post('/asl_stream', json={'words': ['ALL'], 'gender': 'neutral'})
    print(response.status_code)
    try:
        print(response.get_json())
    except:
        print(response.get_data(as_text=True)[:500])
