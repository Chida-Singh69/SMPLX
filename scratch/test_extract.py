import os
import sys
sys.path.append('backend/api')
import app

with app.app.test_request_context('/extract_transcript', json={'url': 'https://www.youtube.com/watch?v=yE0z1Zg2d5c'}):
    response = app.app.test_client().post('/extract_transcript', json={'url': 'https://www.youtube.com/watch?v=yE0z1Zg2d5c'})
    print("Status:", response.status_code)
    try:
        print("Data:", response.get_json())
    except:
        print("Raw:", response.get_data(as_text=True))
