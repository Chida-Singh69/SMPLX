import requests
import json

try:
    response = requests.post('http://127.0.0.1:5000/extract_transcript', json={'url': 'https://www.youtube.com/watch?v=kJQP7kiw5Fk'})
    print("Status:", response.status_code)
    try:
        print("Data:", response.json())
    except:
        print("Raw:", response.text)
except Exception as e:
    print("Error:", e)
