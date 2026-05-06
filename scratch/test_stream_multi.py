import requests
import json
import sys

response = requests.post('http://127.0.0.1:5000/api/stream_youtube_chunks', 
                        json={'sentences': ['hello there', 'how are you doing today', 'i am fine thank you'], 'gender': 'neutral'},
                        stream=True)

print("Status:", response.status_code)
try:
    for line in response.iter_lines():
        if line:
            print("Received:", line.decode('utf-8'))
except Exception as e:
    print("Error:", e)
