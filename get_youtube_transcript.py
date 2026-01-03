from youtube_transcript_api import YouTubeTranscriptApi
import sys
import re

def extract_video_id(url):
    """
    Extracts the video ID from a YouTube URL.
    """
    # Handles various YouTube URL formats
    regex = (
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    )
    match = re.search(regex, url)
    if match:
        return match.group(1)
    elif len(url) == 11:
        return url  # Assume it's a video ID
    else:
        raise ValueError("Invalid YouTube URL or video ID.")

def get_transcript(video_id):
    """
    Fetches the transcript for the given video ID.
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_youtube_transcript.py <YouTube URL or Video ID>")
        sys.exit(1)
    url = sys.argv[1]
    try:
        video_id = extract_video_id(url)
        transcript = get_transcript(video_id)
        if transcript:
            for entry in transcript:
                print(f"{entry['start']:.2f}s: {entry['text']}")
        else:
            print("Transcript not available.")
    except Exception as e:
        print(f"Error: {e}")