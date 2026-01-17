"""
Test script for sentence-level ASL translation.
Run this after installing dependencies to verify the system works.
"""
import requests
import json
import time

# Configuration
API_BASE = "http://localhost:5000"
TEST_VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with actual video

def test_sentence_translation():
    """Test the sentence-based translation endpoint."""
    
    print("=" * 80)
    print("Testing Sentence-Level ASL Translation")
    print("=" * 80)
    
    # Test 1: Extract transcript first (quick preview)
    print("\n[Test 1] Extracting transcript...")
    try:
        response = requests.post(
            f"{API_BASE}/extract_transcript",
            json={"url": TEST_VIDEO_URL},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Transcript extracted: {len(data['transcript'])} characters")
            print(f"  Total words: {data['total_words']}")
            print(f"  Available words (word-level): {data['available_count']}")
            print(f"  Preview: {data['transcript'][:200]}...")
        else:
            print(f"✗ Error: {response.json()}")
            return
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        return
    
    # Test 2: Sentence-level translation
    print("\n[Test 2] Generating sentence-level ASL animation...")
    print("Note: First request may take 2-5 minutes to build index")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{API_BASE}/asl_from_youtube_sentences",
            json={
                "url": TEST_VIDEO_URL,
                "max_sentences": 3  # Limit for testing
            },
            timeout=600  # 10 minutes max (for first-time index build)
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ Translation successful! ({elapsed:.1f}s)")
            print(f"  Video URL: {data['url']}")
            print(f"  Sentences processed: {data['sentences_processed']}")
            print(f"  Sentences successful: {data['sentences_successful']}")
            print(f"  Total frames: {data['total_frames']}")
            
            stats = data['statistics']
            print(f"\n  Confidence breakdown:")
            print(f"    High (≥0.85):   {stats['high_confidence']} sentences")
            print(f"    Medium (0.70-0.85): {stats['medium_confidence']} sentences")
            print(f"    Low (<0.70):    {stats['low_confidence']} sentences")
            print(f"    Average:        {stats['avg_confidence']:.3f}")
            
            # Show translation details
            print(f"\n  Translation details:")
            for i, result in enumerate(data['translation_results'][:5], 1):
                sentence = result.get('input_sentence', '')[:60]
                confidence = result.get('confidence', 0)
                strategy = result.get('strategy', 'unknown')
                print(f"    {i}. [{strategy}] {sentence}... (conf: {confidence:.3f})")
            
            print(f"\n✓ Video saved to: output/{data['url'].split('/')[-1]}")
            
        else:
            print(f"✗ Error {response.status_code}: {response.json()}")
            
    except requests.exceptions.Timeout:
        print("✗ Request timed out (index building may take longer on first run)")
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
    
    print("\n" + "=" * 80)

def test_sentence_matcher_directly():
    """Test the sentence matcher independently (without API)."""
    
    print("\n[Direct Test] Testing SentenceMatcher...")
    
    try:
        from sentence_matcher import SentenceMatcher
        
        # Initialize
        matcher = SentenceMatcher(
            mapping_file="how2sign_mapping.json",
            dataset_dir="how2sign_pkls_cropTrue_shapeFalse"
        )
        
        # Test sentences
        test_sentences = [
            "Hello, how are you?",
            "I am going to the store today",
            "The weather is nice outside",
            "This is a complex sentence with many words that might not match"
        ]
        
        print(f"\nTesting {len(test_sentences)} sentences...")
        
        for sentence in test_sentences:
            print(f"\nInput: {sentence}")
            result = matcher.translate_sentence(sentence, verbose=False)
            
            print(f"  Strategy: {result['strategy']}")
            print(f"  Confidence: {result.get('confidence', 0):.3f}")
            if result.get('warning'):
                print(f"  Warning: {result['warning']}")
            
            if result['strategy'] == 'full':
                match = result['matches'][0]
                print(f"  Match: {match['sentence'][:60]}...")
            elif result['strategy'] == 'chunked':
                print(f"  Chunks matched: {len(result['matches'])}")
                for cm in result['matches'][:3]:
                    print(f"    - {cm['input_chunk']} -> {cm['match']['sentence'][:40]}...")
        
        print("\n✓ Direct test completed")
        
    except ImportError as e:
        print(f"✗ Import error: {str(e)}")
        print("  Make sure dependencies are installed:")
        print("    pip install sentence-transformers faiss-cpu spacy")
        print("    python -m spacy download en_core_web_sm")
    except Exception as e:
        print(f"✗ Error: {str(e)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--direct":
        # Test sentence matcher directly
        test_sentence_matcher_directly()
    else:
        # Test via API (requires Flask server running)
        print("\nMake sure Flask server is running:")
        print("  python app.py\n")
        
        try:
            # Check if server is running
            response = requests.get(f"{API_BASE}/", timeout=5)
            test_sentence_translation()
        except requests.exceptions.ConnectionError:
            print("✗ Cannot connect to Flask server")
            print("\nStart the server first:")
            print("  python app.py")
            print("\nOr test the matcher directly:")
            print("  python test_sentence_translation.py --direct")
