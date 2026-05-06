"""Quick test for chunk_transcript_by_timestamps and blend_adjacent_chunks."""
import sys, numpy as np
sys.path.insert(0, '.')
from app import chunk_transcript_by_timestamps, blend_adjacent_chunks

# Simulate YouTube transcript entries
class FakeEntry:
    def __init__(self, start, duration, text):
        self.start = start
        self.duration = duration
        self.text = text

# --- Test 1: Timestamp chunking ---
transcript = [
    FakeEntry(0.0,  1.5, 'hello how are you'),
    FakeEntry(1.6,  1.2, 'doing today'),
    # gap of 1.2s (> 0.8) -> split here
    FakeEntry(4.0,  2.0, 'I want to show you'),
    FakeEntry(6.1,  1.0, 'something really cool'),
    # gap of 1.5s -> split here
    FakeEntry(8.6,  1.0, 'lets get started'),
]

chunks = chunk_transcript_by_timestamps(transcript, max_gap=0.8)
print('=== Chunking Test ===')
for i, c in enumerate(chunks):
    print(f'  Chunk {i}: "{c["text"]}" [{c["start_time"]:.1f}s - {c["end_time"]:.1f}s]')
print(f'  Total chunks: {len(chunks)}')
assert len(chunks) == 3, f'Expected 3 chunks, got {len(chunks)}'
print('  PASS\n')

# --- Test 2: Blend adjacent chunks ---
seq1 = np.ones((10, 182)) * 1.0
seq2 = np.ones((8, 182)) * 5.0
seq3 = np.ones((12, 182)) * 9.0

blended = blend_adjacent_chunks([seq1, seq2, seq3], blend_frames=4)
expected = 10 + 4 + 8 + 4 + 12
print('=== Blending Test ===')
print(f'  Input:  3 seqs of 10, 8, 12 frames')
print(f'  Output: {blended.shape[0]} frames (expected {expected})')
assert blended.shape == (expected, 182), f'Wrong shape: {blended.shape}'

# Check transition values
t1 = blended[10, 0]  # first transition frame
print(f'  Transition frame value: {t1:.2f} (should be between 1.0 and 5.0)')
assert 1.0 < t1 < 5.0
print('  PASS\n')

# --- Test 3: Edge cases ---
single = blend_adjacent_chunks([seq1])
assert single.shape == (10, 182)
print('=== Single sequence: PASS')

empty = blend_adjacent_chunks([])
assert empty.shape[0] == 0
print('=== Empty list: PASS')

print('\nALL TESTS PASSED')
