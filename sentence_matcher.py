"""
Sentence-level ASL matching using semantic similarity.
Uses sentence-transformers and FAISS for efficient semantic search across 30K+ sentences.
"""
import os
import json
import numpy as np
import pickle
from typing import List, Dict, Optional, Tuple

# Lazy imports - only load when needed
sentence_transformers = None
faiss = None
spacy = None


class SentenceMatcher:
    """
    Semantic sentence matching for How2Sign dataset.
    Lazy-loads dependencies and builds FAISS index on first use.
    """
    
    # Similarity thresholds
    HIGH_CONFIDENCE = 0.85   # Use match as-is
    MEDIUM_CONFIDENCE = 0.70  # Use with caution
    LOW_CONFIDENCE = 0.60    # Must chunk or skip
    
    def __init__(self, mapping_file: str, dataset_dir: str):
        """
        Initialize sentence matcher.
        
        Args:
            mapping_file: Path to how2sign_mapping.json (pkl_file -> sentence)
            dataset_dir: Directory containing .pkl files
        """
        self.mapping_file = mapping_file
        self.dataset_dir = dataset_dir
        
        # Lazy-loaded components
        self.sentence_to_file = {}  # sentence -> pkl_filename
        self.sentence_list = []     # ordered list of sentences
        self.embeddings = None      # sentence embeddings
        self.index = None           # FAISS index
        self.model = None           # SentenceTransformer model
        self.nlp = None             # spaCy model for chunking
        
        self._initialized = False
        self._dependencies_loaded = False
        
        print(f"[SentenceMatcher] Initialized (lazy-loading enabled)")
    
    def _load_dependencies(self):
        """Lazy-load heavy dependencies only when needed."""
        if self._dependencies_loaded:
            return
        
        global sentence_transformers, faiss, spacy
        
        print("[SentenceMatcher] Loading dependencies...")
        
        try:
            import sentence_transformers as st
            sentence_transformers = st
            print("  ✓ sentence-transformers loaded")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        try:
            import faiss as f
            faiss = f
            print("  ✓ faiss loaded")
        except ImportError:
            raise ImportError(
                "faiss not installed. "
                "Install with: pip install faiss-cpu (or faiss-gpu for GPU support)"
            )
        
        # Try to load spacy, but make it optional
        try:
            import spacy as sp
            spacy = sp
            print("  ✓ spacy loaded")
        except Exception as e:
            print(f"  ⚠ spacy unavailable (using simple chunking fallback): {e}")
            spacy = None
        
        self._dependencies_loaded = True
    
    def _load_mapping(self):
        """Load sentence mappings from JSON."""
        print(f"[SentenceMatcher] Loading mappings from {self.mapping_file}...")
        
        with open(self.mapping_file, 'r') as f:
            pkl_to_sentence = json.load(f)
        
        # Create reverse mapping: sentence -> pkl_file
        # Also filter to only sentences with existing .pkl files
        for pkl_file, sentence in pkl_to_sentence.items():
            full_path = os.path.join(self.dataset_dir, pkl_file)
            if os.path.exists(full_path):
                # Normalize sentence (lowercase, strip)
                normalized = sentence.strip().lower()
                self.sentence_to_file[normalized] = pkl_file
        
        self.sentence_list = list(self.sentence_to_file.keys())
        print(f"  ✓ Loaded {len(self.sentence_list)} sentences with available .pkl files")
    
    def _build_index(self):
        """Build FAISS index for semantic search."""
        if self.index is not None:
            return  # Already built
        
        self._load_dependencies()
        
        print("[SentenceMatcher] Building semantic search index...")
        
        # Load sentence-to-file mapping
        self._load_mapping()
        
        # Initialize sentence transformer model
        print("  Loading sentence transformer model...")
        self.model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
        
        # Encode all sentences
        print(f"  Encoding {len(self.sentence_list)} sentences...")
        self.embeddings = self.model.encode(
            self.sentence_list,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=128
        )
        
        # Build FAISS index
        print("  Building FAISS index...")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        print(f"  ✓ Index built with {self.index.ntotal} sentences")
        self._initialized = True
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Find top-k most similar sentences using semantic search.
        
        Args:
            query: Input sentence/phrase to search for
            top_k: Number of results to return
            
        Returns:
            List of dicts with keys: sentence, similarity, pkl_path
        """
        # Build index on first use
        if not self._initialized:
            self._build_index()
        
        # Encode query
        query_embedding = self.model.encode([query.lower().strip()])
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 if not enough results
                continue
            sentence = self.sentence_list[idx]
            pkl_path = os.path.join(self.dataset_dir, self.sentence_to_file[sentence])
            results.append({
                'sentence': sentence,
                'similarity': float(dist),
                'pkl_path': pkl_path,
                'pkl_file': self.sentence_to_file[sentence]
            })
        
        return results
    
    def _load_spacy_model(self):
        """Lazy-load spaCy model for chunking."""
        if self.nlp is not None:
            return
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("[WARNING] spaCy model 'en_core_web_sm' not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            self.nlp = spacy.load("en_core_web_sm")
    
    def _chunk_sentence(self, sentence: str) -> List[str]:
        """
        Break sentence into meaningful chunks (phrases).
        Uses spaCy for noun phrases and clause detection.
        if available, otherwise simple fallback.
        
        Args:
            sentence: Input sentence
            
        Returns:
            List of sentence chunks/phrases
        """
        # Try spaCy if available
        if spacy is not None and self.nlp is not None:
            try:
                doc = self.nlp(sentence)
                chunks = []
                
                # Extract noun phrases
                for chunk in doc.noun_chunks:
                    chunks.append(chunk.text.strip())
                
                # If no noun phrases found, split by punctuation/clauses
                if not chunks:
                    chunks = [sent.text.strip() for sent in doc.sents]
                
                if chunks:
                    return [c for c in chunks if c]
            except Exception as e:
                print(f"  ⚠ spaCy chunking failed: {e}, using simple fallback")
        
        # Simple fallback: split by common punctuation and conjunctions
        import re
        # Split on periods, commas, semicolons, and common conjunctions
        chunks = re.split(r'[.,;]|\s+(?:and|but|or|so|because|while|when|if|although)\s+', sentence, flags=re.IGNORECASE)
        chunks = [c.strip() for c in chunks if c.strip()]
        
        # If still nothing, split into smaller groups of words
        if not chunks or len(chunks) == 1:
            words = sentence.split()
            chunk_size = max(2, len(words) // 3)  # Split into ~3 chunks
            chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        return [c for c in chunks if c]  # Remove empty strings
    
    def translate_sentence(self, input_sentence: str, verbose: bool = True) -> Dict:
        """
        Translate a sentence to ASL animation using semantic matching.
        
        Strategy:
        1. Try full sentence match (high confidence)
        2. If low confidence, chunk into phrases and match each
        3. Return best match(es) with metadata
        
        Args:
            input_sentence: Input sentence in English
            verbose: Print matching progress
            
        Returns:
            Dictionary with:
                - strategy: 'full' or 'chunked'
                - matches: List of match dictionaries
                - confidence: Overall confidence score
                - warning: Optional warning message
        """
        # Build index on first use
        if not self._initialized:
            self._build_index()
        
        input_normalized = input_sentence.strip().lower()
        
        # Try full sentence match
        matches = self.search(input_normalized, top_k=3)
        best_match = matches[0] if matches else None
        
        if not best_match:
            return {
                'strategy': 'failed',
                'matches': [],
                'confidence': 0.0,
                'warning': 'No matches found in dataset'
            }
        
        best_similarity = best_match['similarity']
        
        # High confidence - use directly
        if best_similarity >= self.HIGH_CONFIDENCE:
            if verbose:
                print(f"✓ High confidence match: '{best_match['sentence']}' (sim: {best_similarity:.3f})")
            return {
                'strategy': 'full',
                'matches': [best_match],
                'confidence': best_similarity,
                'input_sentence': input_sentence,
                'warning': None
            }
        
        # Medium confidence - use with warning
        elif best_similarity >= self.MEDIUM_CONFIDENCE:
            if verbose:
                print(f"⚠ Medium confidence match: '{best_match['sentence']}' (sim: {best_similarity:.3f})")
            return {
                'strategy': 'full',
                'matches': [best_match],
                'confidence': best_similarity,
                'input_sentence': input_sentence,
                'warning': 'Medium confidence - translation may not be exact'
            }
        
        # Low confidence - try chunking
        else:
            if verbose:
                print(f"✗ Low confidence ({best_similarity:.3f}) - trying phrase chunking...")
            
            chunks = self._chunk_sentence(input_sentence)
            if verbose:
                print(f"  Chunks: {chunks}")
            
            chunk_matches = []
            for chunk in chunks:
                chunk_results = self.search(chunk, top_k=1)
                if chunk_results and chunk_results[0]['similarity'] >= self.LOW_CONFIDENCE:
                    if verbose:
                        print(f"    ✓ '{chunk}' -> '{chunk_results[0]['sentence']}' (sim: {chunk_results[0]['similarity']:.3f})")
                    chunk_matches.append({
                        'input_chunk': chunk,
                        'match': chunk_results[0]
                    })
                else:
                    if verbose:
                        print(f"    ✗ '{chunk}' - no good match")
            
            if chunk_matches:
                avg_confidence = np.mean([cm['match']['similarity'] for cm in chunk_matches])
                return {
                    'strategy': 'chunked',
                    'matches': chunk_matches,
                    'confidence': float(avg_confidence),
                    'input_sentence': input_sentence,
                    'warning': f'Sentence broken into {len(chunk_matches)} phrases for matching'
                }
            else:
                # No chunks matched - return best overall match with strong warning
                if verbose:
                    print(f"  ⚠ No chunks matched well - using best overall match (low quality)")
                return {
                    'strategy': 'fallback',
                    'matches': [best_match],
                    'confidence': best_similarity,
                    'input_sentence': input_sentence,
                    'warning': f'Low quality match (confidence: {best_similarity:.2f}) - translation may be inaccurate'
                }
