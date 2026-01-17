#!/bin/bash
# Installation script for sentence-level ASL translation system

echo "=================================================="
echo "Installing Sentence-Level ASL Translation System"
echo "=================================================="

# Check Python version
echo -e "\n[1/5] Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Install pip dependencies
echo -e "\n[2/5] Installing Python packages..."
pip install -r requirements.txt

# Download spaCy model
echo -e "\n[3/5] Downloading spaCy language model..."
python -m spacy download en_core_web_sm

# Verify installations
echo -e "\n[4/5] Verifying installations..."

python -c "import sentence_transformers; print('✓ sentence-transformers installed')" 2>/dev/null || echo "✗ sentence-transformers failed"
python -c "import faiss; print('✓ faiss installed')" 2>/dev/null || echo "✗ faiss failed"
python -c "import spacy; print('✓ spacy installed')" 2>/dev/null || echo "✗ spacy failed"
python -c "import torch; print('✓ torch installed')" 2>/dev/null || echo "✗ torch failed"

# Check dataset
echo -e "\n[5/5] Checking dataset..."
if [ -d "how2sign_pkls_cropTrue_shapeFalse" ]; then
    pkl_count=$(ls how2sign_pkls_cropTrue_shapeFalse/*.pkl 2>/dev/null | wc -l)
    echo "✓ Found How2Sign dataset with $pkl_count files"
else
    echo "⚠ Warning: how2sign_pkls_cropTrue_shapeFalse directory not found"
fi

if [ -f "how2sign_mapping.json" ]; then
    echo "✓ Found how2sign_mapping.json"
else
    echo "⚠ Warning: how2sign_mapping.json not found"
fi

echo -e "\n=================================================="
echo "Installation Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Start the Flask server:"
echo "     python app.py"
echo ""
echo "2. Test the system:"
echo "     python test_sentence_translation.py"
echo ""
echo "3. Or test directly (no server needed):"
echo "     python test_sentence_translation.py --direct"
echo ""
echo "Note: The first API request will take 2-5 minutes to build"
echo "      the FAISS index. After that, requests are fast!"
echo ""
