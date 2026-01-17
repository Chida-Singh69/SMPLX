@echo off
REM Installation script for sentence-level ASL translation system (Windows)

echo ==================================================
echo Installing Sentence-Level ASL Translation System
echo ==================================================

REM Check Python version
echo.
echo [1/5] Checking Python version...
python --version

REM Install pip dependencies
echo.
echo [2/5] Installing Python packages...
pip install -r requirements.txt

REM Download spaCy model
echo.
echo [3/5] Downloading spaCy language model...
python -m spacy download en_core_web_sm

REM Verify installations
echo.
echo [4/5] Verifying installations...

python -c "import sentence_transformers; print('✓ sentence-transformers installed')" 2>nul || echo ✗ sentence-transformers failed
python -c "import faiss; print('✓ faiss installed')" 2>nul || echo ✗ faiss failed
python -c "import spacy; print('✓ spacy installed')" 2>nul || echo ✗ spacy failed
python -c "import torch; print('✓ torch installed')" 2>nul || echo ✗ torch failed

REM Check dataset
echo.
echo [5/5] Checking dataset...
if exist "how2sign_pkls_cropTrue_shapeFalse" (
    echo ✓ Found How2Sign dataset directory
) else (
    echo ⚠ Warning: how2sign_pkls_cropTrue_shapeFalse directory not found
)

if exist "how2sign_mapping.json" (
    echo ✓ Found how2sign_mapping.json
) else (
    echo ⚠ Warning: how2sign_mapping.json not found
)

echo.
echo ==================================================
echo Installation Complete!
echo ==================================================
echo.
echo Next steps:
echo 1. Start the Flask server:
echo      python app.py
echo.
echo 2. Test the system:
echo      python test_sentence_translation.py
echo.
echo 3. Or test directly (no server needed):
echo      python test_sentence_translation.py --direct
echo.
echo Note: The first API request will take 2-5 minutes to build
echo       the FAISS index. After that, requests are fast!
echo.
pause
