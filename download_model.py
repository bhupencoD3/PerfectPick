# download_model.py
import os
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer

print('🚀 Starting BGE reranker model pre-download...')
print(f'Cache directory: {os.getenv("HF_HOME")}')

try:
    # Download and cache the model
    model = AutoModelForSequenceClassification.from_pretrained(
        'BAAI/bge-reranker-v2-m3',
        cache_dir=os.getenv('HF_HOME'),
        local_files_only=False,
        force_download=False
    )
    print('✅ BGE reranker model downloaded successfully!')
    
    # Also download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        'BAAI/bge-reranker-v2-m3', 
        cache_dir=os.getenv('HF_HOME')
    )
    print('✅ BGE reranker tokenizer downloaded successfully!')
    
except Exception as e:
    print(f'❌ Model download failed: {e}')
    sys.exit(1)