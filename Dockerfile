FROM python:3.12-slim

# System dependencies including DNS tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    dnsutils \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

COPY requirements.txt .

# Install PyTorch CPU first
RUN pip install --no-cache-dir --upgrade pip --root-user-action=ignore && \
    pip install --no-cache-dir torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# Copy app code AND data
COPY . .

# Verify data files are copied
RUN echo "📁 Data files in container:" && \
    find data/ -name "*.csv" -type f | head -10 && \
    echo "✅ Data files verified"

# Create non-root user but allow model downloads
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app \
    && mkdir -p /home/appuser/.cache \
    && chown -R appuser:appuser /home/appuser

# Set Hugging Face cache to writable location
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p ${HF_HOME} && chown -R appuser:appuser ${HF_HOME}

# 🔥 PRE-DOWNLOAD BGE RERANKER MODEL DURING BUILD
# Switch to appuser for model download
USER appuser

# Pre-download the BGE reranker model (this will take 5-10 minutes)
RUN python -c "
import os
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer

print('🚀 Starting BGE reranker model pre-download...')
print(f'Cache directory: {os.getenv(\\\"HF_HOME\\\")}')

try:
    # Download and cache the model
    model = AutoModelForSequenceClassification.from_pretrained(
        'BAAI/bge-reranker-v2-m3',
        cache_dir=os.getenv('HF_HOME'),
        local_files_only=False,
        force_download=False  # Use cached if available
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
"

# Verify model was downloaded
RUN echo "📦 Verifying downloaded models:" && \
    find ${HF_HOME} -name "*bge-reranker*" -type d | head -5 && \
    echo "✅ Model verification complete"

EXPOSE 8000

ENTRYPOINT ["python", "app.py"]