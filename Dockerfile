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
RUN python -c "\
import os\n\
import sys\n\
from transformers import AutoModelForSequenceClassification, AutoTokenizer\n\
\n\
print('🚀 Starting BGE reranker model pre-download...')\n\
print(f'Cache directory: {os.getenv(\\\"HF_HOME\\\")}')\n\
\n\
try:\n\
    # Download and cache the model\n\
    model = AutoModelForSequenceClassification.from_pretrained(\n\
        'BAAI/bge-reranker-v2-m3',\n\
        cache_dir=os.getenv('HF_HOME'),\n\
        local_files_only=False,\n\
        force_download=False  # Use cached if available\n\
    )\n\
    print('✅ BGE reranker model downloaded successfully!')\n\
    \n\
    # Also download tokenizer\n\
    tokenizer = AutoTokenizer.from_pretrained(\n\
        'BAAI/bge-reranker-v2-m3', \n\
        cache_dir=os.getenv('HF_HOME')\n\
    )\n\
    print('✅ BGE reranker tokenizer downloaded successfully!')\n\
    \n\
except Exception as e:\n\
    print(f'❌ Model download failed: {e}')\n\
    sys.exit(1)\n\
"

# Verify model was downloaded
RUN echo "📦 Verifying downloaded models:" && \
    find ${HF_HOME} -name "*bge-reranker*" -type d | head -5 && \
    echo "✅ Model verification complete"

EXPOSE 8000

ENTRYPOINT ["python", "app.py"]