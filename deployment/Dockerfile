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

USER appuser

EXPOSE 8000

ENTRYPOINT ["python", "app.py"]