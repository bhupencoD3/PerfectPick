#!/bin/bash
set -e

echo "🚀 Starting perfectpick Service..."

# Optional: reingest vector DB (controlled by env variable)
if [ "$INGEST_DB" = "true" ]; then
    echo "📥 Reingesting documents into vector DB..."
    python scripts/ingest_documents.py
fi

# Start Flask app with production settings
exec gunicorn app:app \
    --bind 0.0.0.0:8000 \
    --workers 4 \
    --threads 2 \
    --timeout 120
