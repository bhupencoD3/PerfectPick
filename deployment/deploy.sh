#!/bin/bash

set -e

echo "🚀 Deploying PerfectPick Demo..."

# Create namespace if it doesn't exist
kubectl apply -f k8s/namespace.yaml

# Create secret from environment variables (if provided)
if [ ! -f "k8s/02-secret.yaml" ]; then
    echo "⚠️  Secret file not found. Creating from template..."
    cat > k8s/02-secret.yaml << EOF
apiVersion: v1
kind: Secret
metadata:
  name: perfectpick-secrets
  namespace: perfectpick
type: Opaque
stringData:
  DB_URL: "$DB_URL"
  GROQ_API_KEY: "$GROQ_API_KEY"
  ASTRA_DB_APPLICATION_TOKEN: "$ASTRA_DB_APPLICATION_TOKEN"
  ASTRA_DB_API_ENDPOINT: "$ASTRA_DB_API_ENDPOINT"
  HF_TOKEN: "$HF_TOKEN"
EOF
fi

# Apply all k8s manifests in order
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/perfectpick-api -n perfectpick --timeout=180s

echo "✅ Demo deployment complete!"
echo "📊 Pod status:"
kubectl get pods -n perfectpick

echo "🌐 Services:"
kubectl get svc -n perfectpick

echo "🎯 To access your application:"
echo "   kubectl port-forward -n perfectpick svc/perfectpick-service 8000:80"
echo "   Then visit: http://localhost:8000"