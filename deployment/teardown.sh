#!/bin/bash

set -e

echo "🧹 Cleaning up PerfectPick Demo..."

kubectl delete -f k8s/05-service.yaml --ignore-not-found=true
kubectl delete -f k8s/04-deployment.yaml --ignore-not-found=true
kubectl delete -f k8s/03-configmap.yaml --ignore-not-found=true
kubectl delete -f k8s/02-secret.yaml --ignore-not-found=true
kubectl delete -f k8s/01-namespace.yaml --ignore-not-found=true

# Wait for resources to be deleted
sleep 10

echo "✅ Demo cleanup complete!"