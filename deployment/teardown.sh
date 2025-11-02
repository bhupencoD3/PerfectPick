#!/bin/bash

set -e

echo "🧹 Cleaning up PerfectPick Demo..."

kubectl delete -f k8s/service.yaml --ignore-not-found=true
kubectl delete -f k8s/deployment.yaml --ignore-not-found=true
kubectl delete -f k8s/configmap.yaml --ignore-not-found=true
kubectl delete -f k8s/secret.yaml --ignore-not-found=true
kubectl delete -f k8s/namespace.yaml --ignore-not-found=true

# Wait for resources to be deleted
sleep 10

echo "✅ Demo cleanup complete!"