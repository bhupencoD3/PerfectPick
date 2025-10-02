#!/bin/bash

echo "🚀 Starting PerfectPick deployment..."

# Create namespaces
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/namespace.yaml

# Deploy application
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yml
kubectl apply -f k8s/service.yml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingres.yaml

# Deploy monitoring stack
kubectl apply -f prometheus/prometheus-config.yaml
kubectl apply -f prometheus/prometheus-deployment.yaml
kubectl apply -f prometheus/prometheus-service.yaml

kubectl apply -f grafana/grafana-datasources.yaml
kubectl apply -f grafana/grafana-secret.yaml
kubectl apply -f grafana/grafana-deployment.yaml
kubectl apply -f grafana/grafana-service.yaml

# Wait for pods to be ready
echo "⏳ Waiting for pods to be ready..."
kubectl wait --for=condition=ready pod -l app=perfectpick-api -n perfectpick --timeout=300s
kubectl wait --for=condition=ready pod -l app=prometheus -n monitoring --timeout=300s
kubectl wait --for=condition=ready pod -l app=grafana -n monitoring --timeout=300s

# Get NodePorts
echo "📊 Getting service access information..."
PERFECTPICK_PORT=$(kubectl get svc perfectpick-service -n perfectpick -o jsonpath='{.spec.ports[0].nodePort}')
GRAFANA_PORT=$(kubectl get svc grafana -n monitoring -o jsonpath='{.spec.ports[0].nodePort}')
PROMETHEUS_PORT=$(kubectl get svc prometheus -n monitoring -o jsonpath='{.spec.ports[0].nodePort}')

echo "✅ Deployment completed!"
echo ""
echo "🌐 Access URLs:"
echo "PerfectPick API: http://34.14.203.32:${PERFECTPICK_PORT}"
echo "PerfectPick Ingress: http://34.14.203.32.nip.io"
echo "Grafana: http://34.14.203.32:${GRAFANA_PORT} (admin/your-secure-password-here)"
echo "Prometheus: http://34.14.203.32:${PROMETHEUS_PORT}"
echo ""
echo "🔧 To create secrets, run:"
echo "kubectl apply -f k8s/secret.yaml  # After filling the template with your actual secrets"