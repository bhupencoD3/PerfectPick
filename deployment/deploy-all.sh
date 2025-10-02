#!/bin/bash

echo "🚀 Starting PerfectPick deployment on Minikube..."

# Set Minikube Docker environment (optional - you might have already done this)
eval $(minikube docker-env)

# Create namespaces
kubectl apply -f k8s/namespace.yaml
kubectl create namespace monitoring

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

# Get Minikube IP and NodePorts
MINIKUBE_IP=$(minikube ip)
PERFECTPICK_PORT=$(kubectl get svc perfectpick-service -n perfectpick -o jsonpath='{.spec.ports[0].nodePort}')
GRAFANA_PORT=$(kubectl get svc grafana -n monitoring -o jsonpath='{.spec.ports[0].nodePort}')
PROMETHEUS_PORT=$(kubectl get svc prometheus -n monitoring -o jsonpath='{.spec.ports[0].nodePort}')

echo "✅ Deployment completed!"
echo ""
echo "🌐 Access URLs:"
echo "PerfectPick API: http://${MINIKUBE_IP}:${PERFECTPICK_PORT}"
echo "Grafana: http://${MINIKUBE_IP}:${GRAFANA_PORT} (admin/your-secure-password-here)"
echo "Prometheus: http://${MINIKUBE_IP}:${PROMETHEUS_PORT}"
echo ""
echo "🔧 To create secrets, run:"
echo "kubectl apply -f k8s/secret.yaml"