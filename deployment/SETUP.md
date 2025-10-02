# PerfectPick Deployment Setup

## Prerequisites
- Kubernetes cluster
- kubectl configured
- Docker image `bhupencod3v1.0` available

## Setup Steps

### 1. Create Secrets
First, create your secret file from the template:
```bash
cp k8s/secret-template.yaml k8s/secret.yaml
# Edit k8s/secret.yaml with your actual credentials
kubectl apply -f k8s/secret.yaml