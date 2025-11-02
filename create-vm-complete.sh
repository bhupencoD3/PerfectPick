#!/bin/bash

PROJECT_ID="recommendation-service-473907"
VM_NAME="perfectpick-demo-vm"
ZONE="us-central1-a"

# Create startup script content
STARTUP_SCRIPT_CONTENT='#!/bin/bash

# Update system
apt-get update
apt-get upgrade -y

# Install Docker
apt-get install -y docker.io
systemctl enable docker
systemctl start docker

# Install Minikube dependencies  
apt-get install -y curl conntrack

# Install Minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
install minikube-linux-amd64 /usr/local/bin/minikube
rm minikube-linux-amd64

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
rm kubectl

# Add user to docker group
usermod -aG docker ubuntu

# Clone your project (UPDATE THIS URL)
git clone https://github.com/bhupencoD3/PerfectPick /home/ubuntu/perfectpick
chown -R ubuntu:ubuntu /home/ubuntu/perfectpick

# Start Minikube as ubuntu user
sudo -i -u ubuntu minikube start --driver=docker --memory=1800m --cpus=1

echo "✅ VM setup complete!"
echo "🚀 Ready to deploy PerfectPick demo"'

echo "🚀 Creating VM in project: $PROJECT_ID"

# Set the project
gcloud config set project $PROJECT_ID
echo "✅ Project set to: $(gcloud config get-value project)"

# Create the VM with e2-small using inline startup script
gcloud compute instances create $VM_NAME \
    --machine-type=e2-small \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=20GB \
    --boot-disk-type=pd-standard \
    --zone=$ZONE \
    --tags=http-server \
    --metadata=startup-script="$STARTUP_SCRIPT_CONTENT"

echo "✅ VM created successfully!"

# Get VM details
VM_IP=$(gcloud compute instances describe $VM_NAME --zone=$ZONE --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo ""
echo "🎉 VM Details:"
echo "=============="
echo "Project: $PROJECT_ID"
echo "VM Name: $VM_NAME"
echo "Machine Type: e2-small"
echo "IP Address: $VM_IP"
echo "Zone: $ZONE"
echo ""
echo "🔗 SSH to VM:"
echo "gcloud compute ssh $VM_NAME --zone=$ZONE"
echo ""
echo "⏳ VM is setting up (takes 2-3 minutes)..."
echo "Check setup progress: gcloud compute ssh $VM_NAME --zone=$ZONE --command='tail -f /var/log/syslog'"