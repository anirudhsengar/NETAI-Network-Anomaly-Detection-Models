# Deployment Guide

## Local Development

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

# Full pipeline
python scripts/generate_data.py          # Generate synthetic data
python scripts/train.py                  # Train all models
python scripts/evaluate.py --output results.json  # Evaluate
python scripts/serve.py --port 8000      # Start API server
```

## Docker Deployment

### CPU Inference Image
```bash
docker build -f docker/Dockerfile -t netai-anomaly:latest .
docker run -p 8000:8000 -v ./checkpoints:/app/checkpoints netai-anomaly:latest
```

### GPU Training Image
```bash
docker build -f docker/Dockerfile.gpu -t netai-anomaly:gpu .
docker run --gpus all -v ./data:/app/data -v ./checkpoints:/app/checkpoints netai-anomaly:gpu
```

## Kubernetes Deployment on NRP

### Prerequisites
- Access to NRP Kubernetes cluster
- `kubectl` configured with NRP credentials
- Namespace `netai` created

### Step-by-Step Deployment

1. **Create namespace and storage:**
```bash
kubectl create namespace netai
kubectl apply -f k8s/pvc.yaml
```

2. **Deploy configuration:**
```bash
kubectl apply -f k8s/configmap.yaml
```

3. **Upload training data** (copy data to the PVC):
```bash
kubectl cp data/network_telemetry.db netai/<data-pod>:/app/data/
```

4. **Run GPU training job:**
```bash
kubectl apply -f k8s/gpu-training-job.yaml
kubectl logs -f job/netai-gpu-training -n netai
```

5. **Deploy inference service:**
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

6. **Verify deployment:**
```bash
kubectl get pods -n netai
kubectl port-forward svc/netai-anomaly-service 8000:80 -n netai
curl http://localhost:8000/health
```

### Resource Requirements

| Component | CPU | Memory | GPU |
|-----------|-----|--------|-----|
| Training Job | 2 cores | 4-16 Gi | 1x NVIDIA GPU |
| Inference Pod | 0.25-1 core | 512Mi-2Gi | None (CPU) |

### Scaling

The inference deployment defaults to 2 replicas and can be scaled:
```bash
kubectl scale deployment netai-anomaly-inference --replicas=4 -n netai
```

For GPU inference, modify `k8s/deployment.yaml` to include GPU resource requests.
