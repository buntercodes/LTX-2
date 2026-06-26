# LTX-2 Inference Server - Cloud GPU Deployment Guide

**Version:** 1.0.0  
**Last Updated:** June 2026  
**Target Environment:** Ubuntu 22.04/24.04 LTS on Cloud GPU Instance

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [GPU Instance Setup](#2-gpu-instance-setup)
3. [System Dependencies](#3-system-dependencies)
4. [Python Environment](#4-python-environment)
5. [CUDA & PyTorch Installation](#5-cuda--pytorch-installation)
6. [LTX-2 Repository Setup](#6-ltx-2-repository-setup)
7. [Model Downloads](#7-model-downloads)
8. [Server Configuration](#8-server-configuration)
9. [Server Startup & Verification](#9-server-startup--verification)
10. [Firewall & Security](#10-firewall--security)
11. [Production Deployment (Systemd)](#11-production-deployment-systemd)
12. [Monitoring & Maintenance](#12-monitoring--maintenance)
13. [Troubleshooting](#13-troubleshooting)
14. [Appendix: GPU Instance Recommendations](#appendix-gpu-instance-recommendations)

---

## 1. Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA T4 (16GB) | NVIDIA A100 (80GB) or H100 (80GB) |
| RAM | 32GB | 64GB+ |
| Storage | 100GB SSD | 500GB+ NVMe SSD |
| Network | 1 Gbps | 10 Gbps |

### Software Requirements

- Ubuntu 22.04 LTS or 24.04 LTS
- NVIDIA GPU with CUDA support
- SSH access with key-based authentication
- Internet connection for downloads

### Cloud Provider Examples

- **AWS**: `p3.2xlarge` (V100), `p4d.24xlarge` (A100), `p5.48xlarge` (H100)
- **GCP**: `n1-standard-8` + `nvidia-tesla-t4`, `a2-highgpu-1g` (A100)
- **Azure**: `Standard_NC6s_v3` (V100), `Standard_ND96asr_v4` (A100)
- **Lambda Cloud**: `gpu_1x_a100_80gb`, `gpu_8x_a100_80gb`
- **RunPod**: `A100 80GB`, `H100 80GB`

---

## 2. GPU Instance Setup

### 2.1 Connect to Your Instance

```bash
# Replace with your instance's IP and key
ssh -i /path/to/your-key.pem ubuntu@YOUR_INSTANCE_IP
```

**Verification:**
```bash
# You should see the Ubuntu welcome message
whoami
# Expected output: ubuntu

pwd
# Expected output: /home/ubuntu
```

### 2.2 Update System Packages

```bash
sudo apt update && sudo apt upgrade -y
```

**Verification:**
```bash
# Check that updates completed successfully
echo $?
# Expected output: 0

# Check Ubuntu version
lsb_release -a
# Expected: Ubuntu 22.04.x LTS or 24.04.x LTS
```

---

## 3. System Dependencies

### 3.1 Install Required Packages

```bash
sudo apt install -y \
    build-essential \
    git \
    curl \
    wget \
    htop \
    tmux \
    nano \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release
```

**Verification:**
```bash
# Verify each tool is installed
git --version
curl --version | head -1
wget --version | head -1
htop --version | head -1
tmux -V
nano --version | head -1
```

### 3.2 Install NVIDIA Drivers (if not pre-installed)

```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
wget https://developer.download.nvidia.com/compute/cuda/repos/${distribution}/${distribution}/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install driver and CUDA toolkit
sudo apt install -y nvidia-driver-535 nvidia-utils-535 cuda-toolkit-12-1
```

**Verification:**
```bash
# Check NVIDIA driver
nvidia-smi
# Expected: GPU info table with driver version and CUDA version

# Check CUDA installation
nvcc --version
# Expected: cuda release 12.1
```

### 3.3 Reboot (if drivers were installed)

```bash
sudo reboot
```

After reboot, reconnect and verify:
```bash
nvidia-smi
# Should show GPU details without errors
```

---

## 4. Python Environment

### 4.1 Install Python 3.11+

```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip
```

**Verification:**
```bash
python3.11 --version
# Expected: Python 3.11.x

pip3 --version
# Expected: pip 2x.x.x from python3.11
```

### 4.2 Install UV Package Manager

UV is required for managing the LTX-2 workspace dependencies.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Add UV to your PATH:
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**Verification:**
```bash
uv --version
# Expected: uv x.x.x
```

---

## 5. CUDA & PyTorch Installation

### 5.1 Create Python Virtual Environment

```bash
cd /home/ubuntu
python3.11 -m venv .venv-ltx2
source .venv-ltx2/bin/activate
```

**Verification:**
```bash
which python
# Expected: /home/ubuntu/.venv-ltx2/bin/python

python --version
# Expected: Python 3.11.x
```

### 5.2 Install PyTorch with CUDA

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verification:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
# Expected output:
# PyTorch: 2.x.x+cu121
# CUDA available: True
# CUDA version: 12.1
# GPU: NVIDIA A100 80GB PCIe (or your GPU name)
```

---

## 6. LTX-2 Repository Setup

### 6.1 Clone Repository

```bash
cd /home/ubuntu
git clone https://github.com/Lightricks/LTX-2.git
cd LTX-2
```

**Verification:**
```bash
ls -la
# Expected: README.md, packages/, pyproject.toml, etc.
```

### 6.2 Install Dependencies with UV

```bash
# Sync dependencies (frozen for reproducibility)
uv sync --frozen
```

**Verification:**
```bash
# Check that virtual environment was created
ls -la .venv/
# Expected: bin/, lib/, pyvenv.cfg

# Verify key packages are installed
uv run python -c "import ltx_core; print('ltx_core OK')"
uv run python -c "import ltx_pipelines; print('ltx_pipelines OK')"
```

### 6.3 Verify LTX-2 Installation

```bash
uv run python -c "
from ltx_pipelines.distilled import DistilledPipeline
from ltx_core.model.video_vae import TilingConfig
print('LTX-2 pipeline import successful')
print(f'TilingConfig available: {TilingConfig}')
"
```

### 6.4 Install Server Dependencies

The FastAPI server requires additional Python packages not included in the base LTX-2 installation.

```bash
# Install server-specific dependencies
uv pip install -r server/requirements.txt
```

**Verification:**
```bash
# Verify server packages are installed
uv run python -c "
import fastapi
import uvicorn
import pydantic
import pydantic_settings
import httpx
import aiofiles
print(f'FastAPI: {fastapi.__version__}')
print(f'Uvicorn: {uvicorn.__version__}')
print(f'Pydantic: {pydantic.__version__}')
print('All server packages installed successfully')
"
```

---

## 7. Model Downloads

### 7.1 Install HuggingFace CLI

```bash
uv pip install huggingface_hub
```

**Verification:**
```bash
huggingface-cli --version
# Expected: huggingface-cli x.x.x
```

### 7.2 Download Required Models

Create models directory:
```bash
mkdir -p /home/ubuntu/LTX-2/models
cd /home/ubuntu/LTX-2/models
```

#### 7.2.1 Download Distilled Model Checkpoint

```bash
huggingface-cli download Lightricks/LTX-2.3 \
    ltx-2.3-22b-distilled-1.1.safetensors \
    --local-dir /home/ubuntu/LTX-2/models
```

**Verification:**
```bash
ls -lh /home/ubuntu/LTX-2/models/ltx-2.3-22b-distilled-1.1.safetensors
# Expected: ~44GB file
```

#### 7.2.2 Download Spatial Upscaler

```bash
huggingface-cli download Lightricks/LTX-2.3 \
    ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    --local-dir /home/ubuntu/LTX-2/models
```

**Verification:**
```bash
ls -lh /home/ubuntu/LTX-2/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
# Expected: ~2GB file
```

#### 7.2.3 Download Gemma Text Encoder

```bash
huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized \
    --local-dir /home/ubuntu/LTX-2/models/gemma-3-12b-it-qat-q4_0-unquantized
```

**Verification:**
```bash
ls -la /home/ubuntu/LTX-2/models/gemma-3-12b-it-qat-q4_0-unquantized/
# Expected: config.json, tokenizer.json, model files, etc.
```

#### 7.2.4 (Optional) Download LoRA Adapters

```bash
# Example: Download camera control LoRA
huggingface-cli download Lightricks/LTX-2-19b-LoRA-Camera-Control-Static \
    --local-dir /home/ubuntu/LTX-2/models/loras
```

### 7.3 Verify All Models

```bash
echo "=== Model Files ===" && \
ls -lh /home/ubuntu/LTX-2/models/*.safetensors && \
echo "" && \
echo "=== Gemma Encoder ===" && \
ls -la /home/ubuntu/LTX-2/models/gemma-3-12b-it-qat-q4_0-unquantized/ | head -10
```

---

## 8. Server Configuration

### 8.1 Navigate to Server Directory

```bash
cd /home/ubuntu/LTX-2/server
```

### 8.2 Create Environment Configuration

```bash
cp .env.example .env
```

### 8.3 Edit Configuration File

Open the `.env` file with a text editor:
```bash
nano /home/ubuntu/LTX-2/server/.env
```

**Replace the entire contents with the following** (update paths as needed):

```env
# =============================================================================
# LTX-2 Inference Server Configuration
# =============================================================================

# Server settings
LTX_HOST=0.0.0.0
LTX_PORT=8000
LTX_WORKERS=1
LTX_LOG_LEVEL=INFO

# =============================================================================
# MODEL PATHS (REQUIRED - Verify these paths exist!)
# =============================================================================
LTX_CHECKPOINT_PATH=/home/ubuntu/LTX-2/models/ltx-2.3-22b-distilled-1.1.safetensors
LTX_GEMMA_ROOT=/home/ubuntu/LTX-2/models/gemma-3-12b-it-qat-q4_0-unquantized
LTX_SPATIAL_UPSAMPLER_PATH=/home/ubuntu/LTX-2/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors

# Optional LoRAs (JSON array format)
# LTX_LORA_PATHS=["/home/ubuntu/LTX-2/models/loras/your-lora.safetensors"]

# =============================================================================
# DEFAULT GENERATION PARAMETERS
# =============================================================================
LTX_DEFAULT_HEIGHT=512
LTX_DEFAULT_WIDTH=768
LTX_DEFAULT_NUM_FRAMES=121
LTX_DEFAULT_FRAME_RATE=24.0
LTX_DEFAULT_SEED=42

# =============================================================================
# RESOURCE LIMITS
# =============================================================================
LTX_MAX_CONCURRENT_JOBS=2
LTX_MAX_HEIGHT=1088
LTX_MAX_WIDTH=1920
LTX_MAX_NUM_FRAMES=321

# =============================================================================
# FILE MANAGEMENT
# =============================================================================
LTX_OUTPUT_DIR=/home/ubuntu/LTX-2/server/outputs
LTX_TEMP_DIR=/home/ubuntu/LTX-2/server/temp
LTX_MAX_OUTPUT_FILES=100
LTX_CLEANUP_AFTER_HOURS=24.0

# =============================================================================
# OPTIMIZATION (Uncomment as needed)
# =============================================================================
# LTX_QUANTIZATION=fp8-cast
# LTX_ENABLE_COMPILE=true
# LTX_COMPILE_MODE=reduce-overhead
# LTX_OFFLOAD_MODE=none

# =============================================================================
# CORS
# =============================================================================
LTX_CORS_ORIGINS=["*"]
```

Save and exit: `Ctrl+X`, then `Y`, then `Enter`

### 8.4 Verify Model Paths Exist

```bash
# Run this verification script
echo "=== Verifying Model Paths ===" && \
test -f /home/ubuntu/LTX-2/models/ltx-2.3-22b-distilled-1.1.safetensors && echo "✓ Checkpoint found" || echo "✗ Checkpoint MISSING" && \
test -d /home/ubuntu/LTX-2/models/gemma-3-12b-it-qat-q4_0-unquantized && echo "✓ Gemma encoder found" || echo "✗ Gemma encoder MISSING" && \
test -f /home/ubuntu/LTX-2/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors && echo "✓ Upscaler found" || echo "✗ Upscaler MISSING"
```

### 8.5 Create Output Directories

```bash
mkdir -p /home/ubuntu/LTX-2/server/outputs
mkdir -p /home/ubuntu/LTX-2/server/temp
```

**Verification:**
```bash
ls -la /home/ubuntu/LTX-2/server/
# Expected: outputs/ and temp/ directories visible
```

---

## 9. Server Startup & Verification

### 9.1 Activate Virtual Environment

```bash
cd /home/ubuntu/LTX-2
source .venv/bin/activate
```

**Verification:**
```bash
which python
# Expected: /home/ubuntu/LTX-2/.venv/bin/python
```

### 9.2 Test Server Import

```bash
cd /home/ubuntu/LTX-2/server
python -c "
import sys
sys.path.insert(0, '/home/ubuntu/LTX-2/server')
from config import Settings
from models.schemas import GenerateRequest
from services.pipeline_service import PipelineService
print('✓ All server modules import successfully')
"
```

### 9.3 Start the Server

**Option A: Direct Python execution (for testing)**
```bash
cd /home/ubuntu/LTX-2/server
python main.py
```

**Option B: With tmux (recommended for SSH sessions)**
```bash
# Start a new tmux session
tmux new-session -d -s ltx2-server

# Run the server inside tmux
tmux send-keys -t ltx2-server "cd /home/ubuntu/LTX-2/server && python main.py" Enter

# Attach to the session
tmux attach -t ltx2-server
```

**Option C: With nohup (background process)**
```bash
cd /home/ubuntu/LTX-2/server
nohup python main.py > /home/ubuntu/LTX-2/server/server.log 2>&1 &

# Check if server is running
sleep 10
cat /home/ubuntu/LTX-2/server/server.log | tail -20
```

### 9.4 Wait for Model Loading

The model loading takes 30-120 seconds depending on GPU. Watch for:

```
INFO - Loading DistilledPipeline model...
INFO - Model loaded successfully in XX.XXs
INFO - Server initialized successfully
INFO - Server will run on 0.0.0.0:8000
```

### 9.5 Verify Server is Running

**Test 1: Health Check**
```bash
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "active_jobs": 0,
  "queue_size": 0,
  "gpu_memory_used": 44.5,
  "gpu_memory_total": 80.0,
  "uptime_seconds": 45.23
}
```

**Test 2: Model Info**
```bash
curl http://localhost:8000/api/v1/model/info
```

Expected response:
```json
{
  "checkpoint_path": "/home/ubuntu/LTX-2/models/ltx-2.3-22b-distilled-1.1.safetensors",
  "spatial_upsampler_path": "/home/ubuntu/LTX-2/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
  "lora_count": 0,
  "quantization": null,
  "compile_enabled": false,
  "offload_mode": "none",
  "default_resolution": "768x512",
  "max_resolution": "1920x1088",
  "supported_frame_counts": [9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 129, 137, 145, 153]
}
```

**Test 3: API Documentation**
```bash
# Check Swagger docs are accessible
curl -s http://localhost:8000/docs | head -5
# Should return HTML content
```

**Test 4: Submit Test Generation**
```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over the ocean with gentle waves",
    "height": 512,
    "width": 768,
    "num_frames": 41,
    "seed": 42
  }'
```

Expected response:
```json
{
  "job_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "status": "pending",
  "message": "Job queued successfully",
  "created_at": "2026-06-26T12:00:00",
  "estimated_seconds": 4.1
}
```

**Test 5: Check Job Status**
```bash
# Replace JOB_ID with the job_id from previous response
JOB_ID="your-job-id-here"

curl "http://localhost:8000/api/v1/jobs/$JOB_ID"
```

**Test 6: Download Generated Video**
```bash
curl -o /home/ubuntu/test_output.mp4 "http://localhost:8000/api/v1/download/$JOB_ID"
ls -lh /home/ubuntu/test_output.mp4
```

### 9.6 Verify GPU Utilization

```bash
# In a separate terminal
nvidia-smi
```

Expected: GPU memory usage should show ~40-50GB for the distilled model.

---

## 10. Firewall & Security

### 10.1 Configure UFW Firewall

```bash
# Allow SSH
sudo ufw allow 22/tcp

# Allow HTTP API
sudo ufw allow 8000/tcp

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status
```

**Verification:**
```bash
sudo ufw status numbered
# Expected: Rules for 22/tcp and 8000/tcp
```

### 10.2 (Optional) Cloud Security Group

For AWS/GCP/Azure, ensure your security group allows:
- **Inbound**: Port 8000 (HTTP) from your IP or `0.0.0.0/0`
- **Inbound**: Port 22 (SSH) from your IP

### 10.3 (Optional) Enable Authentication

For production, add API key authentication:

```bash
# Install middleware
uv pip install fastapi-security
```

Update `main.py` to add authentication middleware.

---

## 11. Production Deployment (Systemd)

### 11.1 Create Systemd Service File

```bash
sudo nano /etc/systemd/system/ltx2-server.service
```

Paste the following content:

```ini
[Unit]
Description=LTX-2 Inference Server
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/LTX-2/server
Environment="PATH=/home/ubuntu/LTX-2/.venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/ubuntu/LTX-2/.venv/bin/python main.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/home/ubuntu/LTX-2/server/outputs /home/ubuntu/LTX-2/server/temp

[Install]
WantedBy=multi-user.target
```

Save and exit.

### 11.2 Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable ltx2-server

# Start the service
sudo systemctl start ltx2-server
```

**Verification:**
```bash
# Check service status
sudo systemctl status ltx2-server

# Expected: active (running)

# Check logs
sudo journalctl -u ltx2-server -f --no-pager | tail -20
```

### 11.3 Manage Service

```bash
# Stop the service
sudo systemctl stop ltx2-server

# Restart the service
sudo systemctl restart ltx2-server

# View logs
sudo journalctl -u ltx2-server -f
```

---

## 12. Monitoring & Maintenance

### 12.1 Monitor GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or install nvtop for better visualization
sudo apt install -y nvtop
nvtop
```

### 12.2 Monitor Server Logs

```bash
# If using systemd
sudo journalctl -u ltx2-server -f

# If using tmux
tmux attach -t ltx2-server

# If using nohup
tail -f /home/ubuntu/LTX-2/server/server.log
```

### 12.3 Check Disk Usage

```bash
# Check output directory size
du -sh /home/ubuntu/LTX-2/server/outputs/

# Check available disk space
df -h /home/ubuntu/
```

### 12.4 Clean Up Old Outputs

```bash
# Manual cleanup (older than 24 hours)
find /home/ubuntu/LTX-2/server/outputs/ -name "*.mp4" -mtime +1 -delete

# Or use the API endpoint
curl -X DELETE "http://localhost:8000/api/v1/jobs/JOB_ID"
```

### 12.5 Performance Monitoring Script

Create `/home/ubuntu/LTX-2/monitor.sh`:

```bash
#!/bin/bash
echo "=== LTX-2 Server Status ==="
echo ""
echo "--- Server Process ---"
pgrep -f "python main.py" > /dev/null && echo "Server: RUNNING" || echo "Server: STOPPED"
echo ""
echo "--- GPU Status ---"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo ""
echo "--- API Health ---"
curl -s http://localhost:8000/api/v1/health | python3 -m json.tool 2>/dev/null || echo "API: Not responding"
echo ""
echo "--- Disk Usage ---"
df -h /home/ubuntu/ | tail -1
echo ""
echo "--- Output Files ---"
ls -1 /home/ubuntu/LTX-2/server/outputs/*.mp4 2>/dev/null | wc -l
echo " videos in output directory"
```

Make it executable:
```bash
chmod +x /home/ubuntu/LTX-2/monitor.sh
```

Run monitoring:
```bash
/home/ubuntu/LTX-2/monitor.sh
```

---

## 13. Troubleshooting

### Issue: "CUDA out of memory"

**Solution:**
```bash
# Enable quantization in .env
# Edit /home/ubuntu/LTX-2/server/.env
LTX_QUANTIZATION=fp8-cast

# Or enable offloading
LTX_OFFLOAD_MODE=cpu

# Restart server
sudo systemctl restart ltx2-server
```

### Issue: "Model not loaded"

**Solution:**
```bash
# Check if model files exist
ls -la /home/ubuntu/LTX-2/models/

# Check server logs for errors
sudo journalctl -u ltx2-server -n 50

# Verify paths in .env match actual file locations
cat /home/ubuntu/LTX-2/server/.env | grep PATH
```

### Issue: "Connection refused"

**Solution:**
```bash
# Check if server is running
pgrep -f "python main.py"

# Check if port is in use
sudo lsof -i :8000

# Check firewall
sudo ufw status

# Check server logs
sudo journalctl -u ltx2-server -n 50
```

### Issue: "ModuleNotFoundError"

**Solution:**
```bash
# Ensure virtual environment is activated
source /home/ubuntu/LTX-2/.venv/bin/activate

# Reinstall dependencies
cd /home/ubuntu/LTX-2
uv sync --frozen

# Test import
python -c "import ltx_pipelines; print('OK')"
```

### Issue: Slow generation

**Solution:**
```bash
# Enable torch.compile
# Edit .env
LTX_ENABLE_COMPILE=true
LTX_COMPILE_MODE=reduce-overhead

# Enable quantization
LTX_QUANTIZATION=fp8-cast

# Reduce default frame count
LTX_DEFAULT_NUM_FRAMES=41

# Restart server
sudo systemctl restart ltx2-server
```

### Issue: GPU not detected

**Solution:**
```bash
# Check NVIDIA drivers
nvidia-smi

# If not working, reinstall drivers
sudo apt install -y nvidia-driver-535
sudo reboot
```

---

## Appendix: GPU Instance Recommendations

### By Use Case

| Use Case | Recommended GPU | RAM | Storage | Est. Cost/hr |
|----------|-----------------|-----|---------|--------------|
| Testing/Development | T4 (16GB) | 32GB | 100GB | $0.50-1.00 |
| Small Batch Production | A10G (24GB) | 64GB | 200GB | $1.00-2.00 |
| Production | A100 (40GB) | 64GB | 500GB | $3.00-5.00 |
| High Volume | A100 (80GB) | 128GB | 1TB | $5.00-8.00 |
| Maximum Performance | H100 (80GB) | 256GB | 2TB | $10.00-15.00 |

### Memory Requirements

The LTX-2 model weights are stored and loaded in **BF16 (bfloat16)** precision, not FP32. This halves the memory requirement compared to FP32 while maintaining training stability and inference quality.

| Configuration | GPU Memory | System RAM |
|--------------|------------|------------|
| Default (BF16) | ~45GB | ~64GB |
| With FP8 quantization | ~25GB | ~32GB |
| With CPU offloading | ~8GB | ~80GB |
| With Disk offloading | ~5GB | ~16GB |

**Note:** BF16 uses the same number of exponent bits as FP32 (8 bits) but fewer mantissa bits (7 vs 23), providing the same dynamic range with reduced precision. This is ideal for deep learning workloads.

**Source Reference:** The dtype is set in `packages/ltx-pipelines/src/ltx_pipelines/distilled.py`:
```python
self.dtype = torch.bfloat16  # Line 61
```

### Cloud Provider Quick Start

#### AWS (using CLI)
```bash
# Launch p4d.24xlarge (8x A100)
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type p4d.24xlarge \
    --key-name your-key \
    --security-group-ids sg-xxxxx \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3"}}]'
```

#### GCP (using gcloud)
```bash
# Launch a2-highgpu-1g (1x A100)
gcloud compute instances create ltx2-server \
    --zone=us-central1-a \
    --machine-type=a2-highgpu-1g \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=500GB \
    --accelerator=type=nvidia-tesla-a100,count=1
```

#### Lambda Cloud
```bash
# Via web UI: Create instance → Select GPU → Deploy
# SSH into instance and follow this guide
```

---

## Quick Reference Card

```bash
# === CONNECT ===
ssh -i key.pem ubuntu@INSTANCE_IP

# === START SERVER ===
cd /home/ubuntu/LTX-2
source .venv/bin/activate
cd server
python main.py

# === TEST API ===
curl http://localhost:8000/api/v1/health
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A cat playing piano","num_frames":41}'

# === CHECK STATUS ===
nvidia-smi
sudo systemctl status ltx2-server
/home/ubuntu/LTX-2/monitor.sh

# === MANAGEMENT ===
sudo systemctl restart ltx2-server
sudo journalctl -u ltx2-server -f
```

---

**Document Version:** 1.0.0  
**Last Updated:** June 2026  
**Author:** LTX-2 Server Team
