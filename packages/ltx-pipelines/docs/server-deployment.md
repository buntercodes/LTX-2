# LTX-2 Distilled Pipeline — Cloud GPU FastAPI Server Deployment

This guide covers deploying the FastAPI video generation server on a cloud GPU instance with an **RTX Pro 6000 96GB VRAM**.

---

## Instance Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | RTX Pro 6000 (96 GB) | RTX Pro 6000 (96 GB) |
| VRAM | 30 GB (fp8-cast) / 50 GB (bf16) | 96 GB |
| System RAM | 64 GB | 128 GB |
| Disk | 100 GB (OS + models + scratch) | 200 GB |
| OS | Ubuntu 22.04 / 24.04 LTS | Ubuntu 24.04 LTS |
| CUDA | 12.9 | 12.9 |

---

## Step 1 — OS & Driver Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install build tools
sudo apt install -y build-essential git curl wget ffmpeg

# Install NVIDIA driver (data-center driver for RTX Pro series)
# Option A: Latest Data Center driver
wget https://us.download.nvidia.com/tesla/570.124.06/NVIDIA-Linux-x86_64-570.124.06.run
sudo sh NVIDIA-Linux-x86_64-570.124.06.run --no-questions --ui=none

# Option B: Via Ubuntu's repository (may lag behind)
sudo apt install -y nvidia-driver-570-server nvidia-utils-570-server
sudo reboot

# Verify after reboot
nvidia-smi
```

```bash
# Install CUDA 12.9 toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda-repo-ubuntu2404-12-9-local_12.9.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-9-local_12.9.0-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-9-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install -y cuda-toolkit-12-9
echo 'export PATH=/usr/local/cuda-12.9/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version  # Should show 12.9
```

---

## Step 2 — Clone Repository & Install Dependencies

```bash
# Clone the repo
git clone https://github.com/Lightricks/LTX-2.git
cd LTX-2
```

### Option A: uv (recommended)

```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env

# Install dependencies with server extras
uv sync --extra server

# Activate virtualenv
source .venv/bin/activate
```

> **Note:** `--extra xformers` is not included because it triggers a resolver conflict with `fp8-trtllm` (which pins `transformers==4.53.1` while we require `4.57.6`). The distilled pipeline does not require xformers to function.

### Option B: pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e "packages/ltx-core"
pip install -e "packages/ltx-pipelines[server]"
```

> **xformers** enables memory-efficient attention. For Hopper/Blackwell GPUs, also install [Flash Attention 3](https://github.com/Dao-AILab/flash-attention) (`pip install flash-attn --no-build-isolation`).

---

## Step 3 — Download Required Models

Download these from [HuggingFace LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3):

```bash
# Create model storage directory
mkdir -p ~/models/ltx-2.3 ~/models/gemma

# 1. Distilled checkpoint (~44 GB)
huggingface-cli download Lightricks/LTX-2.3 \
    ltx-2.3-22b-distilled-1.1.safetensors \
    --local-dir ~/models/ltx-2.3

# 2. Spatial upsampler (~500 MB)
huggingface-cli download Lightricks/LTX-2.3 \
    ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    --local-dir ~/models/ltx-2.3

# 3. Gemma text encoder (~24 GB total, all files)
huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized \
    --local-dir ~/models/gemma --no-symlinks
```

> **Tip:** Set `export HF_HUB_ENABLE_HF_TRANSFER=1` before downloading for faster transfers (requires `pip install huggingface-hub[hf-transfer]`).

Alternatively, use Python:

```python
from huggingface_hub import hf_hub_download, snapshot_download

hf_hub_download("Lightricks/LTX-2.3", "ltx-2.3-22b-distilled-1.1.safetensors",
                 local_dir="/root/models/ltx-2.3")
hf_hub_download("Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
                 local_dir="/root/models/ltx-2.3")
snapshot_download("google/gemma-3-12b-it-qat-q4_0-unquantized",
                  local_dir="/root/models/gemma")
```

---

## Step 4 — Environment Variables

```bash
# Required for FP8 quantization memory efficiency
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Disable flash-attention (incompatible with RTX Pro 6000 / Blackwell TMA)
export DISABLE_FLASH_ATTN=1
export XFORMERS_DISABLED=1

# Optional: limit visible GPUs if you have multiple
export CUDA_VISIBLE_DEVICES=0
```

Add these to `~/.bashrc` to persist across sessions:

```bash
echo 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' >> ~/.bashrc
echo 'export DISABLE_FLASH_ATTN=1' >> ~/.bashrc
echo 'export XFORMERS_DISABLED=1' >> ~/.bashrc
```

---

## Step 5 — Run the Server

### Quick start (bf16, no quantization — uses ~50 GB VRAM)

```bash
python -m ltx_pipelines.server \
    --distilled-checkpoint-path ~/models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors \
    --gemma-root ~/models/gemma \
    --spatial-upsampler-path ~/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    --host 0.0.0.0 \
    --port 8000
```

### Production mode (fp8-cast — uses ~30 GB VRAM, faster)

```bash
python -m ltx_pipelines.server \
    --distilled-checkpoint-path ~/models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors \
    --gemma-root ~/models/gemma \
    --spatial-upsampler-path ~/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    --quantization fp8-cast \
    --host 0.0.0.0 \
    --port 8000
```

### With LoRA adapters

```bash
python -m ltx_pipelines.server \
    --distilled-checkpoint-path ~/models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors \
    --gemma-root ~/models/gemma \
    --spatial-upsampler-path ~/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    --quantization fp8-cast \
    --lora "/root/models/lora/ic-lora-detailer.safetensors 0.8" \
    --host 0.0.0.0 \
    --port 8000
```

Expected output:

```
╔══════════════════════════════════════════════════════╗
║              LTX-2 Video Generation API              ║
╚══════════════════════════════════════════════════════╝
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:ltx_server:Loading DistilledPipeline...
INFO:ltx_server:Pipeline loaded successfully.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## Step 5b — Cloud Storage (optional, for fast downloads)

GPU instances typically have limited upload bandwidth. Upload generated videos to S3-compatible cloud storage and serve via pre-signed URLs. The client download then comes from the cloud CDN instead of the GPU instance.

### Cloudflare R2 (recommended — zero egress fees)

1. Create a free [Cloudflare R2](https://developers.cloudflare.com/r2/) account
2. Create a bucket (e.g. `ltx-videos`)
3. Generate an API token with **Object Read & Write** permissions
4. Set environment variables:

```bash
export S3_ENDPOINT="https://<account_id>.r2.cloudflarestorage.com"
export S3_BUCKET="ltx-videos"
export S3_REGION="auto"
export S3_ACCESS_KEY="<token_id>"
export S3_SECRET_KEY="<token_secret>"
export S3_URL_EXPIRES=3600    # URL lifetime in seconds (default 1 hour)
```

### AWS S3

```bash
export S3_ENDPOINT="https://s3.us-east-1.amazonaws.com"
export S3_BUCKET="ltx-videos"
export S3_REGION="us-east-1"
export S3_ACCESS_KEY="AKIA..."
export S3_SECRET_KEY="..."
```

> **Note:** With R2 you pay zero egress. With S3 you pay $0.09/GB egress.

When cloud storage is configured, the server uploads each completed task's video and returns a 302 redirect to a pre-signed URL on `GET /task/{id}/video`. If cloud storage is not configured, the server falls back to serving the video directly from the local filesystem.

Add these to `~/.bashrc` and the systemd service `Environment=` section so they persist across restarts.

---

## Step 6 — Using the API

The API uses an asynchronous task model — you submit a generation job and poll for its status. This avoids timeouts and allows queuing multiple jobs.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server health + queue depth |
| `POST` | `/txt2vid` | Submit text-to-video job → returns `task_id` (202) |
| `POST` | `/img2vid` | Submit image-to-video job → returns `task_id` (202) |
| `GET` | `/task/{task_id}` | Poll task status |
| `GET` | `/task/{task_id}/video` | Download completed video (404 if not ready) |
| `GET` | `/queue` | List all tasks |

### Health Check

```bash
curl http://localhost:8000/health
```

```json
{"status":"ok","gpu_available":true,"queue_depth":0}
```

### Text-to-Video

**Step 1 — Submit:**

```bash
curl -X POST http://localhost:8000/txt2vid \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "A golden retriever running through a sunlit meadow, wind rustling through tall grass, cinematic warm lighting, shallow depth of field",
        "seed": 42,
        "height": 1088,
        "width": 1920,
        "num_frames": 121,
        "frame_rate": 24.0
    }'
```

Response (HTTP 202):
```json
{"task_id":"a1b2c3d4e5f67890abcdef1234567890","status":"queued","message":"Task submitted"}
```

**Step 2 — Poll until completed:**

```bash
TASK_ID="a1b2c3d4e5f67890abcdef1234567890"

# Poll status
curl http://localhost:8000/task/$TASK_ID
```

```json
{"task_id":"a1b2c3d4...","status":"running","prompt":"A golden...","seed":42,...,"elapsed":null}

{"task_id":"a1b2c3d4...","status":"completed","prompt":"A golden...","seed":42,...,"elapsed":43.2}
```

**Step 3 — Download video:**

```bash
curl -o output.mp4 http://localhost:8000/task/$TASK_ID/video
```

### Image-to-Video

```bash
TASK_ID=$(curl -s -X POST http://localhost:8000/img2vid \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "A person walking through a futuristic city at night, neon reflections on wet pavement",
        "seed": 100,
        "height": 1088,
        "width": 1920,
        "num_frames": 121,
        "frame_rate": 24.0,
        "enhance_prompt": true,
        "images": [
            {"path": "/root/images/first_frame.jpg", "frame_idx": 0, "strength": 0.8}
        ]
    }' | python3 -c "import sys,json; print(json.load(sys.stdin)['task_id'])")

# Poll
while true; do
    STATUS=$(curl -s http://localhost:8000/task/$TASK_ID | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
    echo "Status: $STATUS"
    [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ] && break
    sleep 5
done

# Download
curl -o output.mp4 http://localhost:8000/task/$TASK_ID/video
```

### Python client

```python
import json, sys, time
import requests

BASE = "http://<SERVER_IP>:8000"

# Submit
resp = requests.post(f"{BASE}/txt2vid", json={
    "prompt": "A majestic eagle soaring over snow-capped mountain peaks",
    "seed": 42,
    "height": 1088,
    "width": 1920,
    "num_frames": 121,
    "frame_rate": 24.0,
}, timeout=10)
resp.raise_for_status()
task = resp.json()
task_id = task["task_id"]
print(f"Task {task_id} submitted: {task['status']}")

# Poll
while True:
    status = requests.get(f"{BASE}/task/{task_id}", timeout=10).json()
    print(f"  {status['status']}  elapsed={status.get('elapsed')}")
    if status["status"] in ("completed", "failed"):
        break
    time.sleep(3)

if status["status"] == "completed":
    video = requests.get(f"{BASE}/task/{task_id}/video", timeout=60)
    with open("generated.mp4", "wb") as f:
        f.write(video.content)
    print(f"Video saved ({len(video.content)} bytes)")
elif status["status"] == "failed":
    print(f"Task failed: {status.get('error')}", file=sys.stderr)
    sys.exit(1)
```

---

## Performance Tuning (RTX Pro 6000)

### Optimal Configuration

| Setting | Value | Benefit |
|---------|-------|---------|
| `--quantization fp8-cast` | FP8 weights | ~40% less VRAM, marginally faster |
| `--offload none` | All weights on GPU | No streaming overhead |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Mandatory for FP8 |

With 96 GB VRAM, the bf16 model fits entirely without offloading. fp8-cast reduces VRAM to ~30 GB, leaving room for larger batches or concurrent requests.

### Benchmark expectations (RTX Pro 6000)

| Resolution | Frames | Steps | FP8-cast | bf16 |
|------------|--------|-------|----------|------|
| 1088 × 1920 | 121 | 8 + 3 | ~40 sec | ~45 sec |
| 1088 × 1920 | 97  | 8 + 3 | ~35 sec | ~38 sec |
| 704 × 1280  | 121 | 8 + 3 | ~22 sec | ~25 sec |

Times include prompt encoding, two-stage denoising, VAE decoding, and mp4 encoding. Measured on RTX Pro 6000, Ubuntu 24.04, CUDA 12.9, PyTorch 2.7.

---

## Production Deployment

### systemd Service

Create `/etc/systemd/system/ltx-server.service`:

```ini
[Unit]
Description=LTX-2 Video Generation Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/LTX-2
Environment="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
Environment="DISABLE_FLASH_ATTN=1"
Environment="XFORMERS_DISABLED=1"
ExecStart=/root/LTX-2/.venv/bin/python -m ltx_pipelines.server \
    --distilled-checkpoint-path /root/models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors \
    --gemma-root /root/models/gemma \
    --spatial-upsampler-path /root/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    --quantization fp8-cast \
    --host 0.0.0.0 \
    --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable ltx-server
sudo systemctl start ltx-server
sudo systemctl status ltx-server
sudo journalctl -u ltx-server -f  # Follow logs
```

### Nginx Reverse Proxy (TLS termination)

```nginx
server {
    listen 443 ssl;
    server_name ltx-api.example.com;

    ssl_certificate     /etc/nginx/certs/fullchain.pem;
    ssl_certificate_key /etc/nginx/certs/privkey.pem;

    client_max_body_size 256m;
    proxy_read_timeout 600s;   # Allow up to 10 min for generation

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;        # Stream video response
        proxy_request_buffering off;
    }
}
```

### Firewall

```bash
# Allow API port
sudo ufw allow 8000/tcp
# Or restrict to specific IPs
sudo ufw allow from YOUR_STATIC_IP to any port 8000 proto tcp
```

---

## Troubleshooting

### VRAM / OOM errors
- Enable `--quantization fp8-cast`
- Confirm `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set
- Reduce resolution or frame count in the API request

### CUDA errors or slow first generation
- The first call after server start triggers CUDA kernel compilation. Subsequent calls are faster.
- Verify CUDA driver is version 570+: `nvidia-smi`
- Verify CUDA toolkit: `nvcc --version`

### PyAV encoding failures
- Install ffmpeg: `sudo apt install -y ffmpeg`
- Verify: `ffmpeg -version`

### Server fails to bind port
- Check port is free: `ss -tlnp | grep 8000`
- Change port: `--port 8080`

### HuggingFace downloads
- Use `HF_HUB_ENABLE_HF_TRANSFER=1` for faster downloads
- Set `HF_TOKEN` if downloading gated models
