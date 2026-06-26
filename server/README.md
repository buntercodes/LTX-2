# LTX-2 Inference Server

Professional-grade FastAPI inference server for LTX-2 video generation using the Distilled Pipeline.

## Features

- **Async Job Queue**: Submit multiple generation requests, processed concurrently
- **RESTful API**: Clean, well-documented endpoints
- **Health Monitoring**: Real-time GPU and job queue status
- **File Management**: Automatic cleanup of old outputs
- **Configurable**: Environment-based configuration
- **Docker Ready**: Containerized deployment support
- **BF16 Precision**: Model weights loaded in bfloat16 for optimal memory/quality balance

## Quick Start

### 1. Installation

```bash
# From the repository root
cd LTX-2

# Sync base LTX-2 dependencies (creates .venv)
uv sync --frozen

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install server-specific dependencies
uv pip install -r server/requirements.txt
```

### 2. Configuration

Copy the example environment file and update paths:

```bash
cp .env.example .env
```

Edit `.env` with your model paths:

```env
LTX_CHECKPOINT_PATH=/path/to/ltx-2.3-22b-distilled-1.1.safetensors
LTX_GEMMA_ROOT=/path/to/gemma-3-12b-it-qat-q4_0-unquantized
LTX_SPATIAL_UPSAMPLER_PATH=/path/to/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
```

### 3. Run Server

```bash
cd server

# Development
python main.py

# Or with uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Docker (Optional)

```bash
docker build -t ltx2-server .
docker run -p 8000:8000 --gpus all ltx2-server
```

## API Endpoints

### Health & Info

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/api/v1/health` | Health check with GPU stats |
| GET | `/api/v1/model/info` | Model configuration |

### Generation

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/generate` | Submit video generation job |
| GET | `/api/v1/jobs` | List all jobs |
| GET | `/api/v1/jobs/{job_id}` | Get job status |
| GET | `/api/v1/download/{job_id}` | Download generated video |
| DELETE | `/api/v1/jobs/{job_id}` | Cancel/delete job |

### Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Usage Examples

### Submit a Generation Job

```bash
curl -X POST "http://localhost:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene landscape with mountains in the background, golden hour lighting",
    "height": 512,
    "width": 768,
    "num_frames": 121,
    "frame_rate": 24.0,
    "seed": 42
  }'
```

Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Job queued successfully",
  "created_at": "2026-06-26T12:00:00",
  "estimated_seconds": 12.1
}
```

### Check Job Status

```bash
curl "http://localhost:8000/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000"
```

Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 100.0,
  "created_at": "2026-06-26T12:00:00",
  "started_at": "2026-06-26T12:00:01",
  "completed_at": "2026-06-26T12:00:15",
  "result_url": "/api/v1/download/550e8400-e29b-41d4-a716-446655440000",
  "metadata": {
    "generation_time_seconds": 14.2,
    "file_size_mb": 12.5
  }
}
```

### Download Video

```bash
curl -o output.mp4 "http://localhost:8000/api/v1/download/550e8400-e29b-41d4-a716-446655440000"
```

### Python Client

```python
import httpx
import time

# Submit job
response = httpx.post(
    "http://localhost:8000/api/v1/generate",
    json={
        "prompt": "A beautiful sunset over the ocean",
        "height": 512,
        "width": 768,
        "num_frames": 121,
    }
)
job = response.json()
job_id = job["job_id"]

# Poll until completed
while True:
    response = httpx.get(f"http://localhost:8000/api/v1/jobs/{job_id}")
    status = response.json()
    if status["status"] == "completed":
        break
    elif status["status"] == "failed":
        raise Exception(f"Job failed: {status['error']}")
    time.sleep(1)

# Download video
response = httpx.get(f"http://localhost:8000/api/v1/download/{job_id}")
with open("output.mp4", "wb") as f:
    f.write(response.content)
```

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `LTX_HOST` | `0.0.0.0` | Server host |
| `LTX_PORT` | `8000` | Server port |
| `LTX_WORKERS` | `1` | Number of workers |
| `LTX_CHECKPOINT_PATH` | *required* | Model checkpoint path |
| `LTX_GEMMA_ROOT` | *required* | Gemma encoder path |
| `LTX_SPATIAL_UPSAMPLER_PATH` | *required* | Upsampler path |
| `LTX_DEFAULT_HEIGHT` | `512` | Default video height |
| `LTX_DEFAULT_WIDTH` | `768` | Default video width |
| `LTX_DEFAULT_NUM_FRAMES` | `121` | Default frame count |
| `LTX_DEFAULT_FRAME_RATE` | `24.0` | Default FPS |
| `LTX_MAX_CONCURRENT_JOBS` | `2` | Max parallel jobs |
| `LTX_QUANTIZATION` | `None` | FP8 quantization |
| `LTX_ENABLE_COMPILE` | `false` | Enable torch.compile |
| `LTX_OFFLOAD_MODE` | `none` | Weight offloading |

## Memory Requirements

LTX-2 model weights are in **BF16 (bfloat16)** precision:

| Configuration | GPU Memory | System RAM |
|--------------|------------|------------|
| Default (BF16) | ~45GB | ~64GB |
| With FP8 quantization | ~25GB | ~32GB |
| With CPU offloading | ~8GB | ~80GB |

## Architecture

```
server/
├── main.py              # FastAPI application
├── config.py            # Configuration management
├── models/
│   └── schemas.py       # Pydantic models
├── routers/
│   ├── generation.py    # Generation endpoints
│   └── health.py        # Health endpoints
├── services/
│   ├── pipeline_service.py  # LTX-2 pipeline wrapper
│   └── job_queue.py     # Async job queue
└── utils/
    ├── file_utils.py    # File management
    └── exceptions.py    # Error handling
```

## License

See LICENSE file in repository root.
