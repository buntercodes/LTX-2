import logging
import os
import tempfile
import threading
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import ClassVar

os.environ.setdefault("DISABLE_FLASH_ATTN", "1")
os.environ.setdefault("XFORMERS_DISABLED", "1")

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, model_validator
from starlette.background import BackgroundTask

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import DEFAULT_IMAGE_CRF
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import OffloadMode

logger = logging.getLogger("ltx_server")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ImageConditioning(BaseModel):
    path: str = Field(..., description="Path to conditioning image on the server filesystem")
    frame_idx: int = Field(..., description="Frame index to apply conditioning at")
    strength: float = Field(..., description="Conditioning strength (0.0-1.0)")
    crf: int = Field(DEFAULT_IMAGE_CRF, description="JPEG CRF for image preprocessing")


class GenerateRequestBase(BaseModel):
    prompt: str = Field(..., description="Text prompt describing the desired video")
    seed: int = Field(10, ge=0, description="Random seed for reproducible generation")
    height: int = Field(1024, ge=64, description="Output height in pixels (divisible by 64)")
    width: int = Field(1536, ge=64, description="Output width in pixels (divisible by 64)")
    num_frames: int = Field(121, ge=1, description="Number of frames (must be k*8+1)")
    frame_rate: float = Field(24.0, gt=0, description="Output frame rate")
    enhance_prompt: bool = Field(False, description="Enable automatic prompt enhancement")

    @model_validator(mode="after")
    def _validate_dimensions(self) -> "GenerateRequestBase":
        if self.height % 64 != 0:
            raise ValueError(f"height must be divisible by 64, got {self.height}")
        if self.width % 64 != 0:
            raise ValueError(f"width must be divisible by 64, got {self.width}")
        if (self.num_frames - 1) % 8 != 0:
            raise ValueError(f"num_frames must satisfy (frames - 1) % 8 == 0, got {self.num_frames}")
        return self


class Txt2VidRequest(GenerateRequestBase):
    pass


class Img2VidRequest(GenerateRequestBase):
    images: list[ImageConditioning] = Field(..., min_length=1, description="One or more conditioning images")


class SubmitResponse(BaseModel):
    task_id: str
    status: str
    message: str


class ServerConfig(BaseModel):
    distilled_checkpoint_path: str
    gemma_root: str
    spatial_upsampler_path: str
    quantization: str | None = None
    offload_mode: OffloadMode = OffloadMode.NONE
    torch_compile: bool = False
    loras: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_quantization(policy_name: str | None) -> QuantizationPolicy | None:
    if policy_name is None:
        return None
    if policy_name == "fp8-cast":
        return QuantizationPolicy.fp8_cast()
    if policy_name == "fp8-scaled-mm":
        return QuantizationPolicy.fp8_scaled_mm()
    raise ValueError(f"Unknown quantization policy: {policy_name}")


def _build_loras(loras: list[str]) -> list[LoraPathStrengthAndSDOps]:
    result = []
    for spec in loras:
        parts = spec.split(maxsplit=1)
        path = str(Path(parts[0]).expanduser().resolve())
        strength = float(parts[1]) if len(parts) > 1 else 1.0
        result.append(LoraPathStrengthAndSDOps(path, strength, LTXV_LORA_COMFY_RENAMING_MAP))
    return result


# ---------------------------------------------------------------------------
# Pipeline singleton
# ---------------------------------------------------------------------------


class PipelineManager:
    _instance: "PipelineManager | None" = None

    def __init__(self) -> None:
        self._pipeline: DistilledPipeline | None = None
        self._config: ServerConfig | None = None

    @classmethod
    def get_instance(cls) -> "PipelineManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self, config: ServerConfig) -> None:
        if self._pipeline is not None:
            return
        self._config = config
        logger.info("Loading DistilledPipeline ...")
        self._pipeline = DistilledPipeline(
            distilled_checkpoint_path=config.distilled_checkpoint_path,
            gemma_root=config.gemma_root,
            spatial_upsampler_path=config.spatial_upsampler_path,
            loras=_build_loras(config.loras),
            quantization=_build_quantization(config.quantization),
            torch_compile=config.torch_compile,
            offload_mode=config.offload_mode,
        )
        logger.info("Pipeline loaded.")

    @property
    def pipeline(self) -> DistilledPipeline:
        if self._pipeline is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        return self._pipeline


# ---------------------------------------------------------------------------
# Task system
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    task_id: str
    status: TaskStatus = TaskStatus.QUEUED
    req: GenerateRequestBase | None = None
    images: list[ImageConditioningInput] = field(default_factory=list)
    video_path: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    elapsed: float | None = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "prompt": self.req.prompt if self.req else None,
            "seed": self.req.seed if self.req else None,
            "height": self.req.height if self.req else None,
            "width": self.req.width if self.req else None,
            "num_frames": self.req.num_frames if self.req else None,
            "frame_rate": self.req.frame_rate if self.req else None,
            "image_count": len(self.images),
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed": self.elapsed,
        }


class TaskManager:
    MAX_QUEUE_SIZE: ClassVar[int] = 100
    MAX_HISTORY: ClassVar[int] = 200

    def __init__(self, pipeline: DistilledPipeline) -> None:
        self._pipeline = pipeline
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._queue: deque[Task] = deque()
        self._tasks: dict[str, Task] = {}
        self._output_dir = Path(tempfile.mkdtemp(prefix="ltx_tasks_"))
        self._running = True
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        logger.info("Task output dir: %s", self._output_dir)

    # -- submission ----------------------------------------------------------

    def submit(self, req: GenerateRequestBase, images: list[ImageConditioningInput]) -> str:
        task_id = uuid.uuid4().hex
        task = Task(task_id=task_id, req=req, images=images)
        with self._lock:
            if len(self._tasks) >= self.MAX_HISTORY:
                oldest = min(self._tasks.values(), key=lambda t: t.created_at)
                self._cleanup_task(oldest, remove_from_dict=True)
            self._tasks[task_id] = task
            if len(self._queue) >= self.MAX_QUEUE_SIZE:
                raise HTTPException(
                    status_code=429,
                    detail="Too many queued tasks. Try again later.",
                )
            self._queue.append(task)
            self._condition.notify()
        logger.info("Task %s queued (queue depth %d)", task_id, len(self._queue))
        return task_id

    # -- query ---------------------------------------------------------------

    def get(self, task_id: str) -> Task:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
            return task

    def list_all(self) -> list[dict]:
        with self._lock:
            return [t.to_dict() for t in sorted(self._tasks.values(), key=lambda t: t.created_at)]

    # -- worker --------------------------------------------------------------

    def _worker_loop(self) -> None:
        while self._running:
            with self._lock:
                while not self._queue and self._running:
                    self._condition.wait(timeout=5)
                if not self._running:
                    return
                task = self._queue.popleft()

            self._run_task(task)

    def _run_task(self, task: Task) -> None:
        req = task.req
        assert req is not None

        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        logger.info("Task %s started (prompt='%s...')", task.task_id, req.prompt[:80])

        tiling_config = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(req.num_frames, tiling_config)

        try:
            with torch.inference_mode():
                video_iter, audio = self._pipeline(
                    prompt=req.prompt,
                    seed=req.seed,
                    height=req.height,
                    width=req.width,
                    num_frames=req.num_frames,
                    frame_rate=req.frame_rate,
                    images=task.images,
                    tiling_config=tiling_config,
                    enhance_prompt=req.enhance_prompt,
                )

            video_path = str(self._output_dir / f"{task.task_id}.mp4")

            with torch.inference_mode():
                encode_video(
                    video=video_iter,
                    fps=req.frame_rate,
                    audio=audio,
                    output_path=video_path,
                    video_chunks_number=video_chunks_number,
                )

            task.status = TaskStatus.COMPLETED
            task.video_path = video_path
            task.finished_at = time.time()
            task.elapsed = round(task.finished_at - task.started_at, 1)
            logger.info("Task %s completed in %.1fs", task.task_id, task.elapsed)

        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.error = str(exc)
            task.finished_at = time.time()
            task.elapsed = round(task.finished_at - task.started_at, 1)
            logger.exception("Task %s failed in %.1fs", task.task_id, task.elapsed)

    # -- cleanup -------------------------------------------------------------

    def _cleanup_task(self, task: Task, *, remove_from_dict: bool = False) -> None:
        if task.video_path is not None:
            Path(task.video_path).unlink(missing_ok=True)
            task.video_path = None
        if remove_from_dict:
            self._tasks.pop(task.task_id, None)

    def shutdown(self) -> None:
        self._running = False
        with self._lock:
            self._condition.notify_all()
        self._worker.join(timeout=30)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


def create_app(config: ServerConfig) -> FastAPI:
    pipeline_mgr = PipelineManager.get_instance()
    task_mgr: TaskManager | None = None

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        nonlocal task_mgr
        pipeline_mgr.initialize(config)
        task_mgr = TaskManager(pipeline_mgr.pipeline)
        logger.info("Server ready.")
        yield
        if task_mgr is not None:
            task_mgr.shutdown()
        logger.info("Server shutdown complete.")

    app = FastAPI(title="LTX-2 Video Generation API", version="2.0.0", lifespan=lifespan)

    # -- Health --------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "gpu_available": torch.cuda.is_available(),
            "queue_depth": len(task_mgr._queue) if task_mgr else 0,
        }

    # -- Submit --------------------------------------------------------------

    def _to_image_inputs(imgs: list[ImageConditioning]) -> list[ImageConditioningInput]:
        return [
            ImageConditioningInput(path=i.path, frame_idx=i.frame_idx, strength=i.strength, crf=i.crf)
            for i in imgs
        ]

    @app.post("/txt2vid", response_model=SubmitResponse, status_code=202)
    async def txt2vid(req: Txt2VidRequest) -> dict:
        if task_mgr is None:
            raise HTTPException(503, "Server not ready")
        task_id = task_mgr.submit(req, images=[])
        return {"task_id": task_id, "status": "queued", "message": "Task submitted"}

    @app.post("/img2vid", response_model=SubmitResponse, status_code=202)
    async def img2vid(req: Img2VidRequest) -> dict:
        if task_mgr is None:
            raise HTTPException(503, "Server not ready")
        task_id = task_mgr.submit(req, images=_to_image_inputs(req.images))
        return {"task_id": task_id, "status": "queued", "message": "Task submitted"}

    # -- Query ---------------------------------------------------------------

    @app.get("/task/{task_id}")
    async def task_status(task_id: str) -> dict:
        if task_mgr is None:
            raise HTTPException(503, "Server not ready")
        return task_mgr.get(task_id).to_dict()

    @app.get("/task/{task_id}/video")
    async def task_video(task_id: str) -> FileResponse:
        if task_mgr is None:
            raise HTTPException(503, "Server not ready")
        task = task_mgr.get(task_id)
        if task.status != TaskStatus.COMPLETED or task.video_path is None:
            status_code = 404 if task.status != TaskStatus.COMPLETED else 425
            detail = "Video not ready" if task.status != TaskStatus.COMPLETED else "Video file missing"
            if task.status == TaskStatus.FAILED:
                detail = f"Task failed: {task.error}"
            raise HTTPException(status_code, detail=detail)
        return FileResponse(
            task.video_path,
            media_type="video/mp4",
            filename=f"generated_{task_id[:8]}.mp4",
        )

    @app.get("/queue")
    async def queue_list() -> list[dict]:
        if task_mgr is None:
            raise HTTPException(503, "Server not ready")
        return task_mgr.list_all()

    def _shutdown() -> None:
        if task_mgr is not None:
            task_mgr.shutdown()

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_server(config: ServerConfig, host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    app = create_app(config)
    uvicorn.run(app, host=host, port=port)


def _banner() -> None:
    print(
        "\n"
        "  ╔══════════════════════════════════════════════════════╗\n"
        "  ║           LTX-2 Video Generation API v2             ║\n"
        "  ╚══════════════════════════════════════════════════════╝\n"
    )


def main() -> None:
    import argparse

    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="LTX-2 Video Generation API Server")
    parser.add_argument("--distilled-checkpoint-path", type=str, required=True, help="Path to distilled checkpoint .safetensors")
    parser.add_argument("--gemma-root", type=str, required=True, help="Path to Gemma text encoder directory")
    parser.add_argument("--spatial-upsampler-path", type=str, required=True, help="Path to spatial upsampler .safetensors")
    parser.add_argument("--quantization", type=str, default=None, choices=["fp8-cast", "fp8-scaled-mm"], help="Quantization policy")
    parser.add_argument("--offload", type=str, default="none", choices=["none", "cpu", "disk"], help="Weight offloading strategy")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--lora", type=str, action="append", default=[], help="LoRA adapter: PATH [STRENGTH]. Repeat for multiple.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")

    args = parser.parse_args()

    offload_map = {"none": OffloadMode.NONE, "cpu": OffloadMode.CPU, "disk": OffloadMode.DISK}

    config = ServerConfig(
        distilled_checkpoint_path=args.distilled_checkpoint_path,
        gemma_root=args.gemma_root,
        spatial_upsampler_path=args.spatial_upsampler_path,
        quantization=args.quantization,
        offload_mode=offload_map[args.offload],
        torch_compile=args.compile,
        loras=args.lora,
    )

    _banner()
    run_server(config, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
