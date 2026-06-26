"""LTX-2 Inference Server - Main Application."""

import asyncio
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add server directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Settings, get_settings
from routers import generation, health
from services.job_queue import JobQueue
from services.pipeline_service import PipelineService
from utils.exceptions import (
    AppException,
    app_exception_handler,
    general_exception_handler,
)
from utils.file_utils import cleanup_old_files, ensure_directories

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    settings = get_settings()

    # Ensure directories exist
    ensure_directories(settings)

    # Initialize services
    logger.info("Initializing LTX-2 Inference Server...")
    app.state.settings = settings

    # Initialize job queue
    app.state.job_queue = JobQueue(max_concurrent=settings.max_concurrent_jobs)

    # Initialize pipeline service
    pipeline = PipelineService(settings)
    app.state.pipeline = pipeline

    # Load model
    try:
        pipeline.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Start cleanup task
    cleanup_task = asyncio.create_task(_periodic_cleanup(settings))

    logger.info("Server initialized successfully")
    logger.info(f"Server will run on {settings.host}:{settings.port}")

    yield

    # Shutdown
    logger.info("Shutting down server...")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    pipeline.unload_model()
    logger.info("Server shutdown complete")


async def _periodic_cleanup(settings: Settings) -> None:
    """Periodically clean up old files."""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            cleanup_old_files(settings.output_dir, settings.cleanup_after_hours)
            cleanup_old_files(settings.temp_dir, settings.cleanup_after_hours)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


def create_app() -> FastAPI:
    """Create FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="LTX-2 Inference Server",
        description="Professional video generation API using LTX-2 Distilled Pipeline",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    # Include routers
    app.include_router(health.router)
    app.include_router(generation.router)

    # Root endpoint
    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "LTX-2 Inference Server",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health",
            "generate": "/api/v1/generate",
        }

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )
