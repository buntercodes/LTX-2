"""Utility functions for file management and cleanup."""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

from config import Settings

logger = logging.getLogger(__name__)


def cleanup_old_files(directory: Path, max_age_hours: float = 24.0) -> int:
    """
    Delete files older than max_age_hours.

    Args:
        directory: Directory to clean
        max_age_hours: Maximum age in hours

    Returns:
        Number of files deleted
    """
    if not directory.exists():
        return 0

    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    deleted = 0

    for file_path in directory.iterdir():
        if file_path.is_file():
            try:
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mtime < cutoff:
                    file_path.unlink()
                    deleted += 1
                    logger.debug(f"Deleted old file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}: {e}")

    if deleted > 0:
        logger.info(f"Cleaned up {deleted} old files from {directory}")

    return deleted


def cleanup_directory(directory: Path, max_files: int = 100) -> int:
    """
    Keep only the most recent max_files in directory.

    Args:
        directory: Directory to clean
        max_files: Maximum number of files to keep

    Returns:
        Number of files deleted
    """
    if not directory.exists():
        return 0

    files = sorted(
        [f for f in directory.iterdir() if f.is_file()],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    deleted = 0
    for file_path in files[max_files:]:
        try:
            file_path.unlink()
            deleted += 1
            logger.debug(f"Deleted excess file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to delete {file_path}: {e}")

    if deleted > 0:
        logger.info(f"Cleaned up {deleted} excess files from {directory}")

    return deleted


def get_directory_size(directory: Path) -> int:
    """Get total size of directory in bytes."""
    if not directory.exists():
        return 0

    total = 0
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size

    return total


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def ensure_directories(settings: Settings) -> None:
    """Ensure all required directories exist."""
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {settings.output_dir}")
    logger.info(f"Temp directory: {settings.temp_dir}")
