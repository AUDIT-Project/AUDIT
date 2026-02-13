"""
I/O acceleration utilities for Colab / Drive environments.
Owner: S
"""
import logging
import psutil
import shutil
import torch
from pathlib import Path

logger = logging.getLogger("macos_ueba.io")

def _ensure_local_copy(drive_path: Path, local_path: Path) -> Path:
    """Copy file from Drive to local SSD (skip if exists)"""
    if local_path.exists():
        logger.info(f"Local copy exists: {local_path}")
        return local_path
        
    logger.info(f"Copying to local SSD: {drive_path} -> {local_path}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(drive_path, local_path)
    return local_path

def _ensure_local_memmap(drive_memmap_dir: Path, local_memmap_dir: Path) -> Path:
    """Copy memmap dir to local SSD (skip if valid)"""
    if (local_memmap_dir / "shape.txt").exists():
        return local_memmap_dir
        
    logger.info("Copying memmap cache to local SSD...")
    local_memmap_dir.mkdir(parents=True, exist_ok=True)
    if drive_memmap_dir.exists():
        for item in drive_memmap_dir.iterdir():
            shutil.copy2(item, local_memmap_dir / item.name)
            
    return local_memmap_dir

def _log_device_memory():
    """Log GPU/MPS/RSS memory usage"""
    process = psutil.Process()
    rss_mb = process.memory_info().rss / 1024 / 1024
    
    msg = f"RSS Mem: {rss_mb:.1f} MB"
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        msg += f" | CUDA alloc: {allocated:.1f} MB | CUDA res: {reserved:.1f} MB"
    elif torch.backends.mps.is_available():
        # MPS doesn't expose memory APIs robustly in PyTorch 2.1
        msg += " | GPU: MPS"
        
    logger.info(msg)

def replay_from_file(path: Path):
    """Stream NDJSON with progress logging"""
    import json
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
