"""
Chunked Data Manager — auto-splits large NDJSON files, caches metadata.
Owner: S
"""
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Iterator, List, Optional

logger = logging.getLogger("macos_ueba.chunk_manager")

class ChunkedDataManager:
    def __init__(self, data_dir: Path, max_bytes: int = 500 * 1024 * 1024):
        self.data_dir = data_dir
        self.max_bytes = max_bytes
        self.manifest_path = self.data_dir / "manifest.json"
        self.manifest: Dict = {"chunks": [], "ruid_counts": {}}
        
        if self.manifest_path.exists():
            with open(self.manifest_path, "r") as f:
                self.manifest = json.load(f)
                
    def _save_manifest(self):
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def ingest(self, source_file: Path):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ingesting {source_file} into {self.data_dir}")
        
        chunk_idx = len(self.manifest["chunks"])
        current_chunk_path = self.data_dir / f"chunk_{chunk_idx:04d}.ndjson"
        current_f = open(current_chunk_path, "w")
        current_bytes = 0
        
        with open(source_file, "r") as src:
            for line in src:
                # Basic ruid counting if possible
                try:
                    data = json.loads(line)
                    ruid = data.get("process", {}).get("audit_token", {}).get("ruid", -1)
                    if ruid != -1:
                        ruid_str = str(ruid)
                        self.manifest["ruid_counts"][ruid_str] = self.manifest["ruid_counts"].get(ruid_str, 0) + 1
                except:
                    pass
                
                line_len = len(line.encode("utf-8"))
                if current_bytes + line_len > self.max_bytes:
                    current_f.close()
                    self.manifest["chunks"].append(str(current_chunk_path.name))
                    chunk_idx += 1
                    current_chunk_path = self.data_dir / f"chunk_{chunk_idx:04d}.ndjson"
                    current_f = open(current_chunk_path, "w")
                    current_bytes = 0
                
                current_f.write(line)
                current_bytes += line_len
                
        if current_bytes > 0:
            self.manifest["chunks"].append(str(current_chunk_path.name))
        current_f.close()
        self._save_manifest()
        
    def summary(self) -> str:
        return f"Chunks: {len(self.manifest['chunks'])}, RUID counts: {self.manifest['ruid_counts']}"
        
    def get_event_count(self, ruid: Optional[int]) -> Optional[int]:
        if not ruid:
            return sum(self.manifest["ruid_counts"].values()) if self.manifest["ruid_counts"] else None
        return self.manifest["ruid_counts"].get(str(ruid))
        
    def replay_all(self) -> Iterator[Dict]:
        for chunk_name in self.manifest["chunks"]:
            chunk_path = self.data_dir / chunk_name
            if chunk_path.exists():
                with open(chunk_path, "r") as f:
                    for line in f:
                        if line.strip():
                            yield json.loads(line)

    def ensure_local(self, local_dir: Path) -> "ChunkedDataManager":
        """Copy chunks to a faster local SSD if needed."""
        local_dir.mkdir(parents=True, exist_ok=True)
        local_mgr = ChunkedDataManager(local_dir, self.max_bytes)
        
        # Copy manifest
        if self.manifest_path.exists():
            shutil.copy2(self.manifest_path, local_mgr.manifest_path)
            local_mgr.manifest = self.manifest
            
        # Copy chunks
        for chunk_name in self.manifest["chunks"]:
            src = self.data_dir / chunk_name
            dst = local_dir / chunk_name
            if not dst.exists() and src.exists():
                shutil.copy2(src, dst)
                
        return local_mgr
