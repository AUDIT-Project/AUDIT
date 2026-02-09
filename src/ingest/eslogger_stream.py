"""
Live eslogger subprocess ingestion stream.
Owner: S
"""
import subprocess
import json
import logging
from typing import Iterator, Dict

logger = logging.getLogger("macos_ueba.stream")

class ESLoggerStream:
    def __init__(self, events: list[str], executable: str = "/usr/bin/eslogger"):
        self.events = events
        self.executable = executable
        self.process = None
        
    def start(self):
        cmd = [self.executable] + self.events
        logger.info(f"Starting eslogger: {' '.join(cmd)}")
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
    def stream(self) -> Iterator[Dict]:
        if not self.process:
            self.start()
            
        while True:
            line = self.process.stdout.readline()
            if not line:
                break
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass
                
    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
