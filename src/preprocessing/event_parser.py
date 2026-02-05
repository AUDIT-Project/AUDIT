"""
Event Parser — converts raw eslogger JSON into typed ESEvent records.

Owner: S
Consumers: vocabulary builder, memmap encoder, chunked data manager

Critical design decisions:
  - slots=True saves ~30% memory per instance
  - No raw_process/raw_event dicts stored (we extract what we need)
  - parse_event returns None on ANY error (never crashes the pipeline)
  - _FILE_EVENT_TYPES is a frozenset for O(1) lookup in hot loop

Known bugs fixed in this version:
  - cdhash field removed from dataclass (was causing TypeError)
  - datetime import at module level (not inside hot loop)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger("macos_ueba.parser")

@dataclass(slots=True)
class ESEvent:
    """
    Canonical representation of a single Endpoint Security event.
    """
    event_type: str
    timestamp: float
    schema_version: int = 0
    message_version: int = 0
    process_path: str = ""
    pid: int = 0
    ppid: int = 0
    ruid: int = -1
    euid: int = -1
    signing_id: str = ""
    team_id: str = ""
    target_path: str = ""
    target_pid: int = 0

def _safe_get(d: dict, *keys, default=""):
    """Navigate nested dicts safely."""
    cur = d
    for k in keys:
        if isinstance(cur, dict):
            cur = cur.get(k, default)
        else:
            return default
    return cur if cur is not None else default

_FILE_EVENT_TYPES = frozenset({
    "open", "create", "unlink", "rename", "write", "close",
    "truncate", "link", "copyfile", "exchangedata",
    "lookup", "stat", "readdir", "mmap",
})

def parse_event(raw: Dict[str, Any]) -> Optional[ESEvent]:
    """
    Parse one raw eslogger JSON dict -> ESEvent.
    Returns None on any structural error.
    """
    try:
        event_type = raw.get("event_type", "")
        if not event_type:
            event_type = _safe_get(raw, "event", "type", default="unknown")

        process = raw.get("process", {})
        audit = process.get("audit_token", {})
        event_data = raw.get("event", {})

        # Timestamp
        ts_str = raw.get("time", "")
        if ts_str:
            try:
                timestamp = float(ts_str)
            except ValueError:
                try:
                    dt = datetime.fromisoformat(
                        ts_str.replace("Z", "+00:00")
                    )
                    timestamp = dt.timestamp()
                except Exception:
                    timestamp = time.time()
        else:
            timestamp = float(raw.get("mach_time", time.time()))

        # Target path/pid
        target_path = ""
        target_pid = 0
        if event_type == "exec":
            target_path = _safe_get(
                event_data, "target", "executable", "path", default=""
            )
            target_pid = int(_safe_get(
                event_data, "target", "audit_token", "pid", default=0
            ))
        elif event_type in _FILE_EVENT_TYPES:
            target_path = _safe_get(
                event_data, "file", "path", default=""
            )
            if not target_path:
                target_path = _safe_get(
                    event_data, "target", "path", default=""
                )

        return ESEvent(
            event_type=str(event_type),
            timestamp=timestamp,
            schema_version=int(raw.get("schema_version", 0)),
            message_version=int(raw.get("version", 0)),
            process_path=str(
                _safe_get(process, "executable", "path", default="")
            ),
            pid=int(audit.get("pid", 0)),
            ppid=int(audit.get("ppid", 0)),
            ruid=int(audit.get("ruid", -1)),
            euid=int(audit.get("euid", -1)),
            signing_id=str(process.get("signing_id", "")),
            team_id=str(process.get("team_id", "")),
            target_path=str(target_path),
            target_pid=int(target_pid),
        )
    except Exception as e:
        logger.warning("Failed to parse event: %s", e)
        return None

def parse_stream(raw_events) -> Iterator[ESEvent]:
    """Generator: parse iterable of raw JSON dicts -> ESEvent stream."""
    for raw in raw_events:
        ev = parse_event(raw)
        if ev is not None:
            yield ev
