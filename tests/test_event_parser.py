import pytest
from src.preprocessing.event_parser import parse_event, ESEvent

def test_parse_exec_event():
    raw = {
        "event_type": "exec",
        "time": "1710000000.0",
        "schema_version": 1,
        "version": 7,
        "process": {
            "executable": {"path": "/usr/bin/python3"},
            "audit_token": {"pid": 1234, "ppid": 1, "ruid": 501, "euid": 501},
            "signing_id": "com.apple.python3",
        },
        "event": {
            "target": {
                "executable": {"path": "/usr/local/bin/myapp"},
                "audit_token": {"pid": 1235},
            }
        },
    }
    ev = parse_event(raw)
    assert ev is not None
    assert ev.event_type == "exec"
    assert ev.ruid == 501
    assert ev.process_path == "/usr/bin/python3"
    assert ev.target_path == "/usr/local/bin/myapp"
    assert ev.target_pid == 1235

def test_parse_file_event():
    raw = {
        "event_type": "open",
        "time": "1710000001.0",
        "process": {
            "executable": {"path": "/usr/bin/vim"},
            "audit_token": {"pid": 100, "ppid": 99, "ruid": 501, "euid": 501},
        },
        "event": {
            "file": {"path": "/Users/test/document.txt"}
        },
    }
    ev = parse_event(raw)
    assert ev is not None
    assert ev.event_type == "open"
    assert ev.target_path == "/Users/test/document.txt"

def test_parse_malformed_returns_none():
    assert parse_event({}) is not None
    assert parse_event({"process": "not_a_dict"}) is not None
    assert parse_event(None) is None

def test_parse_missing_timestamp_uses_fallback():
    raw = {"event_type": "fork", "process": {"audit_token": {}}}
    ev = parse_event(raw)
    assert ev is not None
    assert ev.timestamp > 0

def test_no_cdhash_field():
    ev = ESEvent(event_type="test", timestamp=1.0)
    assert not hasattr(ev, "cdhash")
