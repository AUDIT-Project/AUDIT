import json
from pathlib import Path
from src.ingest.chunked_data_manager import ChunkedDataManager

def test_chunked_data_manager(tmp_path):
    data_dir = tmp_path / "data"
    source = tmp_path / "source.ndjson"
    
    # Create dummy data
    with open(source, "w") as f:
        for i in range(100):
            f.write(json.dumps({"process": {"audit_token": {"ruid": 501}}, "event_type": "test"}) + "\n")
            
    # Set max_bytes very low to force chunking
    mgr = ChunkedDataManager(data_dir, max_bytes=1000)
    mgr.ingest(source)
    
    assert data_dir.exists()
    assert (data_dir / "manifest.json").exists()
    assert len(mgr.manifest["chunks"]) > 1
    assert mgr.get_event_count(501) == 100
    
    replayed = list(mgr.replay_all())
    assert len(replayed) == 100
