import pytest
import numpy as np
from src.preprocessing.dataset import _compute_temporal_features
from src.preprocessing.event_parser import ESEvent

def test_compute_temporal_features_empty():
    res = _compute_temporal_features([])
    assert res.shape == (0, 4)
    
def test_compute_temporal_features():
    evs = [
        ESEvent(event_type="test", timestamp=100.0),
        ESEvent(event_type="test", timestamp=105.0) # dt = 5
    ]
    res = _compute_temporal_features(evs)
    assert res.shape == (2, 4)
    
    assert res[0, 3] == 0.0 # First dt is 0
    assert np.isclose(res[1, 3], np.log1p(5.0)) # dt is log1p(5)

def test_memmap_cache_ok(tmp_path):
    from src.preprocessing.dataset import MemmapEventStore
    assert not MemmapEventStore._cache_ok(tmp_path)
    with open(tmp_path / "shape.txt", "w") as f:
        f.write("100,24")
    assert MemmapEventStore._cache_ok(tmp_path)
