"""
Memory-mapped event store and PyTorch dataset.
Owner: S
"""
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing.event_parser import ESEvent
from src.preprocessing.vocabulary import VocabularySet
from config import VocabularyConfig, ModelConfig

logger = logging.getLogger("macos_ueba.dataset")

def _compute_temporal_features(events: Union[np.ndarray, List[ESEvent]]) -> np.ndarray:
    """Compute temporal features for a list of events."""
    out = np.zeros((len(events), 4), dtype=np.float32)
    if not len(events):
        return out
        
    for i in range(len(events)):
        ev = events[i]
        # Allow passing ESEvent instances or previously encoded arrays
        if hasattr(ev, 'timestamp'):
            ts = ev.timestamp
        else:
            ts = ev[1] # fallback to raw array timestamp position
            
        hour = (ts / 3600) % 24
        out[i, 0] = hour / 24.0 # normalized hour
        out[i, 1] = math.sin(2 * math.pi * hour / 24.0)
        out[i, 2] = math.cos(2 * math.pi * hour / 24.0)
        
        # Log delta t
        if i == 0:
            out[i, 3] = 0.0
        else:
            prev_ts = events[i-1].timestamp if hasattr(events[i-1], 'timestamp') else events[i-1][1]
            dt = max(0.0, ts - prev_ts)
            out[i, 3] = math.log1p(dt)
            
    return out

class EventFeatureEncoder:
    def __init__(self, vocabs: VocabularySet, vocab_cfg: VocabularyConfig):
        self.vocabs = vocabs
        self.vocab_cfg = vocab_cfg
        
    def encode(self, ev: ESEvent) -> Dict[str, np.ndarray]:
        # event_type
        et_id = self.vocabs.event_type.encode(ev.event_type)
        
        # process path
        from src.preprocessing.vocabulary import tokenize_path
        p_tokens = tokenize_path(ev.process_path)[:self.vocab_cfg.max_path_length]
        p_ids = [self.vocabs.path_token.encode(t) for t in p_tokens]
        p_ids += [0] * (self.vocab_cfg.max_path_length - len(p_ids)) # PAD
        
        # tgt path
        t_tokens = tokenize_path(ev.target_path)[:self.vocab_cfg.max_path_length]
        t_ids = [self.vocabs.path_token.encode(t) for t in t_tokens]
        t_ids += [0] * (self.vocab_cfg.max_path_length - len(t_ids)) # PAD
        
        # signing id
        s_id = self.vocabs.signing_id.encode(ev.signing_id) if ev.signing_id else 0
        
        # numeric
        numeric = np.array([ev.pid, ev.ppid, ev.target_pid], dtype=np.float32)
        
        return {
            "event_type_id": np.array(et_id, dtype=np.int32),
            "proc_path_ids": np.array(p_ids, dtype=np.int32),
            "tgt_path_ids": np.array(t_ids, dtype=np.int32),
            "signing_id": np.array(s_id, dtype=np.int32),
            "numerical": numeric,
            "timestamp": np.array(ev.timestamp, dtype=np.float64)
        }

class MemmapEventStore:
    _FIELDS = ["event_type_id", "proc_path_ids", "tgt_path_ids", "signing_id", "numerical", "timestamp"]
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        with open(self.cache_dir / "shape.txt", "r") as f:
            shape_info = f.read().split(",")
            self.n_events = int(shape_info[0])
            self.max_path_len = int(shape_info[1])
            
        self.arrays = {
            "event_type_id": np.memmap(self.cache_dir / "event_type_id.npy", dtype=np.int32, mode="r", shape=(self.n_events,)),
            "proc_path_ids": np.memmap(self.cache_dir / "proc_path_ids.npy", dtype=np.int32, mode="r", shape=(self.n_events, self.max_path_len)),
            "tgt_path_ids": np.memmap(self.cache_dir / "tgt_path_ids.npy", dtype=np.int32, mode="r", shape=(self.n_events, self.max_path_len)),
            "signing_id": np.memmap(self.cache_dir / "signing_id.npy", dtype=np.int32, mode="r", shape=(self.n_events,)),
            "numerical": np.memmap(self.cache_dir / "numerical.npy", dtype=np.float32, mode="r", shape=(self.n_events, 3)),
            "timestamp": np.memmap(self.cache_dir / "timestamp.npy", dtype=np.float64, mode="r", shape=(self.n_events,)),
        }
    
    @classmethod
    def _cache_ok(cls, cache_dir: Path) -> bool:
        return (cache_dir / "shape.txt").exists()
        
    @classmethod
    def build(cls, data_source, vocabs: VocabularySet, vocab_cfg: VocabularyConfig, 
              target_ruid: Optional[int], cache_dir: Path, n_events_hint: int) -> 'MemmapEventStore':
        """Build memmap dataset from a fast data source."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        encoder = EventFeatureEncoder(vocabs, vocab_cfg)
        
        arrays = {
            "event_type_id": np.memmap(cache_dir / "event_type_id.npy", dtype=np.int32, mode="w+", shape=(n_events_hint,)),
            "proc_path_ids": np.memmap(cache_dir / "proc_path_ids.npy", dtype=np.int32, mode="w+", shape=(n_events_hint, vocab_cfg.max_path_length)),
            "tgt_path_ids": np.memmap(cache_dir / "tgt_path_ids.npy", dtype=np.int32, mode="w+", shape=(n_events_hint, vocab_cfg.max_path_length)),
            "signing_id": np.memmap(cache_dir / "signing_id.npy", dtype=np.int32, mode="w+", shape=(n_events_hint,)),
            "numerical": np.memmap(cache_dir / "numerical.npy", dtype=np.float32, mode="w+", shape=(n_events_hint, 3)),
            "timestamp": np.memmap(cache_dir / "timestamp.npy", dtype=np.float64, mode="w+", shape=(n_events_hint,)),
        }
        
        from src.preprocessing.event_parser import parse_event
        idx = 0
        for raw in data_source.replay_all():
            ev = parse_event(raw)
            if ev is None:
                continue
            if target_ruid is not None and ev.ruid != target_ruid:
                continue
                
            enc = encoder.encode(ev)
            for k in arrays:
                arrays[k][idx] = enc[k]
                
            idx += 1
            if idx >= n_events_hint:
                break
                
        # Verification to catch ghost zero-events
        assert idx == n_events_hint, f"Dataset hint {n_events_hint} mismatched actual {idx}"
                
        for arr in arrays.values():
            arr.flush()
            
        with open(cache_dir / "shape.txt", "w") as f:
            f.write(f"{n_events_hint},{vocab_cfg.max_path_length}")
            
        return cls(cache_dir)

class UserSequenceDataset(Dataset):
    """Sliding window sequence dataset."""
    def __init__(self, store: MemmapEventStore, config: ModelConfig):
        self.store = store
        self.seq_len = config.max_seq_len
        self.stride = config.stride
        self.n_windows = max(0, (store.n_events - self.seq_len) // self.stride + 1)
        
    def __len__(self) -> int:
        return self.n_windows
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.stride
        end = start + self.seq_len
        
        # Arrays returned directly from Memmap to torch tensor
        # No extra memory cloning due to `torch.from_numpy().clone()` unless specified
        batch = {
            "event_type_ids": torch.from_numpy(self.store.arrays["event_type_id"][start:end].astype(np.int64)),
            "proc_path_ids": torch.from_numpy(self.store.arrays["proc_path_ids"][start:end].astype(np.int64)),
            "tgt_path_ids": torch.from_numpy(self.store.arrays["tgt_path_ids"][start:end].astype(np.int64)),
            "signing_ids": torch.from_numpy(self.store.arrays["signing_id"][start:end].astype(np.int64)),
            "numerical": torch.from_numpy(self.store.arrays["numerical"][start:end].astype(np.float32)),
        }
        
        # Compute temporal features dynamically for the window
        # Mocking to objects for the static function signature logic.
        class MockEv:
            def __init__(self, ts):
                self.timestamp = ts
        
        ts_array = self.store.arrays["timestamp"][start:end]
        ev_m = [MockEv(t) for t in ts_array]
        batch["temporal"] = torch.from_numpy(_compute_temporal_features(ev_m))
        
        batch["attention_mask"] = torch.ones(self.seq_len, dtype=torch.bool)
        
        return batch
