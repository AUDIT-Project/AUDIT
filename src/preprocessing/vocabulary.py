"""
Vocabulary and Tokenization utilities.
Owner: S
"""
import json
import os
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional

from config import VocabularyConfig

def tokenize_path(path: str) -> List[str]:
    """Tokenize a file path into components."""
    if not path or path == "/":
        return []
    parts = path.strip("/").split("/")
    return [p for p in parts if p]

class Vocabulary:
    def __init__(self, max_size: int, special_tokens: int = 4):
        self.max_size = max_size
        self.special_tokens = special_tokens
        self.counts = Counter()
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        # Special tokens: 0=PAD, 1=UNK, 2=CLS, 3=SEP
        
    def add(self, token: str):
        self.counts[token] += 1
        
    def add_many(self, tokens: List[str]):
        for t in tokens:
            self.add(t)
            
    def build(self):
        # Assign IDs to top N tokens
        self.token_to_id = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
        self.id_to_token = {0: "[PAD]", 1: "[UNK]", 2: "[CLS]", 3: "[SEP]"}
        
        # Sort by count
        most_common = self.counts.most_common(self.max_size - self.special_tokens)
        
        current_id = self.special_tokens
        for token, _ in most_common:
            self.token_to_id[token] = current_id
            self.id_to_token[current_id] = token
            current_id += 1
            
    def __len__(self) -> int:
        return len(self.token_to_id)
        
    def encode(self, token: str) -> int:
        return self.token_to_id.get(token, 1) # 1 is UNK

class VocabularySet:
    def __init__(self, config: VocabularyConfig):
        self.config = config
        self.event_type = Vocabulary(config.max_event_types)
        self.path_token = Vocabulary(config.max_path_tokens)
        self.signing_id = Vocabulary(1000) # Arbitrary size for signing ID
        
    def save(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, vocab in [("event_type", self.event_type), 
                            ("path_token", self.path_token), 
                            ("signing_id", self.signing_id)]:
            with open(output_dir / f"{name}.json", "w") as f:
                json.dump({"token_to_id": vocab.token_to_id, 
                           "id_to_token": {str(k): v for k, v in vocab.id_to_token.items()}}, f)
                           
    def load(self, input_dir: Path):
        for name, vocab in [("event_type", self.event_type), 
                            ("path_token", self.path_token), 
                            ("signing_id", self.signing_id)]:
            with open(input_dir / f"{name}.json", "r") as f:
                data = json.load(f)
                vocab.token_to_id = data["token_to_id"]
                vocab.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
