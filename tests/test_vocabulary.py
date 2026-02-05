import pytest
from src.preprocessing.vocabulary import Vocabulary, tokenize_path

def test_tokenize_path():
    assert tokenize_path("/usr/bin/python3") == ["usr", "bin", "python3"]
    assert tokenize_path("/") == []
    assert tokenize_path("///usr//bin//") == ["usr", "bin"]

def test_vocabulary_build():
    vocab = Vocabulary(max_size=6)
    vocab.add_many(["apple", "banana", "apple", "cherry", "date", "banana"])
    vocab.build()
    
    assert len(vocab) <= 6
    assert vocab.encode("apple") > 3 # Not special
    assert vocab.encode("unknown_fruit") == 1 # UNK

def test_vocabulary_max_size():
    vocab = Vocabulary(max_size=5)
    vocab.add_many(["a", "b", "c"])
    vocab.build()
    
    assert len(vocab) == 5 # 4 special tokens + 1 word (since max=5, and 5-4 = 1 top word)
    
