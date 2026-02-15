import pytest
import torch
from config import Config, ModelConfig, VocabularyConfig
from src.model.transformer_ae import BehavioralTransformerAE

@pytest.fixture
def model_and_batch():
    mc = ModelConfig(max_seq_len=32, d_model=64, n_heads=4, d_ff=128,
                     n_encoder_layers=2, n_decoder_layers=1,
                     gradient_checkpointing=False)
    vc = VocabularyConfig(max_path_length=8)
    model = BehavioralTransformerAE(
        model_cfg=mc, vocab_cfg=vc,
        event_vocab_size=20, path_vocab_size=100, signing_vocab_size=50,
    )
    B, S, P = 4, 32, 8
    batch = {
        "event_type_ids": torch.randint(0, 20, (B, S)),
        "proc_path_ids": torch.randint(0, 100, (B, S, P)),
        "tgt_path_ids": torch.randint(0, 100, (B, S, P)),
        "signing_ids": torch.randint(0, 50, (B, S)),
        "numerical": torch.randn(B, S, 3),
        "temporal": torch.randn(B, S, 4),
        "attention_mask": torch.ones(B, S, dtype=torch.bool),
    }
    return model, batch

def test_forward_produces_loss(model_and_batch):
    model, batch = model_and_batch
    out = model(batch)
    assert "loss" in out
    assert out["loss"].shape == ()
    assert out["loss"].item() > 0

def test_anomaly_score_shape(model_and_batch):
    model, batch = model_and_batch
    scores = model.compute_anomaly_score(batch)
    assert scores.shape == (4,)

def test_attention_weights_extraction(model_and_batch):
    model, batch = model_and_batch
    weights = model.get_attention_weights(batch)
    assert len(weights) == 2  # n_encoder_layers
    assert weights[0].shape[0] == 4  # batch size
    assert weights[0].shape[2] == 32  # seq_len
