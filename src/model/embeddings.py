"""
Multi-component event embedding pipeline.
Owner: L
"""
import math
import torch
import torch.nn as nn
from config import ModelConfig

class PathEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
    def forward(self, path_ids: torch.Tensor) -> torch.Tensor:
        # path_ids: (B, S, P)
        # embed: (B, S, P, D)
        embed = self.embedding(path_ids)
        # mean pool over P (ignoring PAD=0)
        mask = (path_ids != 0).unsqueeze(-1).float()
        sum_embed = (embed * mask).sum(dim=2)
        valid_len = mask.sum(dim=2).clamp(min=1.0)
        return sum_embed / valid_len

class SinusoidalPositionalEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class HierarchicalEventEmbedding(nn.Module):
    def __init__(self, config: ModelConfig, event_vocab_size: int, path_vocab_size: int, signing_vocab_size: int):
        super().__init__()
        self.config = config
        
        self.event_embed = nn.Embedding(event_vocab_size, config.event_type_embed_dim)
        self.proc_path_embed = PathEmbedding(path_vocab_size, config.path_embed_dim)
        self.tgt_path_embed = PathEmbedding(path_vocab_size, config.path_embed_dim)
        self.signing_embed = nn.Embedding(signing_vocab_size, 32) # Fixed to 32
        
        self.numerical_proj = nn.Linear(3, config.numerical_embed_dim)
        self.temporal_proj = nn.Linear(4, config.temporal_embed_dim)
        
        concat_dim = (
            config.event_type_embed_dim + 
            config.path_embed_dim * 2 + 
            32 + 
            config.numerical_embed_dim + 
            config.temporal_embed_dim
        )
        
        self.proj = nn.Sequential(
            nn.Linear(concat_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        self.pos_enc = SinusoidalPositionalEnc(config.d_model, config.max_seq_len)
        
    def forward(self, batch: dict) -> torch.Tensor:
        e_emb = self.event_embed(batch["event_type_ids"])
        p_emb = self.proc_path_embed(batch["proc_path_ids"])
        t_emb = self.tgt_path_embed(batch["tgt_path_ids"])
        s_emb = self.signing_embed(batch["signing_ids"])
        n_emb = self.numerical_proj(batch["numerical"])
        temp_emb = self.temporal_proj(batch["temporal"])
        
        cat_emb = torch.cat([e_emb, p_emb, t_emb, s_emb, n_emb, temp_emb], dim=-1)
        x = self.proj(cat_emb)
        return self.pos_enc(x)
