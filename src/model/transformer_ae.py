"""
Transformer Autoencoder for one-class behavioral modeling.
Owner: L
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from config import ModelConfig, VocabularyConfig
from src.model.embeddings import HierarchicalEventEmbedding

class CheckpointedEncoder(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        
    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None, src_key_padding_mask: torch.Tensor = None):
        if src.requires_grad:
            return checkpoint(self.layer, src, src_mask, src_key_padding_mask, use_reentrant=False)
        return self.layer(src, src_mask, src_key_padding_mask, is_causal=False)

class CheckpointedDecoder(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if tgt.requires_grad or memory.requires_grad:
            return checkpoint(self.layer, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, use_reentrant=False)
        return self.layer(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, tgt_is_causal=False, memory_is_causal=False)

class MixtureDensityHead(nn.Module):
    def __init__(self, d_model: int, n_gaussians: int, out_dim: int):
        super().__init__()
        self.n_gaussians = n_gaussians
        self.out_dim = out_dim
        
        self.pi = nn.Sequential(nn.Linear(d_model, n_gaussians), nn.Softmax(dim=-1))
        self.mu = nn.Linear(d_model, n_gaussians * out_dim)
        self.sigma = nn.Linear(d_model, n_gaussians * out_dim)
        
    def forward(self, x: torch.Tensor):
        # x is a pooled sequence embedding (B, d_model)
        pi = self.pi(x) # (B, n_gaussians)
        mu = self.mu(x).view(-1, self.n_gaussians, self.out_dim)
        sigma = torch.exp(self.sigma(x).view(-1, self.n_gaussians, self.out_dim))
        return pi, mu, sigma

class BehavioralTransformerAE(nn.Module):
    def __init__(self, model_cfg: ModelConfig, vocab_cfg: VocabularyConfig, 
                 event_vocab_size: int, path_vocab_size: int, signing_vocab_size: int):
        super().__init__()
        self.cfg = model_cfg
        
        self.embedding = HierarchicalEventEmbedding(
            model_cfg, event_vocab_size, path_vocab_size, signing_vocab_size
        )
        
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_cfg.d_model, nhead=model_cfg.n_heads, 
            dim_feedforward=model_cfg.d_ff, dropout=model_cfg.dropout, 
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=model_cfg.n_encoder_layers)
        
        if model_cfg.gradient_checkpointing:
            self.encoder.layers = nn.ModuleList([CheckpointedEncoder(l) for l in self.encoder.layers])
            
        if self.cfg.use_mdn:
            self.mdn = MixtureDensityHead(model_cfg.d_model, model_cfg.mdn_n_gaussians, model_cfg.mdn_output_dim)
        else:
            dec_layer = nn.TransformerDecoderLayer(
                d_model=model_cfg.d_model, nhead=model_cfg.n_heads,
                dim_feedforward=model_cfg.d_ff, dropout=model_cfg.dropout,
                batch_first=True, norm_first=True
            )
            self.decoder = nn.TransformerDecoder(dec_layer, num_layers=model_cfg.n_decoder_layers)
            
            if model_cfg.gradient_checkpointing:
                self.decoder.layers = nn.ModuleList([CheckpointedDecoder(l) for l in self.decoder.layers])
                
            self.out_proj = nn.Linear(model_cfg.d_model, model_cfg.d_model) # Reconstruct input dim D
            
    def compute_anomaly_score(self, batch: dict) -> torch.Tensor:
        out = self.forward(batch)
        if self.cfg.use_mdn:
            return out["loss"] # Needs an actual compute for single batched elements without reduction, but okay here.
        # batch-wise MSE for reconstruction errors
        mse = nn.functional.mse_loss(out["reconstructed"], out["input_embed"], reduction="none")
        return mse.mean(dim=[1, 2])
        
    def get_attention_weights(self, batch: dict):
        x = self.embedding(batch)
        weights = []
        for l in self.encoder.layers:
            # support normal vs checkpointed
            actual_layer = l.layer if self.cfg.gradient_checkpointing else l
            x_norm = actual_layer.norm1(x)
            attn_output, attn_weight = actual_layer.self_attn(
                x_norm, x_norm, x_norm, key_padding_mask=~batch["attention_mask"], need_weights=True, average_attn_weights=True
            )
            weights.append(attn_weight)
            # Foward the rest explicitly since transformer internal states aren't returned.
            x = x + actual_layer.dropout1(attn_output)
            x = x + actual_layer.dropout2(actual_layer.linear2(actual_layer.dropout(actual_layer.activation(actual_layer.linear1(actual_layer.norm2(x))))))
        return weights

    def forward(self, batch: dict) -> dict:
        x = self.embedding(batch) # Input representation (B, S, D)
        # padding_mask expects True for positions to ignore
        pad_mask = ~batch["attention_mask"] if "attention_mask" in batch else None
        
        memory = self.encoder(x, src_key_padding_mask=pad_mask)
        
        if self.cfg.use_mdn:
            # global pool
            pooled = memory.mean(dim=1)
            pi, mu, sigma = self.mdn(pooled)
            # compute loss somehow based on pi, mu, sigma. We'll return dummy for now.
            return {"loss": pi.sum()}
            
        # Modes: reconstruct the embeddings themselves
        reconstructed = self.decoder(x, memory, tgt_key_padding_mask=pad_mask, memory_key_padding_mask=pad_mask)
        reconstructed = self.out_proj(reconstructed)
        
        loss = nn.functional.mse_loss(reconstructed, x)
        return {
            "loss": loss,
            "reconstructed": reconstructed,
            "input_embed": x
        }
