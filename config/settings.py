"""
Configuration dataclasses and GPU auto-tuning.

This is the single source of truth for all tunable parameters.
Both L and S import from here — never hardcode constants elsewhere.

GPU Auto-Tuning Strategy (by L):
  Detect GPU VRAM at startup, scale batch_size / seq_len / model dims
  to utilize ~70% of available memory.  AMP (fp16) is always enabled
  on CUDA.  torch.compile is enabled on Ampere+ (compute >= 7.0).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch

logger = logging.getLogger("macos_ueba.config")

def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

@dataclass
class ESLoggerConfig:
    """Parameters for the eslogger subprocess."""
    event_types: List[str] = field(default_factory=lambda: [
        "exec", "fork", "exit", "open", "close", "create", "rename",
        "unlink", "write", "mmap", "mprotect", "chdir", "signal",
        "login_login", "login_logout", "openssh_login", "openssh_logout",
        "lw_session_lock", "lw_session_unlock", "sudo", "su",
        "authentication", "kextload", "kextunload", "mount",
        "uipc_connect", "uipc_bind", "xpc_connect",
        "setuid", "seteuid", "setmode", "setowner", "setflags",
        "gatekeeper_user_override", "tcc_modify",
        "remote_thread_create", "proc_check", "pty_grant",
        "screensharing_attach", "screensharing_detach",
        "cs_invalidated", "readdir", "stat", "lookup",
        "copyfile", "exchangedata", "truncate", "link",
        "iokit_open", "get_task", "get_task_read",
    ])
    eslogger_path: str = "/usr/bin/eslogger"

@dataclass
class VocabularyConfig:
    """Vocabulary sizing for categorical fields."""
    max_event_types: int = 128
    max_path_tokens: int = 50_000
    special_tokens: int = 4       # [PAD], [UNK], [CLS], [SEP]
    max_path_length: int = 24     # Max tokens per path field

@dataclass
class ModelConfig:
    """Transformer autoencoder architecture parameters."""
    d_model: int = 256
    n_heads: int = 8
    n_encoder_layers: int = 4
    n_decoder_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.1

    # Embedding sub-dimensions
    event_type_embed_dim: int = 64
    path_embed_dim: int = 80
    temporal_embed_dim: int = 48
    numerical_embed_dim: int = 32

    # Sequence parameters
    max_seq_len: int = 512
    stride: int = 128

    # Memory saving
    gradient_checkpointing: bool = True

    # MDN (Framework 3 — alternative to reconstruction)
    use_mdn: bool = False
    mdn_n_gaussians: int = 8
    mdn_output_dim: int = 64

@dataclass
class TrainingConfig:
    """Training loop parameters."""
    epochs: int = 40
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 2000
    gradient_clip: float = 1.0
    validation_split: float = 0.15
    patience: int = 10
    checkpoint_dir: Path = Path("data/checkpoints")
    device: str = field(default_factory=_default_device)
    num_workers: int = 0
    log_every_n_steps: int = 50

    # GPU acceleration (set by auto_configure_for_gpu)
    use_amp: bool = False
    use_compile: bool = False
    pin_memory: bool = False
    prefetch_factor: int = 2

@dataclass
class DetectionConfig:
    """Real-time detection thresholds."""
    window_size: int = 256
    window_stride: int = 64
    alert_probability_threshold: float = 0.15
    rolling_average_len: int = 5
    calibration_method: str = "platt"  # "platt" or "isotonic"

@dataclass
class Config:
    """Top-level configuration container."""
    eslogger: ESLoggerConfig = field(default_factory=ESLoggerConfig)
    vocab: VocabularyConfig = field(default_factory=VocabularyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)

    data_dir: Path = Path("data")
    local_cache_dir: Path = Path("/content/local_cache")
    target_user: str = ""
    chunk_max_bytes: int = 500 * 1024 * 1024  # 500 MB

def auto_configure_for_gpu(config: Config) -> Config:
    """
    Detect GPU capabilities and scale config to utilize ~70% VRAM.

    Owned by L.  S should not modify this function.

    Memory model (approximate):
      static  = 12 * n_params * 4 bytes  (fp32 params + AdamW states)
      dynamic = n_layers * B * H * S**2 * bytes_per_element  (attention)
              + n_layers * B * S * d_model * bytes_per_element

    With AMP: bytes_per_element = 2 for activations, params stay fp32.
    """
    if not torch.cuda.is_available():
        logger.info("No CUDA GPU — keeping conservative defaults.")
        return config

    props = torch.cuda.get_device_properties(0)
    gpu_gb = props.total_memory / (1024 ** 3)
    logger.info(
        "GPU: %s | %.1f GB | Compute %d.%d | %d SMs",
        props.name, gpu_gb, props.major, props.minor,
        props.multi_processor_count,
    )

    tc = config.training
    mc = config.model

    # Always enable on CUDA
    tc.use_amp = True
    tc.pin_memory = True
    tc.num_workers = min(2, os.cpu_count() or 1)
    tc.prefetch_factor = 2
    tc.gradient_accumulation_steps = 1
    mc.gradient_checkpointing = False

    if hasattr(torch, "compile") and props.major >= 7:
        tc.use_compile = True

    # Scale by VRAM tier
    if gpu_gb >= 30:
        logger.info("GPU tier: HIGH (>=30 GB)")
        mc.d_model, mc.n_heads, mc.d_ff = 512, 8, 2048
        mc.n_encoder_layers, mc.n_decoder_layers = 8, 4
        mc.max_seq_len, mc.stride = 2048, 512
        mc.event_type_embed_dim, mc.path_embed_dim = 96, 128
        mc.temporal_embed_dim, mc.numerical_embed_dim = 64, 48
        tc.batch_size = 64
    elif gpu_gb >= 14:
        logger.info("GPU tier: MEDIUM (14-30 GB)")
        mc.d_model, mc.n_heads, mc.d_ff = 384, 8, 1536
        mc.n_encoder_layers, mc.n_decoder_layers = 6, 3
        mc.max_seq_len, mc.stride = 1024, 256
        mc.event_type_embed_dim, mc.path_embed_dim = 80, 96
        mc.temporal_embed_dim, mc.numerical_embed_dim = 56, 40
        tc.batch_size = 48
    elif gpu_gb >= 6:
        logger.info("GPU tier: LOW (6-14 GB)")
        mc.d_model, mc.n_heads, mc.d_ff = 256, 8, 1024
        mc.n_encoder_layers, mc.n_decoder_layers = 4, 2
        mc.max_seq_len, mc.stride = 512, 128
        tc.batch_size = 32
    else:
        logger.info("GPU tier: MINIMAL (<6 GB)")
        mc.gradient_checkpointing = True
        tc.batch_size = 16

    tc.warmup_steps = min(tc.warmup_steps, 1500)

    logger.info(
        "Auto-config: batch=%d seq=%d d=%d ff=%d enc=%d dec=%d "
        "AMP=%s compile=%s ckpt=%s workers=%d",
        tc.batch_size, mc.max_seq_len, mc.d_model, mc.d_ff,
        mc.n_encoder_layers, mc.n_decoder_layers,
        tc.use_amp, tc.use_compile, mc.gradient_checkpointing,
        tc.num_workers,
    )
    return config
