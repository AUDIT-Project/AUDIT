"""
GPU-optimized training loop with AMP, gradient accumulation, and logging.
Owner: L
"""
import logging
import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from config import TrainingConfig
from src.utils.io_utils import _log_device_memory

logger = logging.getLogger("macos_ueba.trainer")

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

class Trainer:
    def __init__(self, model: nn.Module, dataset: Dataset, config):
        self.model = model
        self.dataset = dataset
        self.cfg = config.training
        
        self.device = torch.device(self.cfg.device)
        self.model.to(self.device)
        
        if self.cfg.use_compile and hasattr(torch, "compile"):
            logger.info("Compiling model for speedup...")
            self.model = torch.compile(self.model)
            
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            fused=self.cfg.device == "cuda"
        )
        
        if self.cfg.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.use_amp)
        
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory if self.cfg.device == "cuda" else False,
            prefetch_factor=self.cfg.prefetch_factor if self.cfg.num_workers > 0 else None,
        )
        
        total_steps = len(self.loader) * self.cfg.epochs // self.cfg.gradient_accumulation_steps
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, self.cfg.warmup_steps, total_steps
        )
        
    def fit(self):
        logger.info(f"Starting training on {self.device}")
        _log_device_memory()
        
        self.model.train()
        global_step = 0
        
        for epoch in range(1, self.cfg.epochs + 1):
            epoch_loss = 0.0
            self.optimizer.zero_grad(set_to_none=True)
            
            start_time = time.time()
            micro_step = 0
            
            for i, batch in enumerate(self.loader):
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                with torch.autocast(device_type=self.cfg.device, enabled=self.cfg.use_amp, dtype=torch.float16):
                    out = self.model(batch)
                    loss = out["loss"] / self.cfg.gradient_accumulation_steps
                    
                self.scaler.scale(loss).backward()
                epoch_loss += loss.item() * self.cfg.gradient_accumulation_steps
                
                if (i + 1) % self.cfg.gradient_accumulation_steps == 0 or (i + 1) == len(self.loader):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    
                micro_step += 1
                
                if micro_step % self.cfg.log_every_n_steps == 0:
                    events_per_sec = (micro_step * self.cfg.batch_size * 512) / (time.time() - start_time)
                    logger.info(f"Epoch {epoch} | Step {micro_step}/{len(self.loader)} | Loss: {loss.item():.4f} | Events/s: {events_per_sec:.0f}")
                    
            avg_loss = epoch_loss / max(1, len(self.loader))
            logger.info(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            self.cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": avg_loss,
            }, self.cfg.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")
