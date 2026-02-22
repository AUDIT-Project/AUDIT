"""
Real-time continuous user verification.
Owner: L
"""
import logging
from collections import deque
from typing import Optional, List

import numpy as np
import torch

from src.preprocessing.event_parser import ESEvent
from src.preprocessing.dataset import EventFeatureEncoder, _compute_temporal_features
from src.inference.interpretability import AttentionRollout, generate_report

logger = logging.getLogger("macos_ueba.detector")

class ContinuousDetector:
    def __init__(self, model, calibrator, encoder: EventFeatureEncoder, config):
        self.model = model
        self.calibrator = calibrator
        self.encoder = encoder
        self.cfg = config.detection
        self.model_cfg = config.model
        
        self.device = torch.device(config.training.device)
        self.model.to(self.device)
        self.model.eval()
        
        self.rollout = AttentionRollout(self.model)
        
        # Buffers
        self.window_buffer: List[ESEvent] = []
        self.prob_history = deque(maxlen=self.cfg.rolling_average_len)
        
    def process_event(self, event: ESEvent) -> Optional[dict]:
        """
        Process single event. Returns alert dict if anomaly detected, else None.
        """
        self.window_buffer.append(event)
        
        # We process when we reach window_size, and then we stride.
        if len(self.window_buffer) >= self.cfg.window_size:
            result = self._evaluate_window()
            
            # stride by removing events from the start
            self.window_buffer = self.window_buffer[self.cfg.window_stride:]
            
            return result
        return None
        
    def _evaluate_window(self) -> Optional[dict]:
        # Batch encode
        encoded_events = [self.encoder.encode(e) for e in self.window_buffer]
        
        # Collate into single (1, S, D) batch
        batch = {
            "event_type_ids": torch.tensor([e["event_type_id"] for e in encoded_events]).unsqueeze(0).to(self.device),
            "proc_path_ids": torch.tensor([e["proc_path_ids"] for e in encoded_events]).unsqueeze(0).to(self.device),
            "tgt_path_ids": torch.tensor([e["tgt_path_ids"] for e in encoded_events]).unsqueeze(0).to(self.device),
            "signing_ids": torch.tensor([e["signing_id"] for e in encoded_events]).unsqueeze(0).to(self.device),
            "numerical": torch.tensor([e["numerical"] for e in encoded_events]).unsqueeze(0).to(self.device),
            "attention_mask": torch.ones(1, len(encoded_events), dtype=torch.bool).to(self.device),
        }
        batch["temporal"] = torch.tensor(_compute_temporal_features(self.window_buffer)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            from torch.cuda.amp import autocast
            with autocast(enabled=self.device.type == "cuda", dtype=torch.float16):
                mse_score = self.model.compute_anomaly_score(batch).cpu().numpy()[0]
                
        prob_user = self.calibrator.predict_proba(np.array([mse_score]))[0]
        self.prob_history.append(prob_user)
        
        avg_prob = sum(self.prob_history) / len(self.prob_history)
        
        if avg_prob < self.cfg.alert_probability_threshold:
            # Anomaly detected!
            attributions = self.rollout.generate_explanation(batch)
            report = generate_report(batch, attributions, threshold=0.7)[0]
            
            logger.warning(f"ANOMALY DETECTED. User Prob: {avg_prob:.4f}")
            
            return {
                "avg_probability": float(avg_prob),
                "instant_probability": float(prob_user),
                "reconstruction_error": float(mse_score),
                "explanation": report
            }
            
        return None
