"""
Attention-based explanations — maps probability scores back to events.
Owner: L
"""
import torch

class AttentionRollout:
    def __init__(self, model):
        self.model = model
        
    def generate_explanation(self, batch: dict) -> torch.Tensor:
        """
        Implements Attention Rollout (Abnar & Zuidema 2020)
        Returns per-token attribution score (0-1).
        batch should have batch_size=1 ideally for clarity.
        """
        self.model.eval()
        with torch.no_grad():
            from torch.cuda.amp import autocast
            with autocast(enabled=self.model.cfg.device == "cuda", dtype=torch.float16):
                # We need attention maps
                attentions = self.model.get_attention_weights(batch)
        
        # attentions is a list of [B, S, S]
        # Rollout
        # V_t = V_{t-1} @ (0.5 * I + 0.5 * A_t) -> standard rollout adds identity to account for residual
        B, seq_len = attentions[0].shape[0], attentions[0].shape[1]
        
        rollout = torch.eye(seq_len, device=attentions[0].device).unsqueeze(0).repeat(B, 1, 1)
        
        for A in attentions:
            # Add identity matrix to account for residual connection
            I = torch.eye(seq_len, device=A.device).unsqueeze(0)
            A_aug = 0.5 * A + 0.5 * I
            
            # Divide each row by its sum to normalize
            # A_aug = A_aug / A_aug.sum(dim=-1, keepdim=True)
            
            rollout = torch.bmm(rollout, A_aug)
            
        # The first token is typically [CLS], but here we can just mean pool the rows 
        # to see which tokens were attended to on average, or use the mean rollout
        # Let's say we return the mean over all query tokens.
        token_attributions = rollout.mean(dim=1) # (B, S)
        
        # Normalize to 0-1
        min_v = token_attributions.min(dim=-1, keepdim=True)[0]
        max_v = token_attributions.max(dim=-1, keepdim=True)[0]
        token_attributions = (token_attributions - min_v) / (max_v - min_v + 1e-9)
        
        return token_attributions

def generate_report(batch: dict, token_attributions: torch.Tensor, threshold: float = 0.8) -> list:
    """Returns ranked list of highest attribution events"""
    B, S = token_attributions.shape
    reports = []
    
    for b in range(B):
        # find tokens with score > threshold
        scores = token_attributions[b]
        indices = torch.argsort(scores, descending=True)
        report = []
        for idx in indices:
            if scores[idx] < threshold:
                break
            report.append({
                "sequence_index": idx.item(),
                "score": scores[idx].item(),
                "event_type_id": batch["event_type_ids"][b, idx].item()
            })
        reports.append(report)
    return reports
