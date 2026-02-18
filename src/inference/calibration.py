"""
Probability calibration — transforms raw anomaly scores into true P(user).
Owner: L
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import torch
import copy
from torch.utils.data import Dataset, DataLoader

class SyntheticAnomalyDataset(Dataset):
    """
    Generates foreign data by randomizing event types + permuting order.
    FIX-1 applied: randint upper bound is max(2, vocab_size).
    """
    def __init__(self, base_dataset, event_vocab_size):
        self.base_dataset = base_dataset
        self.event_vocab_size = max(2, event_vocab_size)
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        batch = copy.deepcopy(self.base_dataset[idx])
        # Randomize event types
        shape = batch["event_type_ids"].shape
        batch["event_type_ids"] = torch.randint(2, self.event_vocab_size, shape)
        
        # Permute numeric features
        idx_perm = torch.randperm(shape[-1])
        batch["numerical"] = batch["numerical"][idx_perm]
        return batch

class PlattCalibrator:
    def __init__(self):
        self.lr = LogisticRegression(solver='lbfgs', fit_intercept=True)
        self.is_fitted = False
        
    def fit(self, normal_scores: np.ndarray, anomaly_scores: np.ndarray):
        X = np.concatenate([normal_scores, anomaly_scores]).reshape(-1, 1)
        # y=1 for normal user, y=0 for anomaly
        y = np.concatenate([np.ones(len(normal_scores)), np.zeros(len(anomaly_scores))])
        self.lr.fit(X, y)
        self.is_fitted = True
        
    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros_like(scores)
        # Returns probability of class 1 (normal user)
        return self.lr.predict_proba(scores.reshape(-1, 1))[:, 1]

class IsotonicCalibrator:
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds='clip', increasing=False)
        self.is_fitted = False
        
    def fit(self, normal_scores: np.ndarray, anomaly_scores: np.ndarray):
        X = np.concatenate([normal_scores, anomaly_scores])
        y = np.concatenate([np.ones(len(normal_scores)), np.zeros(len(anomaly_scores))])
        self.iso.fit(X, y)
        self.is_fitted = True
        
    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros_like(scores)
        return self.iso.predict(scores)

def build_calibrator(method: str = "platt"):
    if method == "isotonic":
        return IsotonicCalibrator()
    return PlattCalibrator()

def fit_calibrator_from_model(model, normal_dataset, anomaly_dataset, config, method="platt"):
    model.eval()
    device = torch.device(config.training.device)
    
    def get_scores(dataset):
        loader = DataLoader(dataset, batch_size=config.training.batch_size)
        all_scores = []
        with torch.no_grad():
            from torch.cuda.amp import autocast
            for batch in loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                with autocast(enabled=config.training.use_amp, dtype=torch.float16):
                    scores = model.compute_anomaly_score(batch)
                all_scores.append(scores.cpu().numpy())
        return np.concatenate(all_scores)
        
    normal_scores = get_scores(normal_dataset)
    anomaly_scores = get_scores(anomaly_dataset)
    
    calibrator = build_calibrator(method)
    calibrator.fit(normal_scores, anomaly_scores)
    return calibrator
