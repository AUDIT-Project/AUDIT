import pytest
import numpy as np
from src.inference.calibration import build_calibrator

def test_platt_calibrator():
    calibrator = build_calibrator("platt")
    
    # normal scores is low (mean 0.1), anomaly is high (mean 0.8)
    normal = np.random.normal(0.1, 0.05, 100)
    anomaly = np.random.normal(0.8, 0.1, 100)
    
    calibrator.fit(normal, anomaly)
    
    test_normal = np.array([0.1, 0.12])
    test_anomaly = np.array([0.8, 0.9])
    
    p_normal = calibrator.predict_proba(test_normal)
    p_anomaly = calibrator.predict_proba(test_anomaly)
    
    assert np.all(p_normal > 0.8)
    assert np.all(p_anomaly < 0.2)

def test_isotonic_calibrator():
    calibrator = build_calibrator("isotonic")
    
    normal = np.random.normal(0.1, 0.05, 100)
    anomaly = np.random.normal(0.8, 0.1, 100)
    
    calibrator.fit(normal, anomaly)
    
    test_normal = np.array([0.1, 0.12])
    test_anomaly = np.array([0.8, 0.9])
    
    p_normal = calibrator.predict_proba(test_normal)
    p_anomaly = calibrator.predict_proba(test_anomaly)
    
    assert np.all(p_normal > 0.8)
    assert np.all(p_anomaly < 0.2)
