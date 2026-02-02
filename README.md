# macOS Behavioral Anomaly Detection (UEBA)

Continuous probabilistic user attribution using macOS Endpoint Security logs and Transformer architectures.

## What This System Does

This system answers one question in real time:

> **"What is the probability that the currently active user is the person who originally authenticated?"**

It builds a behavioral fingerprint of a target user by training a Transformer autoencoder exclusively on that user's normal macOS system call sequences (captured via `eslogger`). During inference, any session that deviates from the learned behavioral baseline produces elevated reconstruction error, which is calibrated into a true probability P(user | actions).

## Core Concepts

### 1. One-Class Behavioral Modeling

Unlike traditional classifiers that learn boundaries between multiple known users, this system trains exclusively on one user's data.

### 2. Hierarchical Event Embedding

Each eslogger event is embedded into a dense vector representing its traits like path, temporal properties, and numeric properties.

### 3. Transformer Autoencoder

Reconstructing sequences of events, generating an anomaly score based on the MSE of the reconstruction versus the actual embedding.

### 4. Probability Calibration

Turning arbitrary MSE loss scores into true probabilities via Logistic Platt Scaling.
