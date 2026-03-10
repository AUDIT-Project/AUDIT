# macOS Behavioral Anomaly Detection (UEBA)

Continuous probabilistic user attribution using macOS Endpoint Security logs and Transformer architectures.

## What This System Does

This system answers one question in real time:

> **"What is the probability that the currently active user is the person who originally authenticated?"**

It builds a behavioral fingerprint of a target user by training a Transformer autoencoder exclusively on that user's normal macOS system call sequences (captured via `eslogger`). During inference, any session that deviates from the learned behavioral baseline produces elevated reconstruction error, which is calibrated into a true probability P(user | actions).

```text
                    TRAINING (one-class learning)
                    ════════════════════════════
eslogger NDJSON → [Chunked Data Manager] → [Vocabulary Builder]
                                          → [Memmap Event Store]
                                          → [Transformer Autoencoder]
                                                   │
                                          learns to reconstruct
                                          the user's normal behavior
                                                   │
                    INFERENCE                      ▼
                    ═════════
live eslogger ──→ [Sliding Window] ──→ [Trained Model] ──→ reconstruction error
                                                          │
                                    [Platt Calibration] ◄─┘
                                           │
                                    P(user | actions)
                                           │
                              ┌────────────┴────────────┐
                              │                         │
                         P ≈ 0.95                   P ≈ 0.08
                       "Normal user"           "ANOMALY DETECTED"
                                                       │
                                            [Attention Rollout]
                                                       │
                                            "kextload + remote_thread_create
                                             drove the anomaly score"
```

## Core Concepts

### 1. One-Class Behavioral Modeling

Unlike traditional classifiers that learn boundaries between multiple known users, this system trains exclusively on one user's data. The Transformer autoencoder learns to compress and reconstruct sequences of system calls that match the target user's workflow. Foreign behavior (compromised credentials, insider threat, lateral movement) produces sequences the model cannot reconstruct, yielding high error.

### 2. Hierarchical Event Embedding

Each eslogger event is a deeply nested JSON dictionary. Rather than flattening to a string, the system embeds each component independently:

| Component           | Embedding Method                    | Dimension |
|---------------------|-------------------------------------|-----------|
| Event type          | Learned lookup table                | 64-96     |
| Process path        | Path tokenization + mean-pool       | 80-128    |
| Target path         | Path tokenization + mean-pool       | 80-128    |
| Signing ID          | Learned lookup table                | 32-48     |
| Numerical (pid etc) | Linear projection                   | 32-48     |
| Temporal (Δt, hour) | Linear projection                   | 48-64     |

These are concatenated and projected to `d_model`, then sinusoidal positional encoding is added.

### 3. Transformer Autoencoder

```text
Input embeddings ──→ [Encoder: 4-8 layers self-attention]
                            │
                     compressed representation
                            │
                     [Decoder: 2-4 layers cross-attention] ──→ Reconstructed embeddings
                            │
                     MSE(input, reconstructed) = anomaly score
```

Key architectural decisions:
- **Pre-LN** (`norm_first=True`) for training stability
- **Gradient checkpointing** on memory-constrained devices
- **AMP fp16** for GPU acceleration (2× memory savings)

### 4. Probability Calibration

Raw reconstruction errors are NOT calibrated probabilities. A score of 0.42 does not mean 42% probability. Platt Scaling fits a logistic regression on held-out data to transform arbitrary scores into true frequentist probabilities:

```text
P(user=1 | score) = 1 / (1 + exp(A·score + B))
```

### 5. Chunked Data Management

A 20 GB NDJSON file is split into ≤500 MB numbered chunks. The manifest caches per-chunk, per-ruid event counts so that subsequent runs never re-read the full dataset just to count events.

```text
data/chunks/
├── manifest.json          ← ruid counts, chunk metadata
├── chunk_0000.ndjson      ← 500 MB
├── chunk_0001.ndjson
└── ...
```

## Project Structure

```text
macos-ueba/
├── README.md
├── requirements.txt
├── main.py                          # CLI entry point (orchestration)
│
├── config/
│   ├── __init__.py
│   └── settings.py                  # All dataclass configs + GPU auto-tune
│
├── src/
│   ├── __init__.py
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── eslogger_stream.py       # Live eslogger subprocess
│   │   └── chunked_data_manager.py  # File splitting + manifest
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── event_parser.py          # JSON → ESEvent dataclass
│   │   ├── vocabulary.py            # Token vocabularies
│   │   └── dataset.py               # MemmapEventStore + PyTorch Dataset
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── embeddings.py            # PathEmbedding, positional encoding
│   │   └── transformer_ae.py        # Encoder, Decoder, full AE, MDN head
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py               # AMP training loop + scheduling
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── calibration.py           # Platt + Isotonic calibrators
│   │   ├── interpretability.py      # Attention rollout + SHAP-like
│   │   └── detector.py              # ContinuousDetector (real-time)
│   │
│   └── utils/
│       ├── __init__.py
│       └── io_utils.py              # SSD copy, device memory logging
│
├── notebooks/
│   └── colab_train.py               # Single-file Colab runner
│
├── tests/
│   ├── __init__.py
│   ├── test_event_parser.py
│   ├── test_vocabulary.py
│   ├── test_chunked_manager.py
│   ├── test_dataset.py
│   ├── test_model_forward.py
│   └── test_calibration.py
│
└── data/                            # Created at runtime
    ├── chunks/
    ├── vocabs/
    ├── memmap_cache/
    └── checkpoints/
```

## Quick Start

### 1. Collect training data
On macOS, requires root and Endpoint Security entitlements:
```bash
sudo python main.py collect --user 501 --duration 7200 \
     --output data/raw/session1.ndjson
```

### 2. Train the model
On any machine with a GPU (or CPU):
```bash
python main.py train --data data/raw/session1.ndjson \
     --user 501 --device cuda
```

### 3. Calibrate
Transforms reconstruction scores to true probabilities:
```bash
python main.py calibrate --data data/raw/session1.ndjson \
     --user 501
```

### 4. Real-time detection
On macOS, requires root:
```bash
sudo python main.py detect --user 501
```

### Add more training data later:
```bash
python main.py add-data --data data/raw/session2.ndjson
python main.py train --data data/raw/session2.ndjson \
     --user 501 --force-rebuild-vocab
```

## Requirements

- Python 3.10+
- PyTorch 2.1+ (with CUDA for GPU training)
- scikit-learn
- numpy
- psutil
- macOS 13+ with Endpoint Security entitlement (for data collection streams)

## Authors

- **Lekhit Borole** — Model architecture, training pipeline, calibration, interpretability
- **Sarvesh Halbe** — Data infrastructure, I/O optimization, chunking, testing, deployment
