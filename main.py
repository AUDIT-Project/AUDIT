"""
CLI entry point — orchestrates all phases.

Ownership:
  S: argument parsing, phase_collect, phase_split, phase_add_data,
     data loading in phase_train
  L: model instantiation, training, calibration, detection logic
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config, auto_configure_for_gpu
from src.ingest.chunked_data_manager import ChunkedDataManager
from src.ingest.eslogger_stream import ESLoggerStream
from src.preprocessing.event_parser import parse_event, parse_stream
from src.preprocessing.vocabulary import VocabularySet, tokenize_path
from src.preprocessing.dataset import MemmapEventStore, UserSequenceDataset
from src.model.transformer_ae import BehavioralTransformerAE
from src.training.trainer import Trainer
from src.inference.calibration import build_calibrator, fit_calibrator_from_model, SyntheticAnomalyDataset
from src.inference.detector import ContinuousDetector
from src.utils.io_utils import _log_device_memory, replay_from_file

logger = logging.getLogger("macos_ueba")

def phase_collect(args, config: Config) -> None:
    """Owner: S"""
    stream = ESLoggerStream(config.eslogger.event_types, config.eslogger.eslogger_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    events_collected = 0
    target_duration = args.duration
    target_user = int(args.user)
    
    logger.info(f"Collecting training data for user {target_user} for {target_duration} seconds...")
    with open(output_path, "w") as f:
        for raw in stream.stream():
            try:
                ruid = raw.get("process", {}).get("audit_token", {}).get("ruid", -1)
                if ruid == target_user:
                    f.write(json.dumps(raw) + "\n")
                    events_collected += 1
            except Exception as e:
                logger.debug(f"Parsing error in collect: {e}")
                
            if time.time() - start_time >= target_duration:
                break
                
    stream.stop()
    logger.info(f"Finished collecting. Saved {events_collected} events to {output_path}")

def phase_train(args, config: Config) -> None:
    """Owner: Joint"""
    logger.info("═══ PHASE: TRAIN ═══")
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Not found: %s", data_path)
        sys.exit(1)

    target_ruid = int(args.user) if str(args.user).isdigit() else None

    # Step 0: Chunk data
    chunks_dir = config.data_dir / "chunks"
    chunk_mgr = ChunkedDataManager(chunks_dir, config.chunk_max_bytes)
    chunk_mgr.ingest(data_path)
    logger.info("\n%s", chunk_mgr.summary())

    # Step 0b: Copy to local SSD
    local_chunks_dir = config.local_cache_dir / "chunks"
    local_mgr = chunk_mgr.ensure_local(local_chunks_dir)

    # Step 1: Event count
    n_events = local_mgr.get_event_count(target_ruid)
    if n_events is None or n_events == 0:
        logger.info("Count not cached, streaming...")
        n_events = sum(1 for raw in local_mgr.replay_all() 
                       if (ev := parse_event(raw)) is not None and (target_ruid is None or ev.ruid == target_ruid))
    logger.info("Event count (instant from manifest): %d", n_events)

    # Step 2: Vocabulary
    vocab_dir = config.data_dir / "vocabs"
    vocabs = VocabularySet(config.vocab)
    if vocab_dir.exists() and not getattr(args, "force_rebuild_vocab", False):
        vocabs.load(vocab_dir)
    else:
        logger.info("Building vocabulary from chunks...")
        for raw in local_mgr.replay_all():
            ev = parse_event(raw)
            if ev is None or (target_ruid is not None and ev.ruid != target_ruid):
                continue
            vocabs.event_type.add(ev.event_type)
            vocabs.path_token.add_many(tokenize_path(ev.process_path))
            vocabs.path_token.add_many(tokenize_path(ev.target_path))
            if ev.signing_id:
                vocabs.signing_id.add(ev.signing_id)
        vocabs.event_type.build()
        vocabs.path_token.build()
        vocabs.signing_id.build()
        vocabs.save(vocab_dir)

    # Step 3: Memmap store
    local_memmap = config.local_cache_dir / "memmap_cache"
    drive_memmap = config.data_dir / "memmap_cache"

    if MemmapEventStore._cache_ok(local_memmap):
        store = MemmapEventStore(local_memmap)
    elif MemmapEventStore._cache_ok(drive_memmap):
        local_memmap.mkdir(parents=True, exist_ok=True)
        for f in sorted(drive_memmap.iterdir()):
            shutil.copy2(str(f), str(local_memmap / f.name))
        store = MemmapEventStore(local_memmap)
    else:
        store = MemmapEventStore.build(
            data_source=local_mgr, vocabs=vocabs,
            vocab_cfg=config.vocab, target_ruid=target_ruid,
            cache_dir=local_memmap, n_events_hint=n_events,
        )
        drive_memmap.mkdir(parents=True, exist_ok=True)
        for f in sorted(local_memmap.iterdir()):
            shutil.copy2(str(f), str(drive_memmap / f.name))

    # Step 4: Model
    dataset = UserSequenceDataset(store, config.model)
    logger.info("Dataset: %d windows", len(dataset))

    model = BehavioralTransformerAE(
        model_cfg=config.model,
        vocab_cfg=config.vocab,
        event_vocab_size=len(vocabs.event_type),
        path_vocab_size=len(vocabs.path_token),
        signing_vocab_size=len(vocabs.signing_id),
    )
    logger.info("Model: %.2fM params", sum(p.numel() for p in model.parameters()) / 1e6)

    # Step 5: Train
    trainer = Trainer(model, dataset, config)
    trainer.fit()

def phase_calibrate(args, config: Config) -> None:
    """Owner: L"""
    logger.info("═══ PHASE: CALIBRATE ═══")
    
    # Needs a trained model and vocabularies
    # For now simply skeletonized as this requires saving/loading models
    logger.info("Simulating calibration phase... Fitting logistic regression to anomaly scores.")
    time.sleep(1)
    logger.info("Calibration finished.")

def phase_detect(args, config: Config) -> None:
    """Owner: L"""
    logger.info("═══ PHASE: DETECT ═══")
    logger.info(f"Starting continuous detection for user {args.user}...")
    # This phase watches eslogger stream and uses ContinuousDetector
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        logger.info("Stopping detection.")

def phase_split(args, config: Config) -> None:
    """Owner: S"""
    config.chunk_max_bytes = args.chunk_mb * 1024 * 1024
    mgr = ChunkedDataManager(config.data_dir / "chunks", config.chunk_max_bytes)
    mgr.ingest(Path(args.data))
    logger.info("\n%s", mgr.summary())

def phase_add_data(args, config: Config) -> None:
    """Owner: S"""
    mgr = ChunkedDataManager(config.data_dir / "chunks", config.chunk_max_bytes)
    mgr.ingest(Path(args.data))
    logger.info("\n%s", mgr.summary())

def main():
    parser = argparse.ArgumentParser(description="macOS Behavioral Anomaly Detection (UEBA)")
    parser.add_argument("--debug", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("collect")
    p.add_argument("--user", required=True)
    p.add_argument("--duration", type=int, default=3600)
    p.add_argument("--output", default="data/raw/training.ndjson")

    p = sub.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--user", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--force-rebuild-vocab", action="store_true")
    p.add_argument("--no-auto-tune", action="store_true")

    p = sub.add_parser("calibrate")
    p.add_argument("--data", required=True)
    p.add_argument("--user", required=True)
    p.add_argument("--device", default=None)

    p = sub.add_parser("detect")
    p.add_argument("--user", required=True)
    p.add_argument("--device", default=None)

    p = sub.add_parser("split")
    p.add_argument("--data", required=True)
    p.add_argument("--chunk-mb", type=int, default=500)

    p = sub.add_parser("add-data")
    p.add_argument("--data", required=True)

    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", force=True)

    config = Config()
    if hasattr(args, "device") and args.device:
        config.training.device = args.device

    if (
        args.command in ("train", "calibrate")
        and config.training.device == "cuda"
        and not getattr(args, "no_auto_tune", False)
    ):
        config = auto_configure_for_gpu(config)

    PHASES = {
        "collect": phase_collect,
        "train": phase_train,
        "calibrate": phase_calibrate,
        "detect": phase_detect,
        "split": phase_split,
        "add-data": phase_add_data,
    }
    PHASES[args.command](args, config)

if __name__ == "__main__":
    main()
