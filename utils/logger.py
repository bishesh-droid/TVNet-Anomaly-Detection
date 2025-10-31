# utils/logger.py
"""
Simple logger using Python's logging module and helper functions for checkpointing.
"""
import logging
import os
import json

def get_logger(name="tvnet", log_dir=None, level="INFO"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "run.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

def save_checkpoint(state, path):
    """
    state: dict with at least 'model_state' and optionally 'optimizer_state' etc.
    """
    import torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path, map_location=None):
    import torch
    return torch.load(path, map_location=map_location)

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
