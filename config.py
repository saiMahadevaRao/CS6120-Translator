#!/usr/bin/env python3
"""
Configuration file for the English-French Neural Machine Translation System.

This file contains all the hyperparameters, paths, and settings used throughout
the system. Modify these values to experiment with different configurations.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"

# Language configuration
INPUT_LANGUAGE = "english"
OUTPUT_LANGUAGE = "french"
REVERSE_LANGUAGES = False  # Set to True to swap input/output languages

# Data processing
MAX_LENGTH = 60           # Maximum sentence length in tokens
MAX_FILE_SIZE = 200000    # Maximum number of training examples to load
BATCH_SIZE = 128          # Training batch size

# Model architecture
HIDDEN_SIZE = 256         # Hidden layer dimensions
ENCODER_LAYERS = 3        # Number of encoder layers
DECODER_LAYERS = 3        # Number of decoder layers
ENCODER_HEADS = 8         # Number of attention heads in encoder
DECODER_HEADS = 8         # Number of attention heads in decoder
ENCODER_FF_SIZE = 512     # Feed-forward layer size in encoder
DECODER_FF_SIZE = 512     # Feed-forward layer size in decoder
ENCODER_DROPOUT = 0.1     # Dropout rate in encoder
DECODER_DROPOUT = 0.1     # Dropout rate in decoder

# Training configuration
LEARNING_RATE = 0.0005    # Adam optimizer learning rate
EPOCHS = 10               # Number of training epochs
CHECKPOINT_INTERVAL = 5   # Save checkpoint every N epochs

# Device configuration
DEVICE = "auto"           # "auto", "cpu", "cuda", or "cuda:0"
                          # "auto" will automatically select the best available device

# Evaluation
PROBABILITY_THRESHOLD = 0.1  # Minimum probability threshold for translation
ROUGE_METRICS = ["rouge1", "rouge2", "rougeL"]  # ROUGE metrics to compute

# Logging
LOG_LEVEL = "INFO"        # Logging level: DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Model saving
SAVE_FINAL_MODEL = True   # Whether to save the final trained model
SAVE_CHECKPOINTS = True   # Whether to save intermediate checkpoints
CHECKPOINT_FORMAT = "transformer_epoch_{epoch}.pth"  # Checkpoint filename format
FINAL_MODEL_NAME = "transformer_model.pt"  # Final model filename

# Data preprocessing
NORMALIZE_TEXT = True     # Whether to normalize text (lowercase, remove special chars)
REMOVE_PUNCTUATION = False  # Whether to remove punctuation during preprocessing
MIN_WORD_FREQUENCY = 1    # Minimum word frequency to include in vocabulary

# Validation and testing
VALIDATION_SPLIT = 0.1    # Fraction of data to use for validation
TEST_SPLIT = 0.1          # Fraction of data to use for testing
RANDOM_SEED = 42          # Random seed for reproducibility

# Performance optimization
NUM_WORKERS = 4           # Number of data loading workers
PIN_MEMORY = True         # Whether to pin memory for faster GPU transfer
GRADIENT_CLIP = 1.0       # Gradient clipping threshold (None to disable)

def get_device():
    """Get the appropriate device for training/inference."""
    if DEVICE == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return DEVICE

def get_model_path():
    """Get the path for the trained model."""
    return SAVED_MODELS_DIR / f"{INPUT_LANGUAGE}2{OUTPUT_LANGUAGE}" / FINAL_MODEL_NAME

def get_dictionary_paths():
    """Get the paths for input and output dictionaries."""
    base_path = SAVED_MODELS_DIR / f"{INPUT_LANGUAGE}2{OUTPUT_LANGUAGE}"
    return {
        "input": base_path / "input_dic.pkl",
        "output": base_path / "output_dic.pkl"
    }

def create_directories():
    """Create necessary directories if they don't exist."""
    SAVED_MODELS_DIR.mkdir(exist_ok=True)
    (SAVED_MODELS_DIR / f"{INPUT_LANGUAGE}2{OUTPUT_LANGUAGE}").mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

# Create directories on import
create_directories()
