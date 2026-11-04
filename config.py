"""
Central Configuration for Sketch-to-Photo Landscape cGAN.

Hardware-aware settings tailored for Apple Silicon (M-series) CPU execution:
- Lightweight U-Net Generator (depth 5, filters [32, 64, 128, 256, 256])
- Lightweight PatchGAN Discriminator (depth 3, filters [64, 128, 256])
- 128x128 main resolution with 64x64 pilot mode
- Pure float32 precision, batch size 1, tf.data streaming
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PAIRS_CSV_PATH = PROCESSED_DATA_DIR / "pairs.csv"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LATEST_CHECKPOINT_DIR = CHECKPOINT_DIR / "latest"
BEST_CHECKPOINT_DIR = CHECKPOINT_DIR / "best"
STANDALONE_GENERATOR_DIR = CHECKPOINT_DIR / "generator_standalone"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
SAMPLES_DIR = OUTPUT_DIR / "samples"
GENERATED_DIR = OUTPUT_DIR / "generated"
METRICS_CSV_PATH = OUTPUT_DIR / "evaluation_metrics.csv"
LOG_DIR = PROJECT_ROOT / "logs"

# Ensure runtime directories exist
for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    LATEST_CHECKPOINT_DIR,
    BEST_CHECKPOINT_DIR,
    STANDALONE_GENERATOR_DIR,
    SAMPLES_DIR,
    GENERATED_DIR,
    LOG_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    # Image specifications
    image_size: int = 128
    channels: int = 3
    batch_size: int = 1

    # Generator Architecture (U-Net)
    generator_depth: int = 5
    generator_filters: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 256])

    # Discriminator Architecture (PatchGAN)
    discriminator_depth: int = 3
    discriminator_filters: List[int] = field(default_factory=lambda: [64, 128, 256])

    # Optimization Hyperparameters
    learning_rate: float = 0.0002
    beta_1: float = 0.5
    l1_lambda: float = 100.0

    # Dataset limits
    train_limit: int = 1200
    validation_limit: int = 150
    max_epochs: int = 10

    # Reproducibility
    seed: int = 42

    # Hardware & Performance (CPU on Apple M-series)
    inter_op_threads: int = 2
    intra_op_threads: int = 4
    use_mixed_precision: bool = False  # float32 default on CPU


@dataclass
class PilotConfig(ModelConfig):
    """Lightweight pilot configuration for pipeline verification."""
    image_size: int = 64
    train_limit: int = 100
    validation_limit: int = 20
    max_epochs: int = 3
    generator_filters: List[int] = field(default_factory=lambda: [32, 64, 128, 128, 128])
    discriminator_filters: List[int] = field(default_factory=lambda: [32, 64, 128])


# Default active configuration instance
DEFAULT_CONFIG = ModelConfig()
PILOT_CONFIG = PilotConfig()

# Expose top-level variables for direct import matching prompt spec
IMAGE_SIZE = DEFAULT_CONFIG.image_size
CHANNELS = DEFAULT_CONFIG.channels
BATCH_SIZE = DEFAULT_CONFIG.batch_size
GENERATOR_DEPTH = DEFAULT_CONFIG.generator_depth
GENERATOR_FILTERS = DEFAULT_CONFIG.generator_filters
DISCRIMINATOR_DEPTH = DEFAULT_CONFIG.discriminator_depth
DISCRIMINATOR_FILTERS = DEFAULT_CONFIG.discriminator_filters
LEARNING_RATE = DEFAULT_CONFIG.learning_rate
BETA_1 = DEFAULT_CONFIG.beta_1
L1_LAMBDA = DEFAULT_CONFIG.l1_lambda
TRAIN_LIMIT = DEFAULT_CONFIG.train_limit
VALIDATION_LIMIT = DEFAULT_CONFIG.validation_limit
MAX_EPOCHS = DEFAULT_CONFIG.max_epochs
SEED = DEFAULT_CONFIG.seed


def configure_tf_cpu(config: ModelConfig = DEFAULT_CONFIG):
    """
    Apply CPU threading and precision settings to TensorFlow.
    """
    import tensorflow as tf

    # Force CPU visibility if needed or ensure CPU is the compute device
    cpus = tf.config.list_physical_devices("CPU")
    if cpus:
        try:
            # Set thread pools for CPU execution
            tf.config.threading.set_inter_op_parallelism_threads(config.inter_op_threads)
            tf.config.threading.set_intra_op_parallelism_threads(config.intra_op_threads)
        except RuntimeError:
            pass  # Threads already initialized

    # Float32 precision policy
    tf.keras.backend.set_floatx("float32")
