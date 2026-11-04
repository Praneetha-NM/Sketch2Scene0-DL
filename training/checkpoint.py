"""
Robust TensorFlow Checkpointing and Model Persistence Module.

Saves and restores:
- Generator and Discriminator weights
- Generator and Discriminator Adam optimizer states
- Completed epoch counter
- Best validation metric tracker
- Separate latest/ and best/ managers
- Standalone generator export for zero-overhead inference
"""

import os
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import tensorflow as tf

from config import (
    CHECKPOINT_DIR,
    LATEST_CHECKPOINT_DIR,
    BEST_CHECKPOINT_DIR,
    STANDALONE_GENERATOR_DIR,
    IMAGE_SIZE,
)
from models.generator import build_generator
from models.discriminator import build_discriminator


class CheckpointManagerWrapper:
    """
    Wraps tf.train.Checkpoint and tf.train.CheckpointManager for dual latest/best tracking.
    """

    def __init__(
        self,
        generator: tf.keras.Model,
        discriminator: tf.keras.Model,
        gen_optimizer: tf.keras.optimizers.Optimizer,
        disc_optimizer: tf.keras.optimizers.Optimizer,
        latest_dir: Path = LATEST_CHECKPOINT_DIR,
        best_dir: Path = BEST_CHECKPOINT_DIR,
        max_to_keep: int = 3,
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer

        self.latest_dir = Path(latest_dir)
        self.best_dir = Path(best_dir)
        self.latest_dir.mkdir(parents=True, exist_ok=True)
        self.best_dir.mkdir(parents=True, exist_ok=True)

        self.epoch = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.best_metric = tf.Variable(float("inf"), trainable=False, dtype=tf.float32)

        self.checkpoint = tf.train.Checkpoint(
            generator=self.generator,
            discriminator=self.discriminator,
            gen_optimizer=self.gen_optimizer,
            disc_optimizer=self.disc_optimizer,
            epoch=self.epoch,
            best_metric=self.best_metric,
        )

        self.latest_manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=str(self.latest_dir),
            max_to_keep=max_to_keep,
            checkpoint_name="ckpt_latest",
        )

        self.best_manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=str(self.best_dir),
            max_to_keep=2,
            checkpoint_name="ckpt_best",
        )

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        validation_criterion_key: str = "val_gen_l1_loss",
    ) -> str:
        """
        Save latest checkpoint and conditionally update best checkpoint.

        Validation criterion:
        A lower validation L1 loss indicates that the generator produces outputs
        that strictly align closer in pixel space to real landscape photographs.
        """
        self.epoch.assign(epoch)

        current_val_score = metrics.get(validation_criterion_key, float("inf"))
        save_path = self.latest_manager.save()

        # Save metadata JSON alongside checkpoint
        meta_info = {
            "epoch": epoch,
            "metrics": {k: float(v) for k, v in metrics.items()},
            "validation_criterion": validation_criterion_key,
            "validation_criterion_value": float(current_val_score),
            "latest_checkpoint": save_path,
        }
        with open(self.latest_dir / "latest_meta.json", "w") as f:
            json.dump(meta_info, f, indent=2)

        # Check if this qualifies as best checkpoint
        if is_best or current_val_score < float(self.best_metric.numpy()):
            self.best_metric.assign(current_val_score)
            best_path = self.best_manager.save()
            meta_info["best_checkpoint"] = best_path
            with open(self.best_dir / "best_meta.json", "w") as f:
                json.dump(meta_info, f, indent=2)

        # Save standalone generator weights for portable inference
        self.export_standalone_generator(STANDALONE_GENERATOR_DIR)

        return save_path

    def load_checkpoint(self, prefer_best: bool = False) -> int:
        """
        Restore weights from checkpoint if one exists.
        Returns the completed epoch number (or 0 if none).
        """
        target_manager = self.best_manager if (prefer_best and self.best_manager.latest_checkpoint) else self.latest_manager
        latest_ckpt = target_manager.latest_checkpoint

        if latest_ckpt:
            print(f"Restoring checkpoint from: {latest_ckpt}")
            self.checkpoint.restore(latest_ckpt).expect_partial()
            restored_epoch = int(self.epoch.numpy())
            print(f"Successfully restored state from epoch {restored_epoch}.")
            return restored_epoch
        else:
            print("No existing checkpoint found. Starting from scratch with random initialization.")
            return 0

    def export_standalone_generator(self, export_dir: Path = STANDALONE_GENERATOR_DIR):
        """
        Save standalone generator weights and architecture so inference can run
        independently without loading the discriminator or optimizers.
        """
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        weights_path = export_dir / "generator_weights.weights.h5"
        self.generator.save_weights(str(weights_path))

        meta = {
            "image_size": self.generator.input_shape[1],
            "channels": self.generator.input_shape[-1],
            "epoch": int(self.epoch.numpy()),
        }
        with open(export_dir / "generator_info.json", "w") as f:
            json.dump(meta, f, indent=2)


def load_trained_generator(
    weights_or_ckpt_path: Optional[str] = None,
    image_size: Optional[int] = None,
    encoder_filters: Optional[list] = None,
) -> tf.keras.Model:
    """
    Load a trained U-Net generator model for inference.

    Resolution order:
    1. Specified weights_or_ckpt_path
    2. Standalone generator weights (checkpoints/generator_standalone)
    3. Best checkpoint (checkpoints/best)
    4. Latest checkpoint (checkpoints/latest)
    """
    standalone_weights = STANDALONE_GENERATOR_DIR / "generator_weights.weights.h5"
    info_json = STANDALONE_GENERATOR_DIR / "generator_info.json"

    # Auto-detect image size from metadata if not explicitly provided
    if image_size is None:
        if info_json.exists():
            try:
                with open(info_json, "r") as f:
                    meta = json.load(f)
                    image_size = meta.get("image_size", IMAGE_SIZE)
            except Exception:
                image_size = IMAGE_SIZE
        else:
            image_size = IMAGE_SIZE

    generator = build_generator(
        input_shape=(image_size, image_size, 3),
        encoder_filters=encoder_filters,
        name="inference_generator",
    )

    if weights_or_ckpt_path and os.path.exists(weights_or_ckpt_path):
        target_path = weights_or_ckpt_path
        if target_path.endswith(".h5") or target_path.endswith(".weights.h5"):
            generator.load_weights(target_path)
            print(f"Loaded generator weights from {target_path}")
            return generator
        else:
            # Checkpoint directory or prefix
            ckpt = tf.train.Checkpoint(generator=generator)
            status = ckpt.restore(target_path)
            status.expect_partial()
            print(f"Restored generator from checkpoint prefix {target_path}")
            return generator

    if standalone_weights.exists():
        try:
            generator.load_weights(str(standalone_weights))
            print(f"Loaded standalone generator weights from {standalone_weights}")
            return generator
        except Exception as e:
            print(f"Notice loading standalone weights: {e}, checking checkpoints...")

    # Fall back to best or latest checkpoint
    for ckpt_dir in [BEST_CHECKPOINT_DIR, LATEST_CHECKPOINT_DIR]:
        latest_ckpt = tf.train.latest_checkpoint(str(ckpt_dir))
        if latest_ckpt:
            ckpt = tf.train.Checkpoint(generator=generator)
            status = ckpt.restore(latest_ckpt)
            status.expect_partial()
            print(f"Restored generator from {latest_ckpt}")
            return generator

    raise FileNotFoundError(
        "No trained generator checkpoint or weights found. "
        "Please run training first (e.g., python -m training.train --mode pilot or --mode main)."
    )
