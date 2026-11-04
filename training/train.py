"""
cGAN Training Engine for Sketch-to-Photo Landscape Synthesis.

Supports:
- Stage 1: Pipeline Verification Pilot (--mode pilot): 64x64, 100 pairs, 3 epochs
- Stage 2: Main Training (--mode main): 128x128, 1200 pairs, 150 val pairs, up to 10 epochs
- Stage 3: Resumed / Extended Training (--resume): Picks up from latest saved checkpoint
- Time-bounded graceful stopping (--max-duration-min)
- Process lock to prevent concurrent runs
- TensorBoard and progress logging
- Per-epoch validation visual sample grids and checkpointing
"""

import os
import sys
import time
import argparse
import atexit
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import tensorflow as tf

from config import (
    DEFAULT_CONFIG,
    PILOT_CONFIG,
    ModelConfig,
    configure_tf_cpu,
    PROJECT_ROOT,
    PAIRS_CSV_PATH,
    SAMPLES_DIR,
    OUTPUT_DIR,
    RAW_DATA_DIR,
)
from utils.seed import set_seed
from models.generator import build_generator, get_generator_summary
from models.discriminator import build_discriminator, get_discriminator_summary
from models.losses import generator_loss, discriminator_loss
from training.train_step import train_step, validation_step
from training.checkpoint import CheckpointManagerWrapper
from training.metrics import TrainingMetricsTracker, format_duration
from training.train_step import train_step, validation_step
from data.dataset_loader import (
    build_tf_dataset,
    create_pairs_csv,
    discover_pairs_from_directory,
    download_official_subset,
)
from utils.visualization import save_comparison_grid, plot_loss_curves

LOCK_FILE = PROJECT_ROOT / ".training.lock"


def acquire_training_lock() -> bool:
    """Acquire a simple file lock to prevent concurrent training runs."""
    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text().strip())
            # Check if process with PID is actually running
            os.kill(pid, 0)
            return False  # Still running
        except (OSError, ValueError):
            # Stale lock file
            LOCK_FILE.unlink(missing_ok=True)

    LOCK_FILE.write_text(str(os.getpid()))
    atexit.register(release_training_lock)
    return True


def release_training_lock():
    """Release the training process lock."""
    if LOCK_FILE.exists():
        try:
            LOCK_FILE.unlink()
        except OSError:
            pass


def ensure_data_ready(
    cfg: ModelConfig,
    download_if_missing: bool = True,
) -> bool:
    """
    Ensure the pairs CSV manifest exists. If raw data is missing, offer automatic download.
    """
    if PAIRS_CSV_PATH.exists():
        # Verify CSV has entries
        import pandas as pd
        try:
            df = pd.read_csv(PAIRS_CSV_PATH)
            if len(df) > 0:
                return True
        except Exception:
            pass

    # Manifest not found or empty; scan raw directory
    pairs, stats = discover_pairs_from_directory(RAW_DATA_DIR)
    if stats["valid_pairs"] == 0 and download_if_missing:
        print("\n[Notice] No local dataset found. Downloading official SketchyCOCO paired subset...")
        download_official_subset(
            RAW_DATA_DIR,
            train_count=cfg.train_limit,
            val_count=cfg.validation_limit,
        )
        pairs, stats = discover_pairs_from_directory(RAW_DATA_DIR)

    if stats["valid_pairs"] > 0:
        create_pairs_csv(
            RAW_DATA_DIR,
            PAIRS_CSV_PATH,
            train_limit=cfg.train_limit,
            val_limit=cfg.validation_limit,
            seed=cfg.seed,
        )
        return True

    return False


def train_model(
    config: ModelConfig,
    resume: bool = False,
    max_duration_seconds: Optional[float] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Main training execution loop.

    Args:
        config: ModelConfig instance with hyperparameters
        resume: Whether to restore from latest checkpoint
        max_duration_seconds: Soft time limit in seconds; stops gracefully between epochs
        progress_callback: Optional callback for UI progress updates

    Returns:
        Summary dict of training run results
    """
    set_seed(config.seed)
    configure_tf_cpu(config)

    print("=" * 65)
    print("      Sketch-to-Photo Landscape cGAN Training Engine")
    print("=" * 65)
    print(f"Target Resolution: {config.image_size}x{config.image_size} (3 channels)")
    print(f"Batch Size:        {config.batch_size}")
    print(f"Learning Rate:     {config.learning_rate} (Beta 1: {config.beta_1})")
    print(f"L1 Lambda Weight:  {config.l1_lambda}")
    print(f"Max Epochs:        {config.max_epochs}")
    print(f"Time Budget:       {format_duration(max_duration_seconds) if max_duration_seconds else 'Unlimited'}")

    # Ensure dataset manifest is populated
    ready = ensure_data_ready(config, download_if_missing=True)
    if not ready:
        raise FileNotFoundError(
            "Training halted: dataset is missing. Run python -m utils.dataset_inspection --download-if-missing"
        )

    # Build tf.data streaming pipelines
    print("\nBuilding streaming tf.data datasets...")
    train_ds = build_tf_dataset(
        manifest_csv=PAIRS_CSV_PATH,
        split="train",
        batch_size=config.batch_size,
        image_size=config.image_size,
        shuffle=True,
        augment=True,
        limit=config.train_limit,
    )
    val_ds = build_tf_dataset(
        manifest_csv=PAIRS_CSV_PATH,
        split="val",
        batch_size=config.batch_size,
        image_size=config.image_size,
        shuffle=False,
        augment=False,
        limit=config.validation_limit,
    )

    # Cache fixed validation sample batch for consistent visual progress comparison
    fixed_val_sketches = []
    fixed_val_photos = []
    for s_val, p_val in val_ds.take(4):
        fixed_val_sketches.append(s_val)
        fixed_val_photos.append(p_val)
    fixed_val_sketches = tf.concat(fixed_val_sketches, axis=0) if fixed_val_sketches else None
    fixed_val_photos = tf.concat(fixed_val_photos, axis=0) if fixed_val_photos else None

    # Instantiate Models from scratch (Strictly Random Initialization)
    print("\nInstantiating U-Net Generator and PatchGAN Discriminator from scratch...")
    generator = build_generator(
        input_shape=(config.image_size, config.image_size, config.channels),
        encoder_filters=config.generator_filters,
    )
    discriminator = build_discriminator(
        image_shape=(config.image_size, config.image_size, config.channels),
        filters=config.discriminator_filters,
    )

    # Optimizers
    gen_optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.learning_rate, beta_1=config.beta_1
    )
    disc_optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.learning_rate, beta_1=config.beta_1
    )

    # Checkpoint Manager
    ckpt_wrapper = CheckpointManagerWrapper(
        generator=generator,
        discriminator=discriminator,
        gen_optimizer=gen_optimizer,
        disc_optimizer=disc_optimizer,
    )

    start_epoch = 1
    if resume:
        restored = ckpt_wrapper.load_checkpoint(prefer_best=False)
        if restored > 0:
            start_epoch = restored + 1
            print(f"Resuming training from epoch {start_epoch}...")

    # Metrics & TensorBoard Tracker
    metrics_tracker = TrainingMetricsTracker(
        experiment_name=f"cgan_{config.image_size}x{config.image_size}"
    )

    total_training_start = time.time()
    history = []

    print("\n" + "-" * 65)
    print(f"Beginning Training from Epoch {start_epoch} to {config.max_epochs}...")
    print("-" * 65)

    try:
        for epoch in range(start_epoch, config.max_epochs + 1):
            epoch_start = time.time()

            # Batch accumulators
            running_metrics = {
                "gen_total_loss": 0.0,
                "gen_gan_loss": 0.0,
                "gen_l1_loss": 0.0,
                "disc_loss": 0.0,
            }
            batch_count = 0

            # Training epoch
            for sketch_batch, photo_batch in train_ds:
                step_metrics = train_step(
                    generator=generator,
                    discriminator=discriminator,
                    gen_optimizer=gen_optimizer,
                    disc_optimizer=disc_optimizer,
                    sketch=sketch_batch,
                    photo_target=photo_batch,
                    l1_lambda=config.l1_lambda,
                )
                for k in running_metrics:
                    running_metrics[k] += float(step_metrics[k].numpy())
                batch_count += 1

                if batch_count % 50 == 0:
                    sys.stdout.write(
                        f"\rEpoch [{epoch}/{config.max_epochs}] Batch {batch_count} "
                        f"| Gen: {running_metrics['gen_total_loss']/batch_count:.4f} "
                        f"| L1: {running_metrics['gen_l1_loss']/batch_count:.4f} "
                        f"| Disc: {running_metrics['disc_loss']/batch_count:.4f}"
                    )
                    sys.stdout.flush()

            # Average training metrics
            avg_train = {k: v / max(1, batch_count) for k, v in running_metrics.items()}

            # Validation evaluation
            val_metrics = {
                "val_gen_total_loss": 0.0,
                "val_gen_gan_loss": 0.0,
                "val_gen_l1_loss": 0.0,
                "val_disc_loss": 0.0,
            }
            val_batches = 0
            for v_sketch, v_photo in val_ds:
                v_step = validation_step(
                    generator=generator,
                    discriminator=discriminator,
                    sketch=v_sketch,
                    photo_target=v_photo,
                    l1_lambda=config.l1_lambda,
                )
                for k in val_metrics:
                    val_metrics[k] += float(v_step[k].numpy())
                val_batches += 1

            avg_val = {k: v / max(1, val_batches) for k, v in val_metrics.items()}

            epoch_duration = time.time() - epoch_start
            combined_metrics = {**avg_train, **avg_val}
            metrics_tracker.log_epoch(epoch, combined_metrics, epoch_duration)
            history.append({"epoch": epoch, **combined_metrics, "duration": epoch_duration})

            # Print Epoch Summary
            print(
                f"\rEpoch {epoch:2d}/{config.max_epochs:2d} finished in {format_duration(epoch_duration)} "
                f"| Gen Loss: {combined_metrics['gen_total_loss']:.4f} "
                f"| L1 Loss: {combined_metrics['gen_l1_loss']:.4f} "
                f"| Disc Loss: {combined_metrics['disc_loss']:.4f} "
                f"| Val L1: {combined_metrics['val_gen_l1_loss']:.4f}"
            )

            # Generate and save validation samples
            if fixed_val_sketches is not None:
                generated_samples = generator(fixed_val_sketches, training=False)
                sample_file = SAMPLES_DIR / f"val_sample_epoch_{epoch:03d}.png"
                save_comparison_grid(
                    sketches=fixed_val_sketches,
                    photos=fixed_val_photos,
                    generated=generated_samples,
                    output_path=sample_file,
                    num_samples=4,
                    title=f"Validation Predictions - Epoch {epoch}",
                )

            # Save Checkpoint
            ckpt_wrapper.save_checkpoint(epoch=epoch, metrics=combined_metrics)

            # Update loss curves plot
            plot_loss_curves(history, OUTPUT_DIR / "loss_curves.png")

            # Callback for UI if provided
            if progress_callback:
                progress_callback({
                    "epoch": epoch,
                    "max_epochs": config.max_epochs,
                    "metrics": combined_metrics,
                    "epoch_duration": epoch_duration,
                    "total_duration": time.time() - total_training_start,
                })

            # Time Budget Check
            total_elapsed = time.time() - total_training_start
            if max_duration_seconds and total_elapsed >= max_duration_seconds:
                print(
                    f"\n[Time Budget Reached] Elapsed: {format_duration(total_elapsed)} "
                    f">= Budget: {format_duration(max_duration_seconds)}. Gracefully stopping."
                )
                break

    except KeyboardInterrupt:
        print("\n[Interrupted] Training safely stopped by user. Checkpoint is preserved.")
    finally:
        total_time = time.time() - total_training_start
        print("\n" + "=" * 65)
        print(f"Training completed {len(history)} epoch(s) in {format_duration(total_time)}.")
        print(f"Latest Checkpoint:  {ckpt_wrapper.latest_manager.latest_checkpoint}")
        print(f"Best Checkpoint:    {ckpt_wrapper.best_manager.latest_checkpoint}")
        print(f"Validation Samples: {SAMPLES_DIR}")
        print("=" * 65)

    return {
        "completed_epochs": len(history),
        "total_duration_sec": total_time,
        "history": history,
        "latest_checkpoint": ckpt_wrapper.latest_manager.latest_checkpoint,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Sketch-to-Photo cGAN.")
    parser.add_argument(
        "--mode",
        choices=["pilot", "main"],
        default="main",
        help="Training mode: 'pilot' (64x64 fast verification) or 'main' (128x128)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest available checkpoint",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Maximum training duration in minutes before graceful stop",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Override train pairs limit",
    )
    args = parser.parse_args()

    # Lock check
    if not acquire_training_lock():
        print("Error: Another training process is currently active (.training.lock exists).")
        sys.exit(1)

    # Select configuration based on mode
    cfg = PILOT_CONFIG if args.mode == "pilot" else DEFAULT_CONFIG
    if args.epochs:
        cfg.max_epochs = args.epochs
    if args.train_limit:
        cfg.train_limit = args.train_limit

    max_sec = args.max_duration * 60.0 if args.max_duration else None

    try:
        train_model(
            config=cfg,
            resume=args.resume,
            max_duration_seconds=max_sec,
        )
    finally:
        release_training_lock()


if __name__ == "__main__":
    main()
