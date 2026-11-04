"""
Model Evaluation Module using Held-Out Validation Pairs.

Computes:
- Mean Absolute Error (MAE)
- Peak Signal-to-Noise Ratio (PSNR) via skimage.metrics
- Structural Similarity Index (SSIM) via skimage.metrics
- Strictly operates in [0, 255] uint8 RGB space
- Exports detailed per-pair and aggregate metrics to outputs/evaluation_metrics.csv
- Generates side-by-side comparison grids: Input Sketch | Real Photo | Generated Photo
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import tensorflow as tf

from config import (
    PAIRS_CSV_PATH,
    METRICS_CSV_PATH,
    OUTPUT_DIR,
    SAMPLES_DIR,
    IMAGE_SIZE,
    VALIDATION_LIMIT,
)
from training.checkpoint import load_trained_generator
from preprocessing.image_processing import (
    preprocess_sketch,
    load_image,
    resize_and_pad,
    denormalize_from_gan_range,
)
from utils.visualization import save_comparison_grid


def evaluate_pairs(
    generator: Optional[tf.keras.Model] = None,
    manifest_csv: Path = PAIRS_CSV_PATH,
    output_metrics_csv: Path = METRICS_CSV_PATH,
    max_samples: int = VALIDATION_LIMIT,
    image_size: int = IMAGE_SIZE,
    save_grid: bool = True,
) -> Dict[str, Any]:
    """
    Run evaluation against held-out validation pairs.
    """
    manifest_csv = Path(manifest_csv)
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")

    df = pd.read_csv(manifest_csv)
    val_df = df[df["split"] == "val"]
    if len(val_df) == 0:
        val_df = df  # fallback if no separate split labeled

    if max_samples and max_samples > 0:
        val_df = val_df.head(max_samples)

    if generator is None:
        generator = load_trained_generator(image_size=image_size)

    records: List[Dict[str, Any]] = []
    grid_sketches = []
    grid_photos = []
    grid_generated = []

    print(f"Evaluating generator on {len(val_df)} held-out validation pairs...")

    for i, (_, row) in enumerate(val_df.iterrows(), 1):
        s_path = Path(row["sketch_path"])
        p_path = Path(row["photo_path"])

        if not s_path.exists() or not p_path.exists():
            continue

        # 1. Preprocess input sketch
        sketch_norm = preprocess_sketch(s_path, target_size=(image_size, image_size))
        s_tensor = tf.convert_to_tensor(sketch_norm[np.newaxis, ...], dtype=tf.float32)

        # 2. Generator forward pass
        gen_tensor = generator(s_tensor, training=False)
        gen_uint8 = denormalize_from_gan_range(gen_tensor[0])

        # 3. Load ground truth photo and bring to same size in [0, 255]
        photo_raw = load_image(p_path)
        photo_uint8 = resize_and_pad(photo_raw, target_size=(image_size, image_size), pad_color=(0, 0, 0))

        # 4. Metric calculations in [0, 255] uint8 RGB range
        # MAE: Mean absolute pixel difference
        mae = float(np.mean(np.abs(gen_uint8.astype(np.float32) - photo_uint8.astype(np.float32))))

        # PSNR: Peak Signal to Noise Ratio
        psnr_val = float(psnr_metric(photo_uint8, gen_uint8, data_range=255))

        # SSIM: Structural Similarity Index (multichannel=True / channel_axis=2)
        try:
            ssim_val = float(ssim_metric(photo_uint8, gen_uint8, channel_axis=2, data_range=255))
        except TypeError:
            ssim_val = float(ssim_metric(photo_uint8, gen_uint8, multichannel=True, data_range=255))

        records.append({
            "pair_id": s_path.name,
            "sketch_path": str(s_path),
            "photo_path": str(p_path),
            "mae": round(mae, 4),
            "psnr_db": round(psnr_val, 4),
            "ssim": round(ssim_val, 4),
        })

        if len(grid_sketches) < 6:
            grid_sketches.append(sketch_norm)
            grid_photos.append((photo_uint8.astype(np.float32) / 127.5) - 1.0)
            grid_generated.append((gen_uint8.astype(np.float32) / 127.5) - 1.0)

        if i % 25 == 0 or i == len(val_df):
            print(f"Evaluated {i}/{len(val_df)} pairs...")

    if not records:
        raise RuntimeError("No valid pairs were evaluated.")

    results_df = pd.DataFrame(records)

    # Compute aggregate summary statistics
    mean_mae = float(results_df["mae"].mean())
    mean_psnr = float(results_df["psnr_db"].mean())
    mean_ssim = float(results_df["ssim"].mean())

    output_metrics_csv = Path(output_metrics_csv)
    output_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_metrics_csv, index=False)

    print("\n" + "=" * 60)
    print("                EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Evaluated Pairs:   {len(results_df)}")
    print(f"Mean Absolute Error (MAE): {mean_mae:.4f}  (Lower is closer to target)")
    print(f"Mean PSNR:                 {mean_psnr:.2f} dB (Higher is closer)")
    print(f"Mean SSIM:                 {mean_ssim:.4f}  (1.0 is identical structure)")
    print(f"Detailed Metrics Saved:    {output_metrics_csv.resolve()}")
    print("=" * 60)
    print("Note: Pixel-level metrics reflect mathematical similarity to the ground-truth")
    print("target photo. GAN-generated landscapes may look visually compelling while having")
    print("modest PSNR/SSIM due to multi-modal variations in foliage, sky textures, and lighting.")

    if save_grid and grid_sketches:
        grid_file = OUTPUT_DIR / "evaluation_comparison_grid.png"
        save_comparison_grid(
            sketches=np.array(grid_sketches),
            photos=np.array(grid_photos),
            generated=np.array(grid_generated),
            output_path=grid_file,
            num_samples=len(grid_sketches),
            title="Model Evaluation: Input Sketch | Target Photo | Generated Photo",
        )
        print(f"Saved evaluation comparison grid to: {grid_file.resolve()}")

    return {
        "num_evaluated": len(results_df),
        "mean_mae": mean_mae,
        "mean_psnr": mean_psnr,
        "mean_ssim": mean_ssim,
        "metrics_csv": str(output_metrics_csv),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained cGAN generator.")
    parser.add_argument("--manifest", type=str, default=str(PAIRS_CSV_PATH), help="Path to pairs.csv")
    parser.add_argument("--output", type=str, default=str(METRICS_CSV_PATH), help="Output metrics CSV")
    parser.add_argument("--samples", type=int, default=VALIDATION_LIMIT, help="Max validation samples")
    parser.add_argument("--size", type=int, default=IMAGE_SIZE, help="Image resolution")
    args = parser.parse_args()

    try:
        evaluate_pairs(
            manifest_csv=Path(args.manifest),
            output_metrics_csv=Path(args.output),
            max_samples=args.samples,
            image_size=args.size,
        )
    except Exception as e:
        print(f"\n[Evaluation Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
