"""
Landscape Image Generation / Inference Module.

Inference Pipeline:
1. Load unseen input sketch (filepath, PIL Image, or NumPy array)
2. Preprocess with exact training pipeline (aspect ratio padding, 3-channel RGB, resize, [-1, 1] norm)
3. Expand batch dimension: (1, H, W, 3)
4. Execute trained generator forward pass with training=False
5. Denormalize tensor from [-1, 1] to [0, 255] uint8 RGB
6. Return generated PIL Image, metadata, and optionally save to disk
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Union, Tuple, Optional
import numpy as np
from PIL import Image
import tensorflow as tf

from config import IMAGE_SIZE, GENERATED_DIR, CHECKPOINT_DIR
from preprocessing.image_processing import (
    preprocess_sketch,
    denormalize_from_gan_range,
    load_image,
)
from training.checkpoint import load_trained_generator


def generate_landscape(
    sketch_input: Union[str, Path, Image.Image, np.ndarray],
    generator: Optional[tf.keras.Model] = None,
    checkpoint_or_weights_path: Optional[str] = None,
    image_size: int = IMAGE_SIZE,
    output_path: Optional[Union[str, Path]] = None,
) -> Tuple[Image.Image, float]:
    """
    Generate an RGB landscape photograph from an input sketch using the trained generator.

    Args:
        sketch_input: Input sketch path, PIL image, or numpy array
        generator: Preloaded generator model instance (or loaded automatically if None)
        checkpoint_or_weights_path: Optional specific checkpoint path
        image_size: Input resolution expected by generator
        output_path: Optional destination to save output photograph

    Returns:
        Tuple of (generated_pil_image, elapsed_inference_seconds)
    """
    if generator is None:
        generator = load_trained_generator(
            weights_or_ckpt_path=checkpoint_or_weights_path,
            image_size=image_size,
        )

    # 1. Preprocess Sketch to [-1, 1] float32 array
    start_time = time.time()
    sketch_norm = preprocess_sketch(sketch_input, target_size=(image_size, image_size))

    # 2. Add batch dimension -> (1, H, W, 3)
    sketch_tensor = tf.convert_to_tensor(sketch_norm[np.newaxis, ...], dtype=tf.float32)

    # 3. Generator Inference (strictly with training=False)
    gen_tensor = generator(sketch_tensor, training=False)

    # 4. Denormalize to [0, 255] uint8
    gen_array = denormalize_from_gan_range(gen_tensor[0])
    inference_time = time.time() - start_time

    # 5. Convert to PIL Image
    output_pil = Image.fromarray(gen_array, mode="RGB")

    # 6. Save if output path provided
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        output_pil.save(out_p)
        print(f"Generated landscape saved to: {out_p.resolve()} (Time: {inference_time*1000:.1f}ms)")

    return output_pil, inference_time


def generate_image(
    sketch_path: Union[str, Path],
    generator: Optional[tf.keras.Model] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Image.Image:
    """Convenience alias matching specification."""
    img, _ = generate_landscape(
        sketch_input=sketch_path,
        generator=generator,
        output_path=output_path,
    )
    return img


def main():
    parser = argparse.ArgumentParser(description="Generate landscape photograph from sketch.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input sketch image")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output destination for generated photo")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Optional checkpoint path or directory")
    parser.add_argument("--size", type=int, default=IMAGE_SIZE, help="Image resolution")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input sketch file does not exist: {input_path}")
        sys.exit(1)

    out_dest = args.output
    if not out_dest:
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        out_dest = GENERATED_DIR / f"generated_{input_path.stem}.png"

    try:
        generate_landscape(
            sketch_input=input_path,
            checkpoint_or_weights_path=args.checkpoint,
            image_size=args.size,
            output_path=out_dest,
        )
    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
