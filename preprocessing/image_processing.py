"""
Image Preprocessing and Augmentation Module.

Implements strict paired transformations:
- Converts grayscale or RGBA to 3-channel RGB
- Resizes both sketch and photograph using identical geometry and aspect ratio preservation
- Normalizes pixel values to [-1.0, 1.0] for GAN training
- Synchronized geometric augmentations (e.g., synchronized horizontal flip)
- Reversible denormalization to [0, 255] uint8 for display and saving
"""

from typing import Tuple, Union
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image


def load_image(image_input: Union[str, Path, Image.Image, np.ndarray]) -> np.ndarray:
    """
    Load an image from a filepath, PIL Image, or NumPy array, ensuring 3-channel RGB.

    Args:
        image_input: Path to image file, PIL Image, or numpy array.

    Returns:
        np.ndarray: RGB image with uint8 values in range [0, 255], shape (H, W, 3).
    """
    if isinstance(image_input, (str, bytes, Path)):
        pil_img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    elif isinstance(image_input, np.ndarray):
        if image_input.dtype != np.uint8:
            if image_input.max() <= 1.0:
                image_input = (image_input * 255.0).astype(np.uint8)
            else:
                image_input = image_input.astype(np.uint8)
        if image_input.ndim == 2:
            return np.stack([image_input] * 3, axis=-1)
        elif image_input.ndim == 3:
            if image_input.shape[-1] == 1:
                return np.repeat(image_input, 3, axis=-1)
            elif image_input.shape[-1] == 4:
                return image_input[:, :, :3]
            elif image_input.shape[-1] == 3:
                return image_input
        raise ValueError(f"Unsupported array shape: {image_input.shape}")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Ensure RGB
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    return np.array(pil_img, dtype=np.uint8)


def resize_and_pad(
    image: np.ndarray,
    target_size: Tuple[int, int] = (128, 128),
    pad_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Resize image preserving aspect ratio with symmetric letterbox padding.

    Args:
        image: np.ndarray shape (H, W, 3) in [0, 255]
        target_size: (target_height, target_width)
        pad_color: RGB fill color for padding (default white for sketches)

    Returns:
        np.ndarray shape (target_height, target_width, 3)
    """
    target_h, target_w = target_size
    orig_h, orig_w = image.shape[:2]

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    # Resize with high quality bicubic interpolation
    pil_img = Image.fromarray(image)
    resized_pil = pil_img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    resized_arr = np.array(resized_pil)

    # Create padded canvas
    canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized_arr

    return canvas


def normalize_to_gan_range(image: np.ndarray) -> np.ndarray:
    """
    Normalize image from [0, 255] uint8 or float to [-1.0, 1.0] float32.
    """
    arr = image.astype(np.float32)
    # Range [0, 255] -> [-1, 1]
    return (arr / 127.5) - 1.0


def denormalize_from_gan_range(image: Union[np.ndarray, tf.Tensor]) -> np.ndarray:
    """
    Convert image from [-1.0, 1.0] float32 back to [0, 255] uint8 RGB.
    """
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    arr = np.clip((image + 1.0) * 127.5, 0.0, 255.0)
    return arr.astype(np.uint8)


def preprocess_sketch(
    image_input: Union[str, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (128, 128)
) -> np.ndarray:
    """
    Load, resize with aspect-ratio preservation, and normalize sketch image.

    Returns:
        np.ndarray: shape (target_h, target_w, 3) in float32 [-1.0, 1.0]
    """
    img = load_image(image_input)
    # Sketches are drawn on white background, pad with white
    padded = resize_and_pad(img, target_size=target_size, pad_color=(255, 255, 255))
    return normalize_to_gan_range(padded)


def preprocess_photo(
    image_input: Union[str, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (128, 128)
) -> np.ndarray:
    """
    Load, resize with identical geometry, and normalize photograph.

    Returns:
        np.ndarray: shape (target_h, target_w, 3) in float32 [-1.0, 1.0]
    """
    img = load_image(image_input)
    # Use edge reflection or mirror padding to preserve photo aesthetics
    target_h, target_w = target_size
    orig_h, orig_w = img.shape[:2]
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    pil_img = Image.fromarray(img)
    resized_pil = pil_img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    resized_arr = np.array(resized_pil)

    # Pad with edge/replicate pixels to avoid artificial black/white borders on photos
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        padded = np.pad(
            resized_arr,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="edge"
        )
    else:
        padded = resized_arr

    return normalize_to_gan_range(padded)


def preprocess_pair(
    sketch_input: Union[str, Image.Image, np.ndarray],
    photo_input: Union[str, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (128, 128)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess both sketch and photograph, guaranteeing alignment and shape consistency.
    """
    sketch = preprocess_sketch(sketch_input, target_size=target_size)
    photo = preprocess_photo(photo_input, target_size=target_size)
    return sketch, photo


def augment_pair(
    sketch: tf.Tensor,
    photo: tf.Tensor,
    seed: int = None
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Apply synchronized geometric augmentations to both sketch and photo.
    Never augment sketch and photo independently, preserving correspondence.

    Currently applies synchronized random horizontal flipping.
    """
    if tf.random.uniform([], seed=seed) > 0.5:
        sketch = tf.image.flip_left_right(sketch)
        photo = tf.image.flip_left_right(photo)
    return sketch, photo
