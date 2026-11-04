"""
PatchGAN Discriminator Architecture Built from Scratch in TensorFlow/Keras.

Architecture:
- Inputs: [Sketch (H, W, 3), Photo (H, W, 3)]
- Concatenation along channel axis -> (H, W, 6)
- 3 Convolutional Blocks:
    Block 1: Conv2D(64, k=4, s=2), LeakyReLU(0.2) (No BatchNorm)
    Block 2: Conv2D(128, k=4, s=2), BatchNorm, LeakyReLU(0.2)
    Block 3: Conv2D(256, k=4, s=2), BatchNorm, LeakyReLU(0.2)
- Patch-level output layer: Conv2D(1, k=4, s=1, padding='same') without activation (from_logits=True)
- Output: (H/8, W/8, 1) patch validity score map
"""

from typing import List, Tuple
import tensorflow as tf
from tensorflow.keras import layers, Model


def build_discriminator(
    image_shape: Tuple[int, int, int] = (128, 128, 3),
    filters: List[int] = None,
    name: str = "patchgan_discriminator",
) -> Model:
    """
    Construct PatchGAN Discriminator conditioned on input sketch.

    Args:
        image_shape: (height, width, channels) of each individual image
        filters: Convolutional channel counts (default: [64, 128, 256])
        name: Name for the Keras model

    Returns:
        tf.keras.Model: Multi-input model accepting [sketch, target_or_generated]
    """
    if filters is None:
        filters = [64, 128, 256]

    sketch_in = layers.Input(shape=image_shape, name="sketch_condition_input")
    target_in = layers.Input(shape=image_shape, name="photo_candidate_input")

    # Concatenate condition sketch and photo along channel dimension (shape: H, W, 6)
    merged = layers.Concatenate(axis=-1, name="channel_concat")([sketch_in, target_in])

    initializer = tf.random_normal_initializer(0.0, 0.02)

    # Block 1: Conv2D -> LeakyReLU (No BatchNorm on input layer)
    x = layers.Conv2D(
        filters[0],
        kernel_size=4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        use_bias=True,
        name="disc_conv_1",
    )(merged)
    x = layers.LeakyReLU(negative_slope=0.2, name="disc_leaky_1")(x)

    # Block 2: Conv2D -> BatchNorm -> LeakyReLU
    x = layers.Conv2D(
        filters[1],
        kernel_size=4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        use_bias=False,
        name="disc_conv_2",
    )(x)
    x = layers.BatchNormalization(name="disc_bn_2")(x)
    x = layers.LeakyReLU(negative_slope=0.2, name="disc_leaky_2")(x)

    # Block 3: Conv2D -> BatchNorm -> LeakyReLU
    x = layers.Conv2D(
        filters[2],
        kernel_size=4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        use_bias=False,
        name="disc_conv_3",
    )(x)
    x = layers.BatchNormalization(name="disc_bn_3")(x)
    x = layers.LeakyReLU(negative_slope=0.2, name="disc_leaky_3")(x)

    # Final Patch-level Prediction Map (stride 1, 1 filter, linear logits)
    # Strictly NO Sigmoid because loss uses BinaryCrossentropy(from_logits=True)
    patch_output = layers.Conv2D(
        filters=1,
        kernel_size=4,
        strides=1,
        padding="same",
        kernel_initializer=initializer,
        name="patch_logits",
    )(x)

    model = Model(inputs=[sketch_in, target_in], outputs=patch_output, name=name)
    return model


def get_discriminator_summary(model: Model) -> str:
    """Return formatted summary and parameter counts for Discriminator."""
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    total_params = model.count_params()
    trainable_params = sum(
        tf.reduce_prod(v.shape).numpy() for v in model.trainable_variables
    )
    lines.append(f"\nTotal Discriminator Parameters: {total_params:,}")
    lines.append(f"Trainable Parameters: {trainable_params:,}")
    return "\n".join(lines)
