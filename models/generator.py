"""
U-Net Generator Architecture Built from Scratch in TensorFlow/Keras.

Architecture:
- Input: (H, W, 3) Sketch in [-1.0, 1.0]
- 5 Downsampling blocks (Encoder): Conv2D, BatchNormalization, LeakyReLU
- Bottleneck: Conv2D, BatchNormalization, LeakyReLU
- 5 Upsampling blocks (Decoder): Conv2DTranspose, BatchNormalization, Dropout, Concat Skip
- Output: Conv2DTranspose to (H, W, 3) with Tanh activation
- Strictly randomly initialized weights (He/Glorot), no pretrained weights
"""

from typing import List, Tuple
import tensorflow as tf
from tensorflow.keras import layers, Model


def downsample_block(
    filters: int,
    size: int = 4,
    apply_batchnorm: bool = True,
    name: str = "downsample",
) -> layers.Layer:
    """
    Encoder Downsampling Block: Conv2D (stride 2) -> [BatchNorm] -> LeakyReLU.
    """
    initializer = tf.random_normal_initializer(0.0, 0.02)
    result = tf.keras.Sequential(name=name)
    result.add(
        layers.Conv2D(
            filters,
            kernel_size=size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=not apply_batchnorm,
        )
    )
    if apply_batchnorm:
        result.add(layers.BatchNormalization())
    result.add(layers.LeakyReLU(negative_slope=0.2))
    return result


def upsample_block(
    filters: int,
    size: int = 4,
    apply_dropout: bool = False,
    dropout_rate: float = 0.5,
    name: str = "upsample",
) -> layers.Layer:
    """
    Decoder Upsampling Block: Conv2DTranspose (stride 2) -> BatchNorm -> [Dropout] -> ReLU.
    """
    initializer = tf.random_normal_initializer(0.0, 0.02)
    result = tf.keras.Sequential(name=name)
    result.add(
        layers.Conv2DTranspose(
            filters,
            kernel_size=size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )
    result.add(layers.BatchNormalization())
    if apply_dropout:
        result.add(layers.Dropout(dropout_rate))
    result.add(layers.ReLU())
    return result


def build_generator(
    input_shape: Tuple[int, int, int] = (128, 128, 3),
    encoder_filters: List[int] = None,
    name: str = "unet_generator",
) -> Model:
    """
    Construct a complete U-Net Generator with skip connections.

    Args:
        input_shape: (height, width, channels), e.g. (128, 128, 3)
        encoder_filters: List of 5 filter depths (default: [32, 64, 128, 256, 256])
        name: Name for the Keras model

    Returns:
        tf.keras.Model: Configured and compiled generator outputting [-1.0, 1.0]
    """
    if encoder_filters is None:
        encoder_filters = [32, 64, 128, 256, 256]

    inputs = layers.Input(shape=input_shape, name="sketch_input")

    # Encoder (Downsampling)
    # Down 1: No BatchNorm on first layer
    down1 = downsample_block(encoder_filters[0], apply_batchnorm=False, name="down_1")(inputs)
    down2 = downsample_block(encoder_filters[1], apply_batchnorm=True, name="down_2")(down1)
    down3 = downsample_block(encoder_filters[2], apply_batchnorm=True, name="down_3")(down2)
    down4 = downsample_block(encoder_filters[3], apply_batchnorm=True, name="down_4")(down3)
    down5 = downsample_block(encoder_filters[4], apply_batchnorm=True, name="down_5")(down4)

    # Bottleneck
    bottleneck_filters = encoder_filters[-1]
    bottleneck = downsample_block(bottleneck_filters, apply_batchnorm=True, name="bottleneck")(down5)

    # Decoder (Upsampling) with Skip Connections
    up1 = upsample_block(encoder_filters[4], apply_dropout=True, name="up_1")(bottleneck)
    concat1 = layers.Concatenate(name="skip_1")([up1, down5])

    up2 = upsample_block(encoder_filters[3], apply_dropout=True, name="up_2")(concat1)
    concat2 = layers.Concatenate(name="skip_2")([up2, down4])

    up3 = upsample_block(encoder_filters[2], apply_dropout=True, name="up_3")(concat2)
    concat3 = layers.Concatenate(name="skip_3")([up3, down3])

    up4 = upsample_block(encoder_filters[1], apply_dropout=False, name="up_4")(concat3)
    concat4 = layers.Concatenate(name="skip_4")([up4, down2])

    up5 = upsample_block(encoder_filters[0], apply_dropout=False, name="up_5")(concat4)
    concat5 = layers.Concatenate(name="skip_5")([up5, down1])

    # Final Output Layer: Conv2DTranspose with Tanh activation
    initializer = tf.random_normal_initializer(0.0, 0.02)
    outputs = layers.Conv2DTranspose(
        filters=3,
        kernel_size=4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        activation="tanh",
        name="generated_photo",
    )(concat5)

    model = Model(inputs=inputs, outputs=outputs, name=name)
    return model


def get_generator_summary(model: Model) -> str:
    """Return a formatted string summary of model architecture and parameter count."""
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    total_params = model.count_params()
    trainable_params = sum(
        tf.reduce_prod(v.shape).numpy() for v in model.trainable_variables
    )
    lines.append(f"\nTotal Parameters: {total_params:,}")
    lines.append(f"Trainable Parameters: {trainable_params:,}")
    return "\n".join(lines)
