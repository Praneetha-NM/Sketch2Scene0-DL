"""
TensorFlow GradientTape Training and Validation Steps.

Executes atomic forward and backward passes:
- Dual GradientTape for simultaneous backpropagation
- Independent Adam optimizer updates
- Tracks total generator loss, adversarial loss, L1 reconstruction loss, and discriminator loss
"""

from typing import Dict
import tensorflow as tf
from models.losses import generator_loss, discriminator_loss


@tf.function
def train_step(
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    gen_optimizer: tf.keras.optimizers.Optimizer,
    disc_optimizer: tf.keras.optimizers.Optimizer,
    sketch: tf.Tensor,
    photo_target: tf.Tensor,
    l1_lambda: float = 100.0,
) -> Dict[str, tf.Tensor]:
    """
    Execute a single training step with independent gradient updates.

    Args:
        generator: U-Net generator model
        discriminator: PatchGAN discriminator model
        gen_optimizer: Generator Adam optimizer
        disc_optimizer: Discriminator Adam optimizer
        sketch: Input sketch batch (B, H, W, 3) in [-1, 1]
        photo_target: Target ground truth photo batch (B, H, W, 3) in [-1, 1]
        l1_lambda: Weight for L1 loss

    Returns:
        dict: Scalar metric tensors for monitoring
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generator forward pass
        gen_output = generator(sketch, training=True)

        # Discriminator forward pass on real and generated pairs
        disc_real_output = discriminator([sketch, photo_target], training=True)
        disc_generated_output = discriminator([sketch, gen_output], training=True)

        # Compute losses
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            disc_generated_output, gen_output, photo_target, l1_lambda=l1_lambda
        )
        disc_total_loss, disc_real_loss, disc_fake_loss = discriminator_loss(
            disc_real_output, disc_generated_output
        )

    # Compute gradients independently
    generator_gradients = gen_tape.gradient(
        gen_total_loss, generator.trainable_variables
    )
    discriminator_gradients = disc_tape.gradient(
        disc_total_loss, discriminator.trainable_variables
    )

    # Apply gradients
    gen_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )
    disc_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )

    return {
        "gen_total_loss": gen_total_loss,
        "gen_gan_loss": gen_gan_loss,
        "gen_l1_loss": gen_l1_loss,
        "disc_loss": disc_total_loss,
        "disc_real_loss": disc_real_loss,
        "disc_fake_loss": disc_fake_loss,
    }


@tf.function
def validation_step(
    generator: tf.keras.Model,
    discriminator: tf.keras.Model,
    sketch: tf.Tensor,
    photo_target: tf.Tensor,
    l1_lambda: float = 100.0,
) -> Dict[str, tf.Tensor]:
    """
    Evaluate generator and discriminator on held-out validation batch without computing gradients.
    """
    gen_output = generator(sketch, training=False)
    disc_real_output = discriminator([sketch, photo_target], training=False)
    disc_generated_output = discriminator([sketch, gen_output], training=False)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
        disc_generated_output, gen_output, photo_target, l1_lambda=l1_lambda
    )
    disc_total_loss, disc_real_loss, disc_fake_loss = discriminator_loss(
        disc_real_output, disc_generated_output
    )

    return {
        "val_gen_total_loss": gen_total_loss,
        "val_gen_gan_loss": gen_gan_loss,
        "val_gen_l1_loss": gen_l1_loss,
        "val_disc_loss": disc_total_loss,
    }
