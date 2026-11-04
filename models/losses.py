"""
GAN and Reconstruction Loss Formulations for Conditional GAN (Pix2Pix).

Equations:
- Discriminator Loss:
    L_D = 0.5 * [ E[ log D(x, y) ] + E[ log(1 - D(x, G(x))) ] ]
    Implemented with BinaryCrossentropy(from_logits=True):
    real_loss = BCE(ones_like(D(x, y)), D(x, y))
    fake_loss = BCE(zeros_like(D(x, G(x))), D(x, G(x)))
    disc_loss = 0.5 * (real_loss + fake_loss)

- Generator Loss:
    L_G = L_GAN(G, D) + lambda * L1(G)
    gan_loss = BCE(ones_like(D(x, G(x))), D(x, G(x)))
    l1_loss  = E[ ||y - G(x)||_1 ]
    total_gen_loss = gan_loss + (lambda * l1_loss)
"""

from typing import Tuple
import tensorflow as tf

_bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(
    disc_generated_output: tf.Tensor,
    gen_output: tf.Tensor,
    target: tf.Tensor,
    l1_lambda: float = 100.0,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Compute generator loss composed of adversarial loss and weighted L1 reconstruction loss.

    Args:
        disc_generated_output: Discriminator logits on generated pair D(sketch, generated)
        gen_output: Generated photograph G(sketch) in [-1, 1]
        target: Ground truth real photograph in [-1, 1]
        l1_lambda: Weight for L1 pixel difference term (default: 100.0)

    Returns:
        total_gen_loss: Combined scalar tensor
        gan_loss: Adversarial scalar tensor
        l1_loss: Mean absolute error scalar tensor
    """
    # Adversarial: wants discriminator to classify generated images as real (all ones)
    gan_loss = _bce_loss(tf.ones_like(disc_generated_output), disc_generated_output)

    # Pixel-level L1 reconstruction loss
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    # Total combined generator loss
    total_gen_loss = gan_loss + (l1_lambda * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(
    disc_real_output: tf.Tensor,
    disc_generated_output: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Compute PatchGAN discriminator loss distinguishing real from fake pairs.

    Args:
        disc_real_output: Discriminator logits on real pair D(sketch, real_photo)
        disc_generated_output: Discriminator logits on fake pair D(sketch, generated_photo)

    Returns:
        total_disc_loss: 0.5 * (real_loss + fake_loss)
        real_loss: Scalar loss on real samples
        fake_loss: Scalar loss on fake samples
    """
    # Real pairs: target is 1
    real_loss = _bce_loss(tf.ones_like(disc_real_output), disc_real_output)

    # Generated pairs: target is 0
    fake_loss = _bce_loss(tf.zeros_like(disc_generated_output), disc_generated_output)

    # Standard Pix2Pix averages real and fake losses to balance gradient magnitudes
    total_disc_loss = 0.5 * (real_loss + fake_loss)

    return total_disc_loss, real_loss, fake_loss
