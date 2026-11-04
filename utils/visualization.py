"""
Visualization Utilities for Sketch-to-Photo Landscape cGAN.

Features:
- Comparison Grids: Input Sketch | Real Photograph | Generated Photograph
- Epoch validation sample saving
- Loss curves plotting and saving
"""

from pathlib import Path
from typing import Union, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for CLI and background workers
import matplotlib.pyplot as plt
import tensorflow as tf

from preprocessing.image_processing import denormalize_from_gan_range


def save_comparison_grid(
    sketches: Union[np.ndarray, tf.Tensor],
    photos: Union[np.ndarray, tf.Tensor],
    generated: Union[np.ndarray, tf.Tensor],
    output_path: Path,
    num_samples: int = 4,
    title: Optional[str] = None,
) -> None:
    """
    Generate and save a visual comparison grid:
    Column 1: Input Sketch
    Column 2: Ground Truth Photograph
    Column 3: Generated Landscape Photograph
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(sketches, tf.Tensor):
        sketches = sketches.numpy()
    if isinstance(photos, tf.Tensor):
        photos = photos.numpy()
    if isinstance(generated, tf.Tensor):
        generated = generated.numpy()

    n = min(len(sketches), num_samples)
    fig, axes = plt.subplots(n, 3, figsize=(10, 3.2 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["Input Sketch", "Real Photograph", "Generated Photo"]

    for i in range(n):
        s_img = denormalize_from_gan_range(sketches[i])
        p_img = denormalize_from_gan_range(photos[i])
        g_img = denormalize_from_gan_range(generated[i])

        imgs = [s_img, p_img, g_img]
        for j in range(3):
            ax = axes[i, j]
            ax.imshow(imgs[j])
            ax.axis("off")
            if i == 0:
                ax.set_title(col_titles[j], fontsize=13, fontweight="bold", pad=8)

    if title:
        plt.suptitle(title, fontsize=14, fontweight="bold", y=0.99)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(
    history: List[dict],
    output_path: Path,
    title: str = "cGAN Training Loss Trajectories",
) -> None:
    """
    Plot and save training and validation loss curves.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not history:
        return

    epochs = [h["epoch"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Generator Losses
    if "gen_total_loss" in history[0]:
        ax1.plot(epochs, [h.get("gen_total_loss") for h in history], "b-o", label="Total Gen Loss")
    if "gen_l1_loss" in history[0]:
        ax1.plot(epochs, [h.get("gen_l1_loss") for h in history], "g--s", label="L1 Loss (x100)")
    if "gen_gan_loss" in history[0]:
        ax1.plot(epochs, [h.get("gen_gan_loss") for h in history], "m:^", label="Adversarial Loss")
    if "val_gen_l1_loss" in history[0] and history[0].get("val_gen_l1_loss") is not None:
        ax1.plot(epochs, [h.get("val_gen_l1_loss") for h in history], "r-.*", label="Val L1 Loss")

    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title("Generator Loss Breakdown", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Discriminator Losses
    if "disc_loss" in history[0]:
        ax2.plot(epochs, [h.get("disc_loss") for h in history], "orange", marker="o", label="Disc Loss")
    if "val_disc_loss" in history[0] and history[0].get("val_disc_loss") is not None:
        ax2.plot(epochs, [h.get("val_disc_loss") for h in history], "red", linestyle="--", marker="x", label="Val Disc Loss")

    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.set_title("Discriminator Loss", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
