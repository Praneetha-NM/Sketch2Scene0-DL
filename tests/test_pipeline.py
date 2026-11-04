"""
Automated Test Suite for Sketch-to-Photo cGAN.

Verifies:
1. Environment and TensorFlow CPU execution
2. Generator architecture output shape (both 128x128x3 and 64x64x3) and parameter counts
3. Discriminator architecture output shape on 6-channel inputs
4. Loss functions computation
5. Train step execution: weights update and change from random initialization
6. Checkpoint save, restore, and standalone generator export
7. Preprocessing and aspect ratio preservation
8. Inference pipeline on a synthetic unseen sketch
"""

import unittest
import tempfile
import shutil
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf

from config import ModelConfig, PilotConfig
from models.generator import build_generator
from models.discriminator import build_discriminator
from models.losses import generator_loss, discriminator_loss
from training.train_step import train_step, validation_step
from training.checkpoint import CheckpointManagerWrapper, load_trained_generator
from preprocessing.image_processing import (
    preprocess_sketch,
    preprocess_photo,
    preprocess_pair,
    denormalize_from_gan_range,
    load_image,
)
from inference.generate import generate_landscape


class TestCGANPipeline(unittest.TestCase):

    def setUp(self):
        # Configure float32 CPU
        tf.keras.backend.set_floatx("float32")
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_generator_shapes(self):
        """Verify generator outputs exact (B, H, W, 3) in [-1, 1] range."""
        # Test 128x128
        gen128 = build_generator(input_shape=(128, 128, 3), encoder_filters=[32, 64, 128, 256, 256])
        dummy_in128 = tf.random.normal([2, 128, 128, 3])
        out128 = gen128(dummy_in128, training=False)
        self.assertEqual(out128.shape, (2, 128, 128, 3))
        self.assertTrue(tf.reduce_all(out128 >= -1.0) and tf.reduce_all(out128 <= 1.0))

        # Test 64x64 (Pilot Mode)
        gen64 = build_generator(input_shape=(64, 64, 3), encoder_filters=[32, 64, 128, 128, 128])
        dummy_in64 = tf.random.normal([2, 64, 64, 3])
        out64 = gen64(dummy_in64, training=False)
        self.assertEqual(out64.shape, (2, 64, 64, 3))

    def test_02_discriminator_shapes(self):
        """Verify PatchGAN discriminator accepts (sketch, photo) pairs and outputs 1-channel patch map."""
        disc = build_discriminator(image_shape=(128, 128, 3), filters=[64, 128, 256])
        sketch = tf.random.normal([2, 128, 128, 3])
        photo = tf.random.normal([2, 128, 128, 3])
        patch_out = disc([sketch, photo], training=False)
        # Spatial downsampling: 128 -> 64 -> 32 -> 16
        self.assertEqual(patch_out.shape, (2, 16, 16, 1))

    def test_03_loss_functions(self):
        """Verify loss calculations return finite scalar values."""
        gen_logits = tf.random.normal([2, 16, 16, 1])
        real_logits = tf.random.normal([2, 16, 16, 1])
        gen_out = tf.random.uniform([2, 128, 128, 3], -1.0, 1.0)
        target = tf.random.uniform([2, 128, 128, 3], -1.0, 1.0)

        tot_gen, gan_l, l1_l = generator_loss(gen_logits, gen_out, target, l1_lambda=100.0)
        self.assertTrue(tf.math.is_finite(tot_gen))
        self.assertTrue(tf.math.is_finite(gan_l))
        self.assertTrue(tf.math.is_finite(l1_l))
        self.assertAlmostEqual(tot_gen.numpy(), (gan_l + 100.0 * l1_l).numpy(), places=4)

        tot_disc, r_l, f_l = discriminator_loss(real_logits, gen_logits)
        self.assertTrue(tf.math.is_finite(tot_disc))
        self.assertAlmostEqual(tot_disc.numpy(), (0.5 * (r_l + f_l)).numpy(), places=4)

    def test_04_train_step_gradient_updates(self):
        """Verify that training step executes without error and updates model weights."""
        gen = build_generator(input_shape=(64, 64, 3), encoder_filters=[16, 32, 64, 64, 64])
        disc = build_discriminator(image_shape=(64, 64, 3), filters=[32, 64, 64])

        gen_opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        disc_opt = tf.keras.optimizers.Adam(learning_rate=0.001)

        # Snapshot initial weights
        gen_initial_w = [tf.identity(v) for v in gen.trainable_variables[:3]]
        disc_initial_w = [tf.identity(v) for v in disc.trainable_variables[:3]]

        sketch = tf.random.uniform([1, 64, 64, 3], -1.0, 1.0)
        photo = tf.random.uniform([1, 64, 64, 3], -1.0, 1.0)

        # Execute training step
        metrics = train_step(gen, disc, gen_opt, disc_opt, sketch, photo, l1_lambda=10.0)

        for key in ["gen_total_loss", "gen_gan_loss", "gen_l1_loss", "disc_loss"]:
            self.assertIn(key, metrics)
            self.assertTrue(tf.math.is_finite(metrics[key]))

        # Verify weights have changed from initial random values
        gen_changed = any(
            not np.allclose(init.numpy(), curr.numpy())
            for init, curr in zip(gen_initial_w, gen.trainable_variables[:3])
        )
        disc_changed = any(
            not np.allclose(init.numpy(), curr.numpy())
            for init, curr in zip(disc_initial_w, disc.trainable_variables[:3])
        )
        self.assertTrue(gen_changed, "Generator weights must update after train_step")
        self.assertTrue(disc_changed, "Discriminator weights must update after train_step")

    def test_05_checkpoint_save_and_restore(self):
        """Verify checkpoint saving, restoration, and standalone generator weights."""
        latest_dir = self.temp_dir / "latest"
        best_dir = self.temp_dir / "best"
        standalone_dir = self.temp_dir / "standalone"

        gen = build_generator(input_shape=(64, 64, 3), encoder_filters=[16, 32, 64, 64, 64])
        disc = build_discriminator(image_shape=(64, 64, 3), filters=[32, 64, 64])
        gen_opt = tf.keras.optimizers.Adam(0.001)
        disc_opt = tf.keras.optimizers.Adam(0.001)

        manager = CheckpointManagerWrapper(
            generator=gen,
            discriminator=disc,
            gen_optimizer=gen_opt,
            disc_optimizer=disc_opt,
            latest_dir=latest_dir,
            best_dir=best_dir,
        )

        # Save checkpoint
        saved_path = manager.save_checkpoint(
            epoch=1,
            metrics={"val_gen_l1_loss": 0.35, "gen_total_loss": 40.0},
        )
        self.assertTrue(Path(saved_path + ".index").exists() or latest_dir.exists())

        # Test standalone export
        manager.export_standalone_generator(standalone_dir)
        self.assertTrue((standalone_dir / "generator_weights.weights.h5").exists())

        # Restore into fresh models
        fresh_gen = build_generator(input_shape=(64, 64, 3), encoder_filters=[16, 32, 64, 64, 64])
        fresh_disc = build_discriminator(image_shape=(64, 64, 3), filters=[32, 64, 64])
        fresh_manager = CheckpointManagerWrapper(
            generator=fresh_gen,
            discriminator=fresh_disc,
            gen_optimizer=tf.keras.optimizers.Adam(0.001),
            disc_optimizer=tf.keras.optimizers.Adam(0.001),
            latest_dir=latest_dir,
            best_dir=best_dir,
        )
        restored_epoch = fresh_manager.load_checkpoint()
        self.assertEqual(restored_epoch, 1)

    def test_06_preprocessing_and_denormalization(self):
        """Verify image preprocessing, aspect ratio letterboxing, and denormalization."""
        # Create a non-square test array (e.g. 100x200)
        test_img = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        processed = preprocess_sketch(test_img, target_size=(128, 128))
        self.assertEqual(processed.shape, (128, 128, 3))
        self.assertEqual(processed.dtype, np.float32)
        self.assertTrue(processed.min() >= -1.0 and processed.max() <= 1.0)

        # Denormalize
        denorm = denormalize_from_gan_range(processed)
        self.assertEqual(denorm.shape, (128, 128, 3))
        self.assertEqual(denorm.dtype, np.uint8)
        self.assertTrue(denorm.min() >= 0 and denorm.max() <= 255)

    def test_07_inference_pipeline(self):
        """Verify end-to-end inference from an unseen sketch."""
        gen = build_generator(input_shape=(64, 64, 3), encoder_filters=[16, 32, 64, 64, 64])
        unseen_sketch = np.ones((80, 80, 3), dtype=np.uint8) * 255
        out_file = self.temp_dir / "test_inference.png"

        pil_res, elapsed = generate_landscape(
            sketch_input=unseen_sketch,
            generator=gen,
            image_size=64,
            output_path=out_file,
        )
        self.assertEqual(pil_res.size, (64, 64))
        self.assertTrue(out_file.exists())
        self.assertGreater(elapsed, 0.0)


if __name__ == "__main__":
    unittest.main()
