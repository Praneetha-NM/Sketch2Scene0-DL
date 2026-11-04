# Sketch2Scene: Sketch-to-Photo Landscape Synthesis Using a Conditional GAN Trained From Scratch

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![TensorFlow 2.19](https://img.shields.io/badge/TensorFlow-2.19.0-orange.svg)](https://tensorflow.org/)
[![Hardware](https://img.shields.io/badge/Hardware-Apple%20Silicon%20(CPU)-lightgrey.svg)]()
[![Model](https://img.shields.io/badge/Architecture-Pix2Pix%20cGAN-teal.svg)]()
[![License: CC-BY-4.0](https://img.shields.io/badge/License-CC%20BY%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)

An end-to-end deep learning project for paired image-to-image translation that accepts hand-drawn scene sketches and generates realistic RGB landscape photographs. Both the **U-Net Generator** and **PatchGAN Discriminator** are built and trained **strictly from scratch** from random Gaussian weight initialization, using real paired landscape data from the official **SketchyCOCO** CVPR benchmark.

---

## Table of Contents
1. [Project Overview & Problem Statement](#project-overview--problem-statement)
2. [Why Conditional GANs (cGAN)?](#why-conditional-gans-cgan)
3. [Model Architecture](#model-architecture)
   - [U-Net Generator with Skip Connections](#u-net-generator-with-skip-connections)
   - [PatchGAN Discriminator](#patchgan-discriminator)
4. [Loss Functions & Mathematical Formulation](#loss-functions--mathematical-formulation)
5. [Dataset & Pairing Methodology](#dataset--pairing-methodology)
6. [Image Preprocessing & Augmentation](#image-preprocessing--augmentation)
7. [Hardware-Aware Configuration for Apple Silicon](#hardware-aware-configuration-for-apple-silicon)
8. [Installation & Environment Setup](#installation--environment-setup)
9. [Execution & Workflows](#execution--workflows)
   - [1. Dataset Discovery & Inspection](#1-dataset-discovery--inspection)
   - [2. Stage 1: Pipeline Verification Pilot (64x64)](#2-stage-1-pipeline-verification-pilot-64x64)
   - [3. Stage 2: Main Training (128x128)](#3-stage-2-main-training-128x128)
   - [4. Resuming Training](#4-resuming-training)
   - [5. Standalone Image Inference](#5-standalone-image-inference)
   - [6. Quantitative Evaluation](#6-quantitative-evaluation)
   - [7. Interactive Streamlit Web Application](#7-interactive-streamlit-web-application)
10. [Automated Test Suite](#automated-test-suite)
11. [Limitations & Perceptual Realism](#limitations--perceptual-realism)
12. [Future Enhancements](#future-enhancements)

---

## Project Overview & Problem Statement

Transforming abstract, sparse line drawings into photorealistic natural scenes is a fundamental problem in computer vision and computational creativity. Traditional unconditional image generation maps latent noise vectors $z \sim \mathcal{N}(0, I)$ to unconstrained images, lacking direct spatial control. Text-to-image foundation models (e.g., Stable Diffusion) provide semantic conditioning but do not conform precisely to hand-drawn contours or spatial boundaries.

**Sketch2Scene** approaches this challenge as **supervised paired image-to-image translation**:
$$\mathcal{G}: \mathcal{X}_{\text{sketch}} \rightarrow \mathcal{Y}_{\text{photograph}}$$
Given an input sketch $x \in [-1, 1]^{H \times W \times 3}$, the network learns to synthesize a photorealistic counterpart $G(x) \approx y$ that preserves terrain topology, horizon lines, vegetation placement, and water bodies.

> **Academic Rigor Notice**: This is **supervised image translation**, not image classification or text-to-image prompting. No pretrained models, no fine-tuning, no transfer learning, and no external generative APIs are employed. The model begins with randomly initialized weights and learns landscape textures, foliage colors, and sky gradients directly from paired data.

---

## Why Conditional GANs (cGAN)?

Standard regression models trained solely with Mean Squared Error (MSE / L2) or Mean Absolute Error (MAE / L1) suffer from the **averaging effect**: because many plausible color patterns and textures can correspond to a single sketch line, minimizing pixel distance produces overly smoothed, blurry outputs representing the pixel-wise mean of all possibilities.

A **Conditional GAN (Pix2Pix-style)** overcomes this by pairing an $L_1$ reconstruction loss with an adversarial discriminator:
- The **$L_1$ loss** enforces low-frequency structural fidelity, anchoring the synthesized mountains, trees, and skies to the exact spatial layout of the sketch.
- The **adversarial loss** forces the generator to synthesize crisp, high-frequency realistic textures (grass blades, bark patterns, cloud edges) to fool the discriminator.

---

## Model Architecture

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 U-NET GENERATOR G(x)                   │
                  └────────────────────────────────────────────────────────┘
Input Sketch (x) ──► [Down 1 (32)] ───────── Skip 1 ─────────► [Up 5 (32)] ──► Conv2DTranspose ──► Generated Photo G(x)
(128x128x3)             │                                         ▲              (Tanh)            (128x128x3)
                    [Down 2 (64)] ─────── Skip 2 ───────► [Up 4 (64)]            
                        │                                   ▲                    
                    [Down 3 (128)] ───── Skip 3 ─────► [Up 3 (128)]              
                        │                               ▲                        
                    [Down 4 (256)] ─── Skip 4 ───► [Up 2 (256)]                  
                        │                           ▲                            
                    [Down 5 (256)] ─ Skip 5 ─► [Up 1 (256)]                      
                        │                        ▲                               
                        └──► [Bottleneck (256)] ─┘                               

                  ┌────────────────────────────────────────────────────────┐
                  │              PATCHGAN DISCRIMINATOR D(x, y)            │
                  └────────────────────────────────────────────────────────┘
Sketch (x)     ──┐
                 ├──► Concat (128x128x6) ──► Conv(64,s2) ──► Conv(128,s2) ──► Conv(256,s2) ──► Conv(1,s1) ──► Patch Map (16x16x1)
Photo y or G(x) ─┘                                                                              (No Sigmoid)
```

### U-Net Generator with Skip Connections
The generator adopts an encoder-decoder architecture with 5 downsampling stages, a central bottleneck, and 5 upsampling stages:
- **Encoder**: Each block comprises `Conv2D(kernel=4, stride=2, padding='same', use_bias=False)` $\rightarrow$ `BatchNormalization` $\rightarrow$ `LeakyReLU(0.2)`. (Stage 1 omits BatchNormalization).
  - Filter progression: $32 \rightarrow 64 \rightarrow 128 \rightarrow 256 \rightarrow 256$.
- **Bottleneck**: $4 \times 4 \times 256$ representation capturing deep semantic abstractions.
- **Decoder**: Each block comprises `Conv2DTranspose(kernel=4, stride=2, padding='same', use_bias=False)` $\rightarrow$ `BatchNormalization` $\rightarrow$ `Dropout(0.5)` (first 3 decoders) $\rightarrow$ `Concatenation` with matching encoder skip layer $\rightarrow$ `ReLU`.
- **Output Layer**: `Conv2DTranspose(filters=3, kernel=4, stride=2, padding='same', activation='tanh')` generating the final $128 \times 128 \times 3$ RGB image in range $[-1.0, 1.0]$.
- **Skip Connections**: Vital for sketch translation. They bypass the bottleneck bottlenecking, channeling high-resolution edge and border information directly to the corresponding reconstruction layer.

### PatchGAN Discriminator
Rather than classifying the entire $128 \times 128$ image as real or fake with a single scalar, the **PatchGAN** discriminator maps a 6-channel concatenated input (`[Sketch, Real/Fake Photo]`) to an $N \times N$ matrix of local patch predictions ($16 \times 16 \times 1$).
- Filters: $64 \rightarrow 128 \rightarrow 256$ with stride 2.
- Final layer: `Conv2D(filters=1, kernel=4, stride=1, padding='same')`.
- No sigmoid activation is applied to the output; training uses `BinaryCrossentropy(from_logits=True)` for numerical stability.
- Receptive Field: Penalizes structural incongruities at the patch level, enforcing local stylistic consistency.

---

## Loss Functions & Mathematical Formulation

### 1. Discriminator Loss
The discriminator learns to maximize the log-likelihood of correctly identifying real pairs while penalizing generated fake pairs:
$$\mathcal{L}_{\text{real}} = \text{BCE}\left(\mathbf{1}, D(x, y)\right)$$
$$\mathcal{L}_{\text{fake}} = \text{BCE}\left(\mathbf{0}, D(x, G(x))\right)$$
$$\mathcal{L}_{D} = \frac{1}{2}\left(\mathcal{L}_{\text{real}} + \mathcal{L}_{\text{fake}}\right)$$

### 2. Generator Loss
The generator seeks to fool the discriminator into classifying generated photos as genuine ($\mathbf{1}$), combined with an $L_1$ penalty weighted by $\lambda = 100$:
$$\mathcal{L}_{\text{GAN}} = \text{BCE}\left(\mathbf{1}, D(x, G(x))\right)$$
$$\mathcal{L}_{L1} = \frac{1}{H \cdot W \cdot C} \sum_{i,j,k} |y_{i,j,k} - G(x)_{i,j,k}|$$
$$\mathcal{L}_{G} = \mathcal{L}_{\text{GAN}} + \lambda_{L1} \cdot \mathcal{L}_{L1}$$

---

## Dataset & Pairing Methodology

This project utilizes the scene-level dataset from **SketchyCOCO** ([Gao et al., CVPR 2020](https://arxiv.org/abs/2003.02683)).

### Directory Hierarchy:
```
data/
├── raw/
│   └── Scene/
│       ├── Sketch/
│       │   ├── trainInTrain/   <-- Scene sketch files (e.g., 000000327573.jpg)
│       │   └── val/            <-- Validation sketches
│       └── GT/
│           ├── trainInTrain/   <-- Matching scene ground truth photos
│           └── val/            <-- Validation ground truth photos
└── processed/
    └── pairs.csv               <-- Auto-generated manifest [sketch_path, photo_path, split]
```

### Pairing Rules:
1. **Identifier Matching**: A sketch and photograph pair share the exact same basename stem (e.g., `000000327573`).
2. **Split Isolation**: Official validation images (`Scene/GT/val` and `Scene/Sketch/val`) are strictly partitioned from training pairs.
3. **Reproducible Subsetting**: A fixed seed (`SEED = 42`) governs subset selection (default 1,200 train, 150 val).

---

## Image Preprocessing & Augmentation

Implemented in `preprocessing/image_processing.py`:
- **Format Normalization**: Guarantees 3-channel RGB across all inputs (handling grayscale sketches and RGBA files).
- **Aspect Ratio Preservation**: Resizes images using high-quality bicubic interpolation and applies symmetric letterbox padding to maintain true terrain proportions.
- **Normalization**: Pixel intensities $[0, 255]$ are scaled to $[-1.0, 1.0]$ via $x_{\text{norm}} = \frac{x}{127.5} - 1.0$.
- **Synchronized Augmentation**: Synchronized random horizontal flips (`tf.image.flip_left_right`) applied identically to both sketch and photograph. Never augmented independently.

---

## Hardware-Aware Configuration for Apple Silicon

Trained on macOS on Apple Silicon (M-series, 16 GB unified memory, CPU execution):
- **Float32 Precision**: Pure `float32` avoid CPU overhead of unsupported mixed-precision types.
- **Batch Size 1**: Minimizes working memory, matching standard Pix2Pix instance normalization dynamics.
- **Thread Tuning**: Configured with `intra_op_parallelism_threads=4` and `inter_op_parallelism_threads=2`.
- **Dynamic Streaming**: `tf.data.Dataset` streams images on demand with `prefetch(AUTOTUNE)`, keeping RAM usage under 1.5 GB.

---

## Installation & Environment Setup

### 1. Clone & Enter Repository
```bash
git clone https://github.com/sysu-imsl/SketchyCOCO.git # (optional reference)
cd /Users/praneetha18/Sketch2Photo-DL
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Execution & Workflows

### 1. Dataset Discovery & Inspection
Scans the dataset directory, matches pairs, checks alignments, and builds `data/processed/pairs.csv`:
```bash
python -m utils.dataset_inspection --download-if-missing
```

### 2. Stage 1: Pipeline Verification Pilot (64x64)
Quickly verifies the complete pipeline at $64 \times 64$ resolution over 100 pairs:
```bash
python -m training.train --mode pilot
```

### 3. Stage 2: Main Training (128x128)
Runs main training from scratch on 1,200 pairs for up to 10 epochs:
```bash
python -m training.train --mode main
```

With custom duration limit (e.g. 30 minutes):
```bash
python -m training.train --mode main --max-duration 30
```

### 4. Resuming Training
Picks up from the latest saved checkpoint:
```bash
python -m training.train --resume
```

### 5. Standalone Image Inference
Generates a landscape photo from a new sketch:
```bash
python -m inference.generate --input data/raw/Scene/Sketch/val/000000327573.jpg --output outputs/generated/my_landscape.png
```

### 6. Quantitative Evaluation
Evaluates held-out validation pairs and generates `outputs/evaluation_metrics.csv` and visual grids:
```bash
python -m evaluation.evaluate
```

### 7. Interactive Streamlit Web Application
Launches the full interactive UI:
```bash
streamlit run app.py
```

---

## Automated Test Suite

A complete test suite is provided in `tests/test_pipeline.py`. To execute:
```bash
python tests/test_pipeline.py
```
**Tests Verified**:
- U-Net Generator output shapes ($128 \times 128 \times 3$ and $64 \times 64 \times 3$) and $[-1, 1]$ bounds.
- PatchGAN Discriminator output shapes ($16 \times 16 \times 1$) with 6-channel input.
- Loss functions (adversarial BCE + L1 penalty).
- Dual `tf.GradientTape` backward pass and weight updates.
- Checkpoint save, restoration, and standalone generator weight export.
- Aspect-ratio letterbox preprocessing and denormalization.
- End-to-end inference from an unseen sketch.

---

## Limitations & Perceptual Realism

- **Pixel-Based vs Perceptual Metrics**: MAE, PSNR, and SSIM compare generated outputs directly against the paired ground-truth photo. However, landscapes have infinite plausible leaf, cloud, and water textures. A generated image may be perceptually realistic while exhibiting moderate PSNR/SSIM.
- **CPU Epoch Constraints**: Training a GAN from scratch on CPU yields observable structural colorization and macro-features within early epochs. Fine photorealism typically requires prolonged multi-epoch GPU runs.
- **Sparse vs Detailed Sketches**: Sketches with well-defined horizons and contours generate more coherent terrain than abstract single-line doodles.

---

## Future Enhancements
- Multi-scale PatchGAN discriminators ($70\times70$ and $286\times286$ receptive fields).
- Semantic segmentation mask guidance (incorporating SketchyCOCO annotation ground truths).
- Self-attention blocks in deep generator layers to improve distant mountain and cloud coherence.
- Metal Performance Shaders (MPS) acceleration on Apple Silicon via `tensorflow-metal`.
