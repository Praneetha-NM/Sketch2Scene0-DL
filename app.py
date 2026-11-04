"""
Sketch2Scene – AI-Powered Sketch-to-Landscape Generation
Streamlit Web Application.

Theme: Clean, modern white background, dark navy headings (#0F172A), subtle teal accents (#0D9488, #0EA5E9).
5 Pages:
1. Home / Project Overview
2. Dataset Explorer
3. Train Model
4. Generate Landscape
5. Evaluation
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st

# Setup paths and environment
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    IMAGE_SIZE,
    CHANNELS,
    BATCH_SIZE,
    LEARNING_RATE,
    BETA_1,
    L1_LAMBDA,
    TRAIN_LIMIT,
    VALIDATION_LIMIT,
    MAX_EPOCHS,
    RAW_DATA_DIR,
    PAIRS_CSV_PATH,
    CHECKPOINT_DIR,
    LATEST_CHECKPOINT_DIR,
    BEST_CHECKPOINT_DIR,
    STANDALONE_GENERATOR_DIR,
    SAMPLES_DIR,
    GENERATED_DIR,
    METRICS_CSV_PATH,
    OUTPUT_DIR,
    DEFAULT_CONFIG,
    PILOT_CONFIG,
    ModelConfig,
)
from preprocessing.image_processing import (
    load_image,
    preprocess_sketch,
    preprocess_pair,
    denormalize_from_gan_range,
    resize_and_pad,
)
from data.dataset_loader import (
    discover_pairs_from_directory,
    create_pairs_csv,
    download_official_subset,
)
from inference.generate import generate_landscape
from training.checkpoint import load_trained_generator

# Page Configuration
st.set_page_config(
    page_title="Sketch2Scene – cGAN Landscape Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (White Background, Dark Navy Headings, Subtle Teal Accents)
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1E293B;
    }

    .main {
        background-color: #F8FAFC;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #0F172A !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    /* Teal Badge */
    .teal-badge {
        display: inline-block;
        background-color: #CCFBF1;
        color: #0F766E;
        font-weight: 600;
        font-size: 0.78rem;
        padding: 4px 10px;
        border-radius: 9999px;
        border: 1px solid #99F6E4;
        margin-bottom: 8px;
    }

    .navy-badge {
        display: inline-block;
        background-color: #F1F5F9;
        color: #334155;
        font-weight: 600;
        font-size: 0.78rem;
        padding: 4px 10px;
        border-radius: 9999px;
        border: 1px solid #E2E8F0;
        margin-bottom: 8px;
    }

    /* Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(13, 148, 136, 0.08);
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #0D9488;
        font-weight: 500;
        margin-top: 2px;
    }

    /* Buttons */
    .stButton>button {
        background-color: #0D9488 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 22px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 4px rgba(13, 148, 136, 0.2) !important;
    }
    .stButton>button:hover {
        background-color: #0F766E !important;
        box-shadow: 0 4px 8px rgba(13, 148, 136, 0.3) !important;
        transform: translateY(-1px);
    }

    /* Code & Mono */
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Helper: Check trained model availability
def check_model_status() -> Tuple[bool, Optional[str], Optional[int]]:
    standalone_file = STANDALONE_GENERATOR_DIR / "generator_weights.weights.h5"
    if standalone_file.exists():
        return True, "Standalone Generator Available", None

    latest_ckpt = tf.train.latest_checkpoint(str(LATEST_CHECKPOINT_DIR))
    if latest_ckpt:
        return True, f"Checkpoint Ready ({Path(latest_ckpt).name})", None

    best_ckpt = tf.train.latest_checkpoint(str(BEST_CHECKPOINT_DIR))
    if best_ckpt:
        return True, f"Best Checkpoint Ready ({Path(best_ckpt).name})", None

    return False, "Not Trained Yet (From Scratch)", None


# Cached generator loader
@st.cache_resource(show_spinner=False)
def get_cached_generator(img_size: int = IMAGE_SIZE):
    try:
        gen = load_trained_generator(image_size=img_size)
        return gen
    except Exception:
        return None


# Sidebar Navigation
st.sidebar.markdown(
    """
    <div style="padding: 10px 0 20px 0;">
        <span class="teal-badge">Deep Learning • cGAN</span>
        <h2 style="margin: 0; font-size: 1.35rem; color: #0F172A;">Sketch2Scene</h2>
        <p style="margin: 4px 0 0 0; font-size: 0.8rem; color: #64748B;">Landscape Image Synthesis</p>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home & Overview",
        "🔍 Dataset Explorer",
        "⚙️ Train Model",
        "🎨 Generate Landscape",
        "📊 Evaluation & Metrics",
    ],
)

has_model, model_msg, model_epoch = check_model_status()

# Sidebar Model Status Card
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div class="metric-card" style="padding: 12px 14px;">
        <div class="metric-label">Model Status</div>
        <div style="font-size: 0.95rem; font-weight: 700; color: {'#0D9488' if has_model else '#E11D48'}; margin-top: 4px;">
            {'● ' + model_msg}
        </div>
        <div style="font-size: 0.75rem; color: #64748B; margin-top: 4px;">
            Target: Apple M-Series CPU (128x128)
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# PAGE 1: HOME & PROJECT OVERVIEW
# ==============================================================================
if page == "🏠 Home & Overview":
    st.markdown('<span class="teal-badge">Architectural Specification</span>', unsafe_allow_html=True)
    st.title("Sketch-to-Photo Landscape Generation")
    st.markdown(
        """
        **A Conditional Generative Adversarial Network (cGAN) trained entirely from scratch**
        to translate hand-drawn landscape sketches into realistic RGB landscape photographs.
        """
    )

    # Top KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Architecture</div>
                <div class="metric-value">Pix2Pix</div>
                <div class="metric-sub">U-Net + PatchGAN</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Pretrained Weights</div>
                <div class="metric-value">0%</div>
                <div class="metric-sub">Strictly from scratch</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Resolution</div>
                <div class="metric-value">128 × 128</div>
                <div class="metric-sub">64x64 pilot mode</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Hardware Target</div>
                <div class="metric-value">Apple CPU</div>
                <div class="metric-sub">Pure float32 tf.data</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Architectural Overview Section
    c_left, c_right = st.columns([1.1, 0.9])
    with c_left:
        st.subheader("Architectural Flow & Training Methodology")
        st.markdown(
            """
            Unlike unconditional GANs that sample from random noise $z$, a **Conditional GAN (cGAN)**
            conditions image generation directly on a spatial sketch representation $x$.

            1. **U-Net Generator ($G$)**:
               - **Encoder**: 5 downsampling blocks utilizing $4\\times4$ convolutions with stride 2 and LeakyReLU.
               - **Bottleneck**: Deep latent feature representation at spatial resolution $4\\times4$.
               - **Decoder**: 5 upsampling blocks with transpose convolutions and **symmetric skip connections**
                 that carry fine spatial sketch boundaries directly across the network.
               - **Output Activation**: $\\tanh$ mapping normalized activations to $[-1, 1]$.

            2. **PatchGAN Discriminator ($D$)**:
               - Penalizes structure at the scale of local $16\\times16$ image patches.
               - Accepts a **6-channel concatenated tensor**: Condition sketch $x$ + candidate photo $y$.
               - Outputs a patch validity score matrix without sigmoid activation (`from_logits=True`).

            3. **Objective Formulation**:
               $$\\mathcal{L}_{\\text{cGAN}}(G, D) = \\mathbb{E}_{x, y}[\\log D(x, y)] + \\mathbb{E}_{x}[\\log(1 - D(x, G(x)))]$$
               $$\\mathcal{L}_{L1}(G) = \\mathbb{E}_{x, y}[\\|y - G(x)\\|_1]$$
               $$\\mathcal{L}_{\\text{total}} = \\mathcal{L}_{\\text{cGAN}} + \\lambda_{L1} \\cdot \\mathcal{L}_{L1} \\quad (\\lambda_{L1} = 100)$$
            """
        )

    with c_right:
        st.subheader("Official Dataset: SketchyCOCO")
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px;">
                <h4 style="margin-top: 0; color: #0F172A;">Scene-Level Paired Corpus</h4>
                <p style="font-size: 0.88rem; color: #475569;">
                    The model is trained strictly on genuine paired freehand scene sketches and their matching
                    real-world scene photographs from the CVPR 2020 <b>SketchyCOCO</b> benchmark.
                </p>
                <ul style="font-size: 0.85rem; color: #334155; padding-left: 20px;">
                    <li><b>Official Training Pairs:</b> 11,265 scene pairs</li>
                    <li><b>Official Validation Pairs:</b> 2,816 scene pairs</li>
                    <li><b>Pilot / Active Subset:</b> 1,200 train / 150 val</li>
                    <li><b>Strict Alignment:</b> Aspect-ratio preserving geometric transformations</li>
                </ul>
                <div style="margin-top: 12px;">
                    <span class="navy-badge">CVPR 2020 Benchmark</span>
                    <span class="teal-badge">Zero Random Shuffling</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Training from Scratch Integrity")
        st.info(
            "⚠️ **Strict Academic Rule**: Both Generator and Discriminator start with randomly "
            "initialized weights ($(\\mu=0.0, \\sigma=0.02)$). No pretrained feature extractors (VGG, Inception, ResNet), "
            "no transfer learning, and no external generative APIs are used."
        )


# ==============================================================================
# PAGE 2: DATASET EXPLORER
# ==============================================================================
elif page == "🔍 Dataset Explorer":
    st.markdown('<span class="teal-badge">Ground Truth & Pair Verification</span>', unsafe_allow_html=True)
    st.title("SketchyCOCO Dataset Explorer")

    pairs, stats = discover_pairs_from_directory(RAW_DATA_DIR)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Discovered Pairs</div>
                <div class="metric-value">{stats['valid_pairs']}</div>
                <div class="metric-sub">Basename matched</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Train Pairs</div>
                <div class="metric-value">{stats['train_pairs']}</div>
                <div class="metric-sub">Scene/trainInTrain</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Val Pairs</div>
                <div class="metric-value">{stats['val_pairs']}</div>
                <div class="metric-sub">Scene/val</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Manifest Exists</div>
                <div class="metric-value">{'Yes' if PAIRS_CSV_PATH.exists() else 'No'}</div>
                <div class="metric-sub">pairs.csv</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if stats["valid_pairs"] == 0:
        st.warning("⚠️ No paired images currently detected in `data/raw/`.")
        st.markdown(
            """
            You can either:
            1. **Download official subset via CDN** (Recommended for fast start without 6.7GB disk space)
            2. **Download full Scene.tar manually** from Google Drive and place into `data/raw/Scene/`
            """
        )
        if st.button("📥 Download Official SketchyCOCO Subset (Train: 1200, Val: 150)"):
            with st.spinner("Downloading paired images from official SketchyCOCO CDN mirror..."):
                success = download_official_subset(RAW_DATA_DIR, train_count=TRAIN_LIMIT, val_count=VALIDATION_LIMIT)
                if success:
                    create_pairs_csv(RAW_DATA_DIR, PAIRS_CSV_PATH)
                    st.success("Download complete and pairs.csv manifest created!")
                    st.rerun()
                else:
                    st.error("Failed to connect or download from CDN. Check internet connection.")
    else:
        # Show sample pairs
        st.subheader("Sample Sketch-Photo Alignments")
        st.caption("Verifying that sketches directly correspond to target photographs:")

        if not PAIRS_CSV_PATH.exists():
            create_pairs_csv(RAW_DATA_DIR, PAIRS_CSV_PATH)

        manifest_df = pd.read_csv(PAIRS_CSV_PATH)

        # Allow user to refresh random samples
        if "sample_seed" not in st.session_state:
            st.session_state.sample_seed = 42

        if st.button("🎲 Shuffle Samples"):
            st.session_state.sample_seed = np.random.randint(0, 10000)

        sampled_rows = manifest_df.sample(min(4, len(manifest_df)), random_state=st.session_state.sample_seed)

        cols = st.columns(len(sampled_rows))
        for idx, (_, row) in enumerate(sampled_rows.iterrows()):
            with cols[idx]:
                s_img = Image.open(row["sketch_path"])
                p_img = Image.open(row["photo_path"])

                st.markdown(f"**Pair {idx+1} ({row['split']})**")
                st.caption(f"ID: `{Path(row['sketch_path']).name}`")
                st.image(s_img, caption="Input Sketch", use_container_width=True)
                st.image(p_img, caption="Real Photo (GT)", use_container_width=True)

        st.markdown("---")
        st.subheader("Dataset Manifest (data/processed/pairs.csv)")
        st.dataframe(manifest_df.head(50), use_container_width=True)


# ==============================================================================
# PAGE 3: TRAIN MODEL
# ==============================================================================
elif page == "⚙️ Train Model":
    st.markdown('<span class="teal-badge">GradientTape Execution</span>', unsafe_allow_html=True)
    st.title("Train Conditional GAN from Scratch")

    pairs, stats = discover_pairs_from_directory(RAW_DATA_DIR)
    if stats["valid_pairs"] == 0:
        st.error("⚠️ Dataset must be discovered or downloaded before training can commence. Navigate to **Dataset Explorer**.")
    else:
        st.markdown(
            "Configure hyperparameters and start training directly on the Apple M-series CPU. "
            "Progress is updated live and model checkpoints are saved safely after each epoch."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            mode = st.selectbox("Training Mode", ["Pilot Mode (64x64)", "Main Mode (128x128)"])
            res = 64 if "Pilot" in mode else 128
            epochs = st.number_input("Max Epochs", min_value=1, max_value=50, value=3 if "Pilot" in mode else 10)
        with c2:
            train_subset = st.number_input(
                "Training Pairs Limit", min_value=10, max_value=stats["valid_pairs"], value=100 if "Pilot" in mode else min(1200, stats["train_pairs"])
            )
            batch_sz = st.selectbox("Batch Size", [1, 2, 4], index=0, help="Batch size 1 is standard for Pix2Pix instance norm behavior")
        with c3:
            time_limit_min = st.number_input("Time Budget (Minutes, 0=Unlimited)", min_value=0, max_value=600, value=15)
            resume_check = st.checkbox("Resume from latest checkpoint", value=has_model)

        # Active lock check
        lock_file = PROJECT_ROOT / ".training.lock"
        is_running = lock_file.exists()

        if is_running:
            st.warning("⚠️ A training process is currently running or locked by `.training.lock`.")
            if st.button("Unlock / Force Reset Lock"):
                lock_file.unlink(missing_ok=True)
                st.success("Lock released.")
                st.rerun()

        start_btn = st.button("🚀 Start Training", disabled=is_running)

        if start_btn:
            cfg = PilotConfig() if res == 64 else ModelConfig()
            cfg.image_size = res
            cfg.max_epochs = epochs
            cfg.train_limit = train_subset
            cfg.batch_size = batch_sz

            max_sec = time_limit_min * 60.0 if time_limit_min > 0 else None

            # Training UI status
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            loss_chart_placeholder = st.empty()
            sample_preview_placeholder = st.empty()

            from training.train import train_model as execute_train

            loss_history_data = []

            def ui_callback(info):
                ep = info["epoch"]
                max_ep = info["max_epochs"]
                progress_bar.progress(float(ep / max_ep))
                status_text.markdown(
                    f"**Epoch {ep}/{max_ep}** completed in {info['epoch_duration']:.1f}s | "
                    f"Gen Loss: `{info['metrics']['gen_total_loss']:.4f}` | "
                    f"Disc Loss: `{info['metrics']['disc_loss']:.4f}` | "
                    f"Val L1: `{info['metrics']['val_gen_l1_loss']:.4f}`"
                )
                loss_history_data.append({
                    "Epoch": ep,
                    "Generator Loss": info["metrics"]["gen_total_loss"],
                    "Discriminator Loss": info["metrics"]["disc_loss"],
                    "Val L1 Loss (x100)": info["metrics"]["val_gen_l1_loss"] * 100.0,
                })
                loss_chart_placeholder.line_chart(
                    pd.DataFrame(loss_history_data).set_index("Epoch")
                )

                # Display latest sample
                latest_sample = SAMPLES_DIR / f"val_sample_epoch_{ep:03d}.png"
                if latest_sample.exists():
                    sample_preview_placeholder.image(
                        str(latest_sample),
                        caption=f"Validation Predictions - Epoch {ep}",
                        use_container_width=True,
                    )

            try:
                with st.spinner("Training neural networks from random initialization..."):
                    result = execute_train(
                        config=cfg,
                        resume=resume_check,
                        max_duration_seconds=max_sec,
                        progress_callback=ui_callback,
                    )
                st.success(f"Training run finished successfully! Completed {result['completed_epochs']} epoch(s).")
                # Clear cached generator so updated weights take effect
                st.cache_resource.clear()
            except Exception as e:
                st.error(f"Training Error: {e}")


# ==============================================================================
# PAGE 4: GENERATE LANDSCAPE (INFERENCE)
# ==============================================================================
elif page == "🎨 Generate Landscape":
    st.markdown('<span class="teal-badge">Inference Studio</span>', unsafe_allow_html=True)
    st.title("Landscape Synthesis from Sketch")
    st.markdown("Supply a hand-drawn landscape sketch to generate a realistic RGB photograph.")

    if not has_model:
        st.warning(
            "⚠️ No trained generator model checkpoint found. "
            "Please train the model first on the **Train Model** page or run `python -m training.train --mode pilot`."
        )
    else:
        tab1, tab2 = st.tabs(["📤 Upload Sketch", "🖼️ Curated Validation Sketches"])

        sketch_to_process = None

        with tab1:
            uploaded_file = st.file_uploader(
                "Upload a landscape sketch image (JPEG/PNG)", type=["png", "jpg", "jpeg"]
            )
            if uploaded_file is not None:
                sketch_to_process = Image.open(uploaded_file)

        with tab2:
            st.caption("Select a genuine unseen sketch from the held-out validation set:")
            if PAIRS_CSV_PATH.exists():
                val_df = pd.read_csv(PAIRS_CSV_PATH)
                val_df = val_df[val_df["split"] == "val"]
                if len(val_df) > 0:
                    val_samples = val_df.head(6)
                    val_cols = st.columns(6)
                    selected_val_idx = None
                    for idx, (_, vrow) in enumerate(val_samples.iterrows()):
                        with val_cols[idx]:
                            v_thumb = Image.open(vrow["sketch_path"])
                            st.image(v_thumb, use_container_width=True)
                            if st.button(f"Use #{idx+1}", key=f"sel_val_{idx}"):
                                sketch_to_process = v_thumb
                                st.session_state["picked_sketch"] = v_thumb

                    if "picked_sketch" in st.session_state and sketch_to_process is None:
                        sketch_to_process = st.session_state["picked_sketch"]

        if sketch_to_process is not None:
            st.markdown("---")
            col_in, col_out = st.columns(2)

            with col_in:
                st.subheader("Input Sketch")
                st.image(sketch_to_process, caption=f"Original ({sketch_to_process.size[0]}x{sketch_to_process.size[1]})", use_container_width=True)

            with col_out:
                st.subheader("Generated Landscape Photo")
                gen_btn = st.button("✨ Generate Landscape Photograph", type="primary")

                if gen_btn:
                    with st.spinner("Generating landscape via trained U-Net Generator..."):
                        gen_model = get_cached_generator(img_size=IMAGE_SIZE)
                        if gen_model is None:
                            # Try 64x64 if 128x128 not found
                            gen_model = get_cached_generator(img_size=64)

                        current_size = gen_model.input_shape[1] if gen_model else IMAGE_SIZE

                        generated_photo, elapsed = generate_landscape(
                            sketch_input=sketch_to_process,
                            generator=gen_model,
                            image_size=current_size,
                        )

                        st.image(
                            generated_photo,
                            caption=f"Synthesized Landscape ({current_size}x{current_size} RGB)",
                            use_container_width=True,
                        )

                        st.markdown(
                            f"""
                            <div style="margin-top: 8px;">
                                <span class="teal-badge">Inference: {elapsed*1000:.1f} ms</span>
                                <span class="navy-badge">Generator: Trained From Scratch</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Download button
                        buf = Path(GENERATED_DIR / "temp_download.png")
                        buf.parent.mkdir(parents=True, exist_ok=True)
                        generated_photo.save(buf, format="PNG")
                        with open(buf, "rb") as f:
                            st.download_button(
                                label="💾 Download Generated Photograph",
                                data=f,
                                file_name="generated_landscape.png",
                                mime="image/png",
                            )


# ==============================================================================
# PAGE 5: EVALUATION
# ==============================================================================
elif page == "📊 Evaluation & Metrics":
    st.markdown('<span class="teal-badge">Quantitative Assessment</span>', unsafe_allow_html=True)
    st.title("Model Evaluation & Held-Out Metrics")

    st.markdown(
        """
        Evaluation metrics are calculated on the held-out validation pairs
        using **Mean Absolute Error (MAE)**, **Peak Signal-to-Noise Ratio (PSNR)**,
        and **Structural Similarity Index (SSIM)** via `scikit-image`.
        """
    )

    if not has_model:
        st.warning("⚠️ Please train the generator before running quantitative evaluation.")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            eval_samples = st.number_input("Validation Samples to Evaluate", min_value=5, max_value=150, value=30)
            run_eval_btn = st.button("🧪 Compute Evaluation Metrics")

        if run_eval_btn or METRICS_CSV_PATH.exists():
            if run_eval_btn:
                from evaluation.evaluate import evaluate_pairs
                with st.spinner("Evaluating pairs across MAE, PSNR, and SSIM..."):
                    eval_results = evaluate_pairs(
                        manifest_csv=PAIRS_CSV_PATH,
                        output_metrics_csv=METRICS_CSV_PATH,
                        max_samples=eval_samples,
                        image_size=IMAGE_SIZE,
                    )
                st.success("Evaluation complete!")

            if METRICS_CSV_PATH.exists():
                m_df = pd.read_csv(METRICS_CSV_PATH)
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                with kpi1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Mean MAE</div>
                            <div class="metric-value">{m_df['mae'].mean():.2f}</div>
                            <div class="metric-sub">Lower is closer</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with kpi2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Mean PSNR</div>
                            <div class="metric-value">{m_df['psnr_db'].mean():.2f} dB</div>
                            <div class="metric-sub">Peak SNR</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with kpi3:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Mean SSIM</div>
                            <div class="metric-value">{m_df['ssim'].mean():.4f}</div>
                            <div class="metric-sub">Structural similarity</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with kpi4:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Evaluated Pairs</div>
                            <div class="metric-value">{len(m_df)}</div>
                            <div class="metric-sub">Held-out validation</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # Show Loss Curves & Evaluation Grid
                grid_img = OUTPUT_DIR / "evaluation_comparison_grid.png"
                loss_img = OUTPUT_DIR / "loss_curves.png"

                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("Training Loss Curves")
                    if loss_img.exists():
                        st.image(str(loss_img), use_container_width=True)
                    else:
                        st.info("Loss curves will appear once training has logged epochs.")

                with col_b:
                    st.subheader("Sample Comparisons")
                    if grid_img.exists():
                        st.image(str(grid_img), use_container_width=True)
                    else:
                        st.info("Run evaluation to generate comparison grid.")

                st.markdown("---")
                st.subheader("Per-Sample Evaluation Table")
                st.dataframe(m_df, use_container_width=True)

                # CSV Download Button
                with open(METRICS_CSV_PATH, "rb") as f:
                    st.download_button(
                        label="📥 Download Evaluation Metrics CSV",
                        data=f,
                        file_name="evaluation_metrics.csv",
                        mime="text/csv",
                    )
