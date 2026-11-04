"""
Dataset Loader and Discovery Module for SketchyCOCO Scene Pairs.

Features:
- Scans local raw dataset directories for SketchyCOCO paired structures:
    Scene/GT/trainInTrain, Scene/Sketch/trainInTrain
    Scene/GT/val, Scene/Sketch/val
    or flat paired directories
- Matches pairs using exact image IDs (e.g. 000000327573.jpg)
- Generates reproducible CSV manifest at data/processed/pairs.csv
- Official split preservation with deterministic subset sampling
- Built-in downloader for official SketchyCOCO CDN mirror (for pilot/main runs)
- tf.data pipeline with streaming, parallel decoding, synchronized augmentation, and prefetching
"""

import os
import csv
import json
import random
import urllib.request
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import pandas as pd
import tensorflow as tf

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    PAIRS_CSV_PATH,
    TRAIN_LIMIT,
    VALIDATION_LIMIT,
    SEED,
    IMAGE_SIZE,
    BATCH_SIZE,
)
from preprocessing.image_processing import augment_pair

OFFICIAL_CDN_SCENE_JSON = (
    "https://cdn.jsdelivr.net/gh/sysu-imsl/CDN-for-SketchyCOCO@v1.0/Scene.json"
)
OFFICIAL_CDN_SKETCH_BASE = (
    "https://cdn.jsdelivr.net/gh/sysu-imsl/CDN-for-SketchyCOCO@v1.0/data/Scene/Sketch/"
)
OFFICIAL_CDN_GT_BASE = (
    "https://cdn.jsdelivr.net/gh/sysu-imsl/CDN-for-SketchyCOCO@v1.0/data/Scene/GT/"
)


def discover_pairs_from_directory(
    root_dir: Path = RAW_DATA_DIR,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """
    Scan root_dir recursively to discover matching Sketch and Ground Truth pairs.

    Matches files based on basename (e.g. 000000123456.jpg).
    Detects official train/val folder splits (trainInTrain vs val/valInTrain).

    Returns:
        List of dicts with keys: {'sketch_path', 'photo_path', 'split', 'pair_id'}
        Summary statistics dictionary.
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        return [], {"total_sketches": 0, "total_photos": 0, "valid_pairs": 0}

    # Find candidate sketches and photos
    valid_extensions = {".jpg", ".jpeg", ".png"}
    sketch_files: Dict[str, Path] = {}
    photo_files: Dict[str, Path] = {}
    sketch_splits: Dict[str, str] = {}
    photo_splits: Dict[str, str] = {}

    for path in root_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in valid_extensions:
            continue

        try:
            rel_parts = [p.lower() for p in path.relative_to(root_dir).parts]
        except ValueError:
            rel_parts = [p.lower() for p in path.parts]

        stem = path.stem.lower()

        # Determine split
        if any("val" in part for part in rel_parts):
            split = "val"
        else:
            split = "train"

        # Determine whether this is a Sketch or a Photo (GT)
        is_sketch = any(s in part for part in rel_parts for s in ["sketch", "sketches"])
        is_gt = any(g in part for part in rel_parts for g in ["gt", "photo", "photos", "groundtruth", "scene_image"])

        if is_sketch and not is_gt:
            sketch_files[stem] = path
            sketch_splits[stem] = split
        elif is_gt and not is_sketch:
            photo_files[stem] = path
            photo_splits[stem] = split
        elif "sketch" in stem and "photo" not in stem:
            clean_stem = stem.replace("sketch_", "").replace("_sketch", "")
            sketch_files[clean_stem] = path
            sketch_splits[clean_stem] = split
        elif "photo" in stem and "sketch" not in stem:
            clean_stem = stem.replace("photo_", "").replace("_photo", "")
            photo_files[clean_stem] = path
            photo_splits[clean_stem] = split

    # Match pairs by common stems
    common_stems = sorted(list(set(sketch_files.keys()).intersection(photo_files.keys())))
    pairs: List[Dict[str, str]] = []

    for stem in common_stems:
        # If either is labeled val, treat as val; otherwise train
        split = "val" if (sketch_splits.get(stem) == "val" or photo_splits.get(stem) == "val") else "train"
        pairs.append({
            "pair_id": stem,
            "sketch_path": str(sketch_files[stem].resolve()),
            "photo_path": str(photo_files[stem].resolve()),
            "split": split,
        })

    stats = {
        "total_sketches": len(sketch_files),
        "total_photos": len(photo_files),
        "valid_pairs": len(pairs),
        "train_pairs": sum(1 for p in pairs if p["split"] == "train"),
        "val_pairs": sum(1 for p in pairs if p["split"] == "val"),
    }
    return pairs, stats


def download_official_subset(
    target_dir: Path = RAW_DATA_DIR,
    train_count: int = TRAIN_LIMIT,
    val_count: int = VALIDATION_LIMIT,
    timeout: int = 15,
) -> bool:
    """
    Download a subset of genuine SketchyCOCO scene pairs from the official author CDN.
    Guarantees true paired landscape sketches and photographs without needing 6.7GB full tar.
    """
    target_dir = Path(target_dir)
    sketch_dir_train = target_dir / "Scene" / "Sketch" / "trainInTrain"
    photo_dir_train = target_dir / "Scene" / "GT" / "trainInTrain"
    sketch_dir_val = target_dir / "Scene" / "Sketch" / "val"
    photo_dir_val = target_dir / "Scene" / "GT" / "val"

    for d in [sketch_dir_train, photo_dir_train, sketch_dir_val, photo_dir_val]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Fetching scene list from official CDN: {OFFICIAL_CDN_SCENE_JSON}...")
    req = urllib.request.Request(OFFICIAL_CDN_SCENE_JSON, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            scene_catalog = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"Error connecting to official CDN: {e}")
        return False

    all_scenes = scene_catalog.get("data", [])
    if not all_scenes:
        print("No scene files found in CDN catalog.")
        return False

    # Official split: train 11265, val 2816
    official_train_scenes = all_scenes[:11265]
    official_val_scenes = all_scenes[11265:]

    # Sample reproducibly
    rng = random.Random(SEED)
    sampled_train = rng.sample(official_train_scenes, min(train_count, len(official_train_scenes)))
    sampled_val = rng.sample(official_val_scenes, min(val_count, len(official_val_scenes)))

    print(f"Downloading {len(sampled_train)} train pairs and {len(sampled_val)} val pairs...")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_pair(fname: str, s_dest: Path, p_dest: Path) -> bool:
        s_path = s_dest / fname
        p_path = p_dest / fname
        if s_path.exists() and p_path.exists() and s_path.stat().st_size > 0 and p_path.stat().st_size > 0:
            return True

        s_url = OFFICIAL_CDN_SKETCH_BASE + fname
        p_url = OFFICIAL_CDN_GT_BASE + fname
        try:
            req_s = urllib.request.Request(s_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req_s, timeout=timeout) as resp:
                s_bytes = resp.read()
            req_p = urllib.request.Request(p_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req_p, timeout=timeout) as resp:
                p_bytes = resp.read()

            with open(s_path, "wb") as f:
                f.write(s_bytes)
            with open(p_path, "wb") as f:
                f.write(p_bytes)
            return True
        except Exception:
            if s_path.exists():
                s_path.unlink()
            if p_path.exists():
                p_path.unlink()
            return False

    def download_list(items: List[str], s_dest: Path, p_dest: Path, label: str):
        success_count = 0
        total = len(items)
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(_fetch_pair, fname, s_dest, p_dest): fname for fname in items}
            for i, fut in enumerate(as_completed(futures), 1):
                if fut.result():
                    success_count += 1
                if i % 25 == 0 or i == total:
                    print(f"[{label}] Processed {i}/{total} pairs ({success_count} succeeded)...")
        return success_count

    t_done = download_list(sampled_train, sketch_dir_train, photo_dir_train, "Train")
    v_done = download_list(sampled_val, sketch_dir_val, photo_dir_val, "Validation")
    print(f"Download complete: {t_done} train pairs, {v_done} val pairs ready.")
    return (t_done + v_done) > 0


def create_pairs_csv(
    raw_dir: Path = RAW_DATA_DIR,
    output_csv: Path = PAIRS_CSV_PATH,
    train_limit: int = TRAIN_LIMIT,
    val_limit: int = VALIDATION_LIMIT,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Generate the pairs manifest CSV file:
    columns: sketch_path, photo_path, split

    Enforces reproducible sampling while strictly preserving official train/val partitions.
    """
    pairs, stats = discover_pairs_from_directory(raw_dir)

    if not pairs:
        print(f"Warning: No valid pairs found in {raw_dir}.")
        # Create empty DataFrame with expected columns
        df = pd.DataFrame(columns=["sketch_path", "photo_path", "split"])
        df.to_csv(output_csv, index=False)
        return df

    train_pairs = [p for p in pairs if p["split"] == "train"]
    val_pairs = [p for p in pairs if p["split"] == "val"]

    # If no explicit val folder exists, perform deterministic stratified 80/20 split
    if not val_pairs:
        rng = random.Random(seed)
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        split_idx = int(0.85 * len(shuffled))
        train_pairs = shuffled[:split_idx]
        val_pairs = shuffled[split_idx:]
        for p in train_pairs:
            p["split"] = "train"
        for p in val_pairs:
            p["split"] = "val"

    # Limit subset size reproducibly
    rng = random.Random(seed)
    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)

    train_subset = train_pairs[:train_limit] if train_limit else train_pairs
    val_subset = val_pairs[:val_limit] if val_limit else val_pairs

    final_pairs = train_subset + val_subset
    df = pd.DataFrame(final_pairs)[["sketch_path", "photo_path", "split"]]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(
        f"Manifest created at {output_csv} with {len(train_subset)} train and {len(val_subset)} val pairs."
    )
    return df


def load_and_preprocess_tensor(
    sketch_path: tf.Tensor,
    photo_path: tf.Tensor,
    image_size: int = IMAGE_SIZE,
    augment: bool = True,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    TensorFlow graph-compatible file reading, decoding, resizing, and normalization.
    """
    # Read sketch
    sketch_bytes = tf.io.read_file(sketch_path)
    sketch = tf.io.decode_image(sketch_bytes, channels=3, expand_animations=False)
    sketch = tf.image.resize(
        sketch, [image_size, image_size], method=tf.image.ResizeMethod.BICUBIC
    )
    sketch = (tf.cast(sketch, tf.float32) / 127.5) - 1.0

    # Read photo
    photo_bytes = tf.io.read_file(photo_path)
    photo = tf.io.decode_image(photo_bytes, channels=3, expand_animations=False)
    photo = tf.image.resize(
        photo, [image_size, image_size], method=tf.image.ResizeMethod.BICUBIC
    )
    photo = (tf.cast(photo, tf.float32) / 127.5) - 1.0

    # Enforce static shape for downstream graph execution
    sketch.set_shape([image_size, image_size, 3])
    photo.set_shape([image_size, image_size, 3])

    if augment:
        sketch, photo = augment_pair(sketch, photo)

    return sketch, photo


def build_tf_dataset(
    manifest_csv: Path = PAIRS_CSV_PATH,
    split: str = "train",
    batch_size: int = BATCH_SIZE,
    image_size: int = IMAGE_SIZE,
    shuffle: bool = True,
    augment: bool = True,
    limit: Optional[int] = None,
) -> tf.data.Dataset:
    """
    Construct an optimized tf.data.Dataset pipeline.
    Avoids loading full images into RAM; streams dynamically from storage.
    """
    manifest_csv = Path(manifest_csv)
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Pairs manifest not found at: {manifest_csv}")

    df = pd.read_csv(manifest_csv)
    subset_df = df[df["split"] == split]

    if limit is not None and limit > 0:
        subset_df = subset_df.head(limit)

    sketch_paths = subset_df["sketch_path"].tolist()
    photo_paths = subset_df["photo_path"].tolist()

    if not sketch_paths:
        raise ValueError(f"No pairs found in manifest for split='{split}'")

    dataset = tf.data.Dataset.from_tensor_slices((sketch_paths, photo_paths))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(sketch_paths), 1000), seed=SEED)

    def _mapper(s_path, p_path):
        return load_and_preprocess_tensor(
            s_path, p_path, image_size=image_size, augment=augment
        )

    dataset = dataset.map(_mapper, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset
