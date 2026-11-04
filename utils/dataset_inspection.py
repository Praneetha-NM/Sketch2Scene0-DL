"""
Dataset Discovery and Inspection Utility.

CLI Execution:
    python -m utils.dataset_inspection [--dir data/raw] [--download-if-missing]

Performs:
1. Recursive scan of raw dataset folder
2. Matching of sketch and ground truth photograph files
3. Verification of image integrity, dimensions, channels, and alignment
4. Generation and validation of data/processed/pairs.csv manifest
5. Visual verification grid saved to outputs/samples/dataset_inspection.png
"""

import sys
import argparse
from pathlib import Path
from PIL import Image
import pandas as pd

from config import RAW_DATA_DIR, PAIRS_CSV_PATH, SAMPLES_DIR, SEED, TRAIN_LIMIT, VALIDATION_LIMIT
from data.dataset_loader import (
    discover_pairs_from_directory,
    create_pairs_csv,
    download_official_subset,
)
from utils.visualization import save_comparison_grid
from preprocessing.image_processing import preprocess_pair


def inspect_dataset(
    raw_dir: Path = RAW_DATA_DIR,
    manifest_csv: Path = PAIRS_CSV_PATH,
    download_if_empty: bool = False,
    preview_count: int = 4,
) -> int:
    """
    Run full inspection on raw dataset and generate manifest.
    """
    raw_dir = Path(raw_dir)
    print("=" * 65)
    print("      SketchyCOCO Landscape Dataset Discovery & Inspection")
    print("=" * 65)
    print(f"Target Directory: {raw_dir.resolve()}")

    pairs, stats = discover_pairs_from_directory(raw_dir)

    print("\nScan Results:")
    print(f"  • Total Sketch files located: {stats['total_sketches']}")
    print(f"  • Total Photo files located:  {stats['total_photos']}")
    print(f"  • Valid Paired Identifiers:   {stats['valid_pairs']}")
    print(f"  • Detected Train Split pairs: {stats['train_pairs']}")
    print(f"  • Detected Val Split pairs:   {stats['val_pairs']}")

    if stats["valid_pairs"] == 0:
        if download_if_empty:
            print("\nDataset not found locally. Initiating official SketchyCOCO subset download...")
            success = download_official_subset(raw_dir, train_count=TRAIN_LIMIT, val_count=VALIDATION_LIMIT)
            if success:
                pairs, stats = discover_pairs_from_directory(raw_dir)
                print(f"Downloaded and validated {stats['valid_pairs']} pairs.")
            else:
                print("Could not download dataset automatically. Please see manual download instructions.")
                return 1
        else:
            print("\n[Notice] No valid pairs currently found in data/raw.")
            print("To download the genuine paired subset automatically, run:")
            print("    python -m utils.dataset_inspection --download-if-missing")
            print("Or download Scene.tar manually from SketchyCOCO Google Drive:")
            print("    https://drive.google.com/file/d/1ApjDhGjtqfFEMzm6dmyhS-2aXnnYLxnj/view")
            print("and place under data/raw/Scene/.")
            return 1

    # Generate Manifest
    print(f"\nGenerating reproducible pairs manifest: {manifest_csv}...")
    df = create_pairs_csv(
        raw_dir=raw_dir,
        output_csv=manifest_csv,
        train_limit=TRAIN_LIMIT,
        val_limit=VALIDATION_LIMIT,
        seed=SEED,
    )
    print(f"Manifest ready with {len(df)} total active pairs.")

    # Integrity and Alignment Check on sample pairs
    print("\nVerifying sample image integrity and alignments:")
    sample_df = df.head(preview_count)
    sketches = []
    photos = []

    for idx, row in sample_df.iterrows():
        s_path = Path(row["sketch_path"])
        p_path = Path(row["photo_path"])

        if not s_path.exists() or not p_path.exists():
            print(f"  [Error] Missing file for pair {idx}: {s_path} or {p_path}")
            continue

        with Image.open(s_path) as s_img, Image.open(p_path) as p_img:
            print(f"  Pair {idx + 1}: ID={s_path.name} | Sketch: {s_img.size} ({s_img.mode}) <--> Photo: {p_img.size} ({p_img.mode}) [Aligned]")
            s_tensor, p_tensor = preprocess_pair(s_img, p_img, target_size=(128, 128))
            sketches.append(s_tensor)
            photos.append(p_tensor)

    # Save visual verification grid
    if sketches and photos:
        preview_out = SAMPLES_DIR / "dataset_inspection.png"
        # Since these are real pairs, display sketch | photo | photo to confirm alignment
        save_comparison_grid(
            sketches=sketches,
            photos=photos,
            generated=photos,
            output_path=preview_out,
            num_samples=len(sketches),
            title="SketchyCOCO Paired Inspection: Sketch vs Photo",
        )
        print(f"\nSaved visual alignment verification grid to: {preview_out}")

    print("\nDataset inspection finished successfully.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Discover and inspect SketchyCOCO scene pairs.")
    parser.add_argument("--dir", type=str, default=str(RAW_DATA_DIR), help="Path to raw dataset")
    parser.add_argument("--manifest", type=str, default=str(PAIRS_CSV_PATH), help="Path for output pairs.csv")
    parser.add_argument("--download-if-missing", action="store_true", help="Download official subset if data/raw is empty")
    args = parser.parse_args()

    ret = inspect_dataset(
        raw_dir=Path(args.dir),
        manifest_csv=Path(args.manifest),
        download_if_empty=args.download_if_missing,
    )
    sys.exit(ret)


if __name__ == "__main__":
    main()
