"""
PITSEC: Unified Feature Extraction Pipeline

Combines ALL features in a single pipeline:
  1. SSIM features (from bulk_classify.py)
  2. DCT features (dct_features.py)
  3. YCbCr features (ycbcr_features.py)
  4. Chroma Wrinkles (chroma_features.py)

Creates a single CSV with all features + labels for ML classification.

Workflow:
  1. Reads all compressed images (24 versions × N images)
  2. Extracts SSIM features (8 features)
  3. Extracts DCT features (15 features)
  4. Extracts YCbCr features (8 features)
  5. Extracts Chroma features (4 features)
  6. Outputs single CSV: all_features_combined.csv (35 total features)
  7. Ready for Decision Tree / Random Forest classification

Total Features: 35
  - SSIM: 8 (diff_C0-C3, norm_C0-C3)
  - DCT: 15 (ac_energy, dc_variance, etc. × 3 channels + combined)
  - YCbCr: 8 (Y, Cr, Cb SSIM + weighted average × 2)
  - Chroma: 4 (block variance, banding, etc. × 2 channels)

Expected Accuracy:
  - SSIM only: 95%
  - + DCT: 97%
  - + YCbCr: 98%
  - + Chroma: 99%+
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import os
import tempfile
import gc
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import jpeglib
import cv2
from PIL import Image
import torch

CUDA_AVAILABLE = torch.cuda.is_available()

# Import feature extraction modules
# These should be in src/feature_extraction/
from ssim_features import compare

# Reference encoders used by the original SSIM fingerprinting pipeline.
SSIM_ENCODERS = ['6b', '7', '9e', 'mozjpeg300']
from dct_features import extract_dct_features
from ycbcr_features import extract_ycbcr_features
from chroma_features import extract_chroma_features

# ===== CONFIGURATION =====

# Input/Output paths
COMPRESSED_DIR = Path("/mnt/c/pitsec-jpeg-fingerprinting/data/compressed")
OUTPUT_DIR = Path("/mnt/c/pitsec-jpeg-fingerprinting/output")
COMBINED_CSV = OUTPUT_DIR / "all_features_combined.csv"
LOG_FILE = "/mnt/c/pitsec-jpeg-fingerprinting/feature_extraction.log"

# Processing controls - OPTIMIZED FOR 18-CORE CPU
MAX_WORKERS = 12  # 18 cores - 2 for OS/I/O = 16 workers
CHECKPOINT_EVERY = 15  # Smaller batches to prevent memory accumulation
RESUME = True
MEMORY_THRESHOLD_GB = 20  # Trigger cleanup if memory exceeds this

# Version to cluster mapping (for labels)
VERSION_TO_CLUSTER = {
    # C0
    '6b': 'C0', 'turbo120': 'C0', 'turbo130': 'C0', 'turbo140': 'C0',
    'turbo150': 'C0', 'turbo200': 'C0', 'turbo210': 'C0',
    'mozjpeg101': 'C0', 'mozjpeg201': 'C0',
    # C1
    '7': 'C1', '8': 'C1', '8a': 'C1', '8b': 'C1', '8c': 'C1', '8d': 'C1',
    '9': 'C1', '9a': 'C1', '9b': 'C1', '9c': 'C1', '9d': 'C1',
    # C2
    '9e': 'C2', '9f': 'C2',
    # C3
    'mozjpeg300': 'C3', 'mozjpeg403': 'C3'
}

# Feature column names
FEATURE_COLUMNS = {
    'ssim': [
        'diff_C0', 'diff_C1', 'diff_C2', 'diff_C3',
        'norm_C0', 'norm_C1', 'norm_C2', 'norm_C3'
    ],
    'dct': [
        'ac_energy_y', 'dc_variance_y', 'ac_variance_y', 'zero_ratio_y', 'energy_conc_y',
        'ac_energy_cr', 'dc_variance_cr', 'ac_variance_cr', 'zero_ratio_cr', 'energy_conc_cr',
        'ac_energy_cb', 'dc_variance_cb', 'ac_variance_cb', 'zero_ratio_cb', 'energy_conc_cb'
    ],
    'ycbcr': [
        'Y_mean', 'Y_std', 'Y_var', 'Y_entropy',
        'Cb_mean', 'Cb_std', 'Cr_mean', 'Cr_std'
    ],
    'chroma': [
        'chroma_Cb_blockvar_mean', 'chroma_Cb_blockvar_std',
        'chroma_Cr_blockvar_mean', 'chroma_Cr_blockvar_std'
    ]
}

# ===== HELPER FUNCTIONS =====

def log_message(message, level="INFO"):
    """Log message to file and print."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] {message}"
    print(log_entry)
    with open(LOG_FILE, 'a') as f:
        f.write(log_entry + "\n")

def cleanup_memory(force=False):
    """Monitor and cleanup memory if threshold exceeded."""
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / (1024 ** 3)
    
    if force or mem_gb > MEMORY_THRESHOLD_GB:
        log_message(f"Memory cleanup triggered: {mem_gb:.2f}GB", "INFO")
        gc.collect()
        mem_gb_after = process.memory_info().rss / (1024 ** 3)
        log_message(f"After cleanup: {mem_gb_after:.2f}GB", "INFO")
        return mem_gb_after
    return mem_gb

def extract_version_from_filename(filename):
    """
    Extract libjpeg version from filename.
    
    Format: {image_id}_{version}.jpeg
    Example: 00001_6b.jpeg → '6b'
    
    Args:
        filename: string like "00001_6b.jpeg"
        
    Returns:
        version string like '6b'
    """
    parts = filename.replace('.jpeg', '').split('_')
    if len(parts) >= 2:
        return parts[-1]  # Last part after splitting by underscore
    return None

def get_cluster_label(version):
    """
    Get cluster label from version.
    
    Args:
        version: version string like '6b', '9e', etc.
        
    Returns:
        cluster: 'C0', 'C1', 'C2', or 'C3'
    """
    return VERSION_TO_CLUSTER.get(version, 'UNKNOWN')

def _gpu_dct_channel(channel, block_size=8):
    """Compute 8x8 DCT blocks on CUDA."""
    if not CUDA_AVAILABLE:
        return None

    h, w = channel.shape
    h = (h // block_size) * block_size
    w = (w // block_size) * block_size
    if h == 0 or w == 0:
        return None

    x = torch.as_tensor(channel[:h, :w], dtype=torch.float32, device="cuda")
    blocks = x.unfold(0, block_size, block_size).unfold(1, block_size, block_size)
    blocks = blocks.contiguous()

    n = block_size
    k = torch.arange(n, device="cuda", dtype=torch.float32).reshape(-1, 1)
    i = torch.arange(n, device="cuda", dtype=torch.float32).reshape(1, -1)
    basis = (2.0 / n) ** 0.5 * torch.cos(
        torch.pi * (2 * i + 1) * k / (2 * n)
    )
    basis[0, :] = (1.0 / n) ** 0.5

    dct = basis @ blocks @ basis.T
    dc = dct[..., 0, 0]
    ac = torch.cat(
        (
            dct[..., 0, 1:].reshape(-1, n - 1),
            dct[..., 1:, :].reshape(-1, (n - 1) * n)
        ),
        dim=1
    )
    return dc, ac


def _extract_dct_features_cuda(jpeg_path):
    """GPU-accelerated equivalent of the existing DCT feature extraction."""
    img_bgr = cv2.imread(str(jpeg_path))
    if img_bgr is None:
        raise ValueError(f"Could not read image: {jpeg_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)

    h, w = img_ycbcr.shape[:2]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h or pad_w:
        img_ycbcr = np.pad(
            img_ycbcr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect"
        )

    features = {}

    with torch.inference_mode():
        for name, channel in {
            "y": img_ycbcr[:, :, 0],
            "cr": img_ycbcr[:, :, 1],
            "cb": img_ycbcr[:, :, 2],
        }.items():
            dc, ac = _gpu_dct_channel(channel)

            total_energy = torch.sum(ac * ac)
            if total_energy.item() == 0:
                energy_conc = torch.tensor(0.0, device="cuda")
            else:
                flat = torch.abs(ac).reshape(-1)
                top_n = min(10, flat.numel())
                top = torch.topk(flat, k=top_n).values
                energy_conc = torch.sum(top * top) / total_energy

            features[f"ac_energy_{name}"] = float(torch.sum(ac * ac).cpu())
            features[f"dc_variance_{name}"] = float(torch.var(dc).cpu())
            features[f"ac_variance_{name}"] = float(torch.var(ac).cpu())
            features[f"zero_ratio_{name}"] = float(
                torch.mean((torch.abs(ac) < 0.5).float()).cpu()
            )
            features[f"energy_conc_{name}"] = float(energy_conc.cpu())

    features["ac_energy_avg"] = sum(
        features[f"ac_energy_{c}"] for c in ("y", "cr", "cb")
    ) / 3
    features["dc_variance_avg"] = sum(
        features[f"dc_variance_{c}"] for c in ("y", "cr", "cb")
    ) / 3
    features["zero_ratio_avg"] = sum(
        features[f"zero_ratio_{c}"] for c in ("y", "cr", "cb")
    ) / 3

    return features


def extract_ssim_features(jpeg_path):
    """
    Reproduce the original SSIM fingerprinting workflow using compare(img1, img2).

    The original SSIM code uses four reference encoders:
    6b, 7, 9e and mozjpeg300. For each source encoding, it compares that
    image against the four reference re-encodings and then uses the
    source-matched (diagonal) comparison as the four cluster features.
    """
    temp_before = None
    temp_after = None

    try:
        image = Image.open(jpeg_path)
        image_array = np.asarray(image)

        with tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False) as f1:
            temp_before = f1.name
        with tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False) as f2:
            temp_after = f2.name

        cluster_diffs = []

        for source_version in SSIM_ENCODERS:
            with jpeglib.version(source_version):
                compressed = jpeglib.from_spatial(image_array)
                compressed.write_spatial(temp_before)

            before = cv2.imread(temp_before)
            if before is None:
                raise ValueError(
                    f"Failed to read temporary JPEG for encoder {source_version}"
                )

            comparisons = []
            for target_version in SSIM_ENCODERS:
                with jpeglib.version(target_version):
                    intermediate = Image.open(temp_before)
                    intermediate_array = np.asarray(intermediate)
                    recompressed = jpeglib.from_spatial(intermediate_array)
                    recompressed.write_spatial(temp_after)

                after = cv2.imread(temp_after)
                if after is None:
                    raise ValueError(
                        f"Failed to read temporary JPEG for encoder {target_version}"
                    )

                comparisons.append(compare(before, after))

            # Source-matched comparison: C0→C0, C1→C1, C2→C2, C3→C3.
            cluster_diffs.append(comparisons[len(cluster_diffs)])

        lowest = min(cluster_diffs)
        highest = max(cluster_diffs)

        if highest == lowest:
            normalized = [0.0] * 4
        else:
            normalized = [
                (value - lowest) / (highest - lowest)
                for value in cluster_diffs
            ]

        return {
            **{f"diff_C{i}": float(cluster_diffs[i]) for i in range(4)},
            **{f"norm_C{i}": float(normalized[i]) for i in range(4)}
        }

    finally:
        for temp_path in (temp_before, temp_after):
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)


def extract_all_features(jpeg_path):
    """
    Extract exactly the 35 features used by the unified dataset.

    Each imported feature module is called with the argument type its public
    extraction function actually expects. Only the required feature keys are
    retained so extra fields from the standalone modules do not change the
    unified feature count.
    """
    try:
        features = {}

        # 1. SSIM: compare() requires two decoded image arrays, so reproduce
        # the original re-encode/compare workflow inside this combined file.
        ssim = extract_ssim_features(jpeg_path)
        if not ssim:
            return None
        features.update(ssim)

        # 2. DCT: extract_dct_features() accepts a JPEG path and returns
        # 18 values; retain the 15 per-channel features used by this dataset.
        dct = _extract_dct_features_cuda(jpeg_path) if CUDA_AVAILABLE else extract_dct_features(jpeg_path)
        if not dct:
            return None
        features.update({key: dct[key] for key in FEATURE_COLUMNS['dct']})

        # 3. YCbCr: extract_ycbcr_features() accepts a JPEG path and returns
        # 14 values; retain 8 actual statistics available from that function.
        ycbcr = extract_ycbcr_features(jpeg_path)
        if not ycbcr:
            return None
        features.update({key: ycbcr[key] for key in FEATURE_COLUMNS['ycbcr']})

        # 4. Chroma: extract_chroma_features() accepts a JPEG path and returns
        # 20 values; retain the four 8x8 block-variance features used here.
        chroma = extract_chroma_features(jpeg_path)
        if not chroma:
            return None
        features.update({key: chroma[key] for key in FEATURE_COLUMNS['chroma']})

        # Verify the feature contract before metadata is added.
        expected = sum(len(columns) for columns in FEATURE_COLUMNS.values())
        if len(features) != expected:
            raise ValueError(
                f"Expected {expected} features, got {len(features)}"
            )

        return features

    except Exception as e:
        log_message(f"Error extracting features from {jpeg_path}: {e}", "ERROR")
        return None


def main():
    """Main unified feature extraction pipeline."""
    
    print("\n" + "="*70)
    print("  PITSEC: Unified Feature Extraction Pipeline")
    print("  Combining SSIM + DCT + YCbCr + Chroma Features")
    print(f"  DCT acceleration: {'CUDA (RTX 4060)' if CUDA_AVAILABLE else 'CPU'}")
    print("="*70 + "\n")
    
    # Clear log
    if Path(LOG_FILE).exists():
        Path(LOG_FILE).unlink()
    
    log_message("Starting unified feature extraction")
    
    # ===== STEP 1: Verify Input =====
    
    print("STEP 1: Verifying input directory...")
    
    if not COMPRESSED_DIR.exists():
        print(f"❌ Compressed images directory not found: {COMPRESSED_DIR}")
        return
    
    jpeg_files = sorted(COMPRESSED_DIR.glob("*.jpeg"))
    if not jpeg_files:
        print(f"❌ No JPEG files found in {COMPRESSED_DIR}")
        return
    
    
    print(f"✓ Found {len(jpeg_files)} JPEG files\n")
    log_message(f"Found {len(jpeg_files)} JPEG files")
    
    # ===== STEP 2: Create Output Directory =====
    
    print("STEP 2: Creating output directory...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory ready: {OUTPUT_DIR}\n")
    
    # ===== STEP 3: Extract Features with Resume + Checkpoints =====

    processed_files = set()
    if RESUME and COMBINED_CSV.exists() and COMBINED_CSV.stat().st_size > 0:
        try:
            existing = pd.read_csv(COMBINED_CSV, usecols=['file'])
            processed_files = set(existing['file'].dropna().astype(str))
            print(f"  Resume: {len(processed_files)} images already completed")
        except Exception as e:
            print(f"  Warning: could not read existing checkpoint: {e}")
            print("  Starting from the beginning.")

    remaining_files = [f for f in jpeg_files if f.name not in processed_files]

    print("STEP 3: Extracting features from all images...")
    print("  This will process each image and extract all 35 features")
    print("  (SSIM + DCT + YCbCr + Chroma)")
    print(f"  Total images: {len(jpeg_files)}")
    print(f"  Already completed: {len(processed_files)}")
    print(f"  Remaining: {len(remaining_files)}")
    print(f"  CPU workers: {MAX_WORKERS}")
    print(f"  Checkpoint: every {CHECKPOINT_EVERY} images")
    print(f"  DCT acceleration: {'CUDA (RTX 4060)' if CUDA_AVAILABLE else 'CPU'}\n")

    output_columns = (
        FEATURE_COLUMNS['ssim'] +
        FEATURE_COLUMNS['dct'] +
        FEATURE_COLUMNS['ycbcr'] +
        FEATURE_COLUMNS['chroma'] +
        ['file', 'version', 'LABEL']
    )

    def process_one(jpeg_file):
        features = extract_all_features(jpeg_file)
        if features is None:
            return None
        version = extract_version_from_filename(jpeg_file.name)
        features['file'] = jpeg_file.name
        features['version'] = version
        features['LABEL'] = get_cluster_label(version)
        return features

    csv_exists = COMBINED_CSV.exists() and COMBINED_CSV.stat().st_size > 0
    pending_rows = []
    failed_count = 0
    completed_this_run = 0
    processed_count = 0

    if remaining_files:
        # ProcessPoolExecutor provides true parallelism (no GIL contention)
        # Each worker process has its own interpreter, avoiding GIL slowdown
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_one, f): f for f in remaining_files}
            pbar = tqdm(total=len(remaining_files), desc="Extracting", unit="image")

            for future in as_completed(futures):
                jpeg_file = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = None
                    log_message(f"Error processing {jpeg_file}: {e}", "ERROR")

                if result is None:
                    failed_count += 1
                else:
                    pending_rows.append(result)
                    completed_this_run += 1

                processed_count += 1
                pbar.update(1)

                if len(pending_rows) >= CHECKPOINT_EVERY:
                    checkpoint = pd.DataFrame(pending_rows).reindex(columns=output_columns)
                    checkpoint.to_csv(
                        COMBINED_CSV,
                        mode='a' if csv_exists else 'w',
                        header=not csv_exists,
                        index=False
                    )
                    csv_exists = True
                    pending_rows.clear()
                    log_message(
                        f"Checkpoint saved: {completed_this_run} completed this run, "
                        f"{failed_count} failed"
                    )
                    
                    # Periodic garbage collection to prevent memory accumulation
                    if processed_count % 50 == 0:
                        mem_gb = cleanup_memory()
                        pbar.set_postfix({'memory_gb': f'{mem_gb:.1f}'})

            pbar.close()

    if pending_rows:
        checkpoint = pd.DataFrame(pending_rows).reindex(columns=output_columns)
        checkpoint.to_csv(
            COMBINED_CSV,
            mode='a' if csv_exists else 'w',
            header=not csv_exists,
            index=False
        )
        pending_rows.clear()

    print("\n✓ Feature extraction complete")
    print(f"  Completed this run: {completed_this_run} images")
    print(f"  Failed this run: {failed_count} images\n")

    if not COMBINED_CSV.exists() or COMBINED_CSV.stat().st_size == 0:
        print("No output CSV was created.")
        return

    # ===== STEP 4: Create DataFrame =====
    
    print("STEP 4: Creating unified feature dataframe...")
    
    df = pd.read_csv(COMBINED_CSV)
    
    print(f"✓ DataFrame created")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}\n")
    
    # ===== STEP 5: Data Analysis =====
    
    print("STEP 5: Analyzing combined features...")
    
    print(f"\n  Class Distribution:")
    class_dist = df['LABEL'].value_counts()
    for cls, count in class_dist.items():
        pct = (count / len(df)) * 100
        print(f"    {cls}: {count:5d} ({pct:5.1f}%)")
    
    print(f"\n  Feature Statistics:")
    print(f"    Min values:\n{df.describe().loc['min']}")
    print(f"    Max values:\n{df.describe().loc['max']}")
    
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"  ⚠️  Warning: {missing_count} missing values detected!")
        # Fill NaN with 0
        df = df.fillna(0)
        print(f"     Filled with 0")
    
    # ===== STEP 6: Save Combined CSV =====
    
    print("\nSTEP 6: Verifying combined feature CSV...")
    print(f"✓ Checkpoint CSV already saved: {COMBINED_CSV}")
    print(f"  Total rows: {len(df)}")
    print(f"  Total columns: {len(df.columns)}\n")
    
    log_message(f"Saved combined features to {COMBINED_CSV}")
    
    # ===== STEP 7: Save Feature Metadata =====
    
    print("STEP 7: Saving feature metadata...")
    
    metadata = {
        "dataset_name": "PITSEC_Combined_Features",
        "total_images": len(df),
        "total_features": len(FEATURE_COLUMNS['ssim']) + len(FEATURE_COLUMNS['dct']) + 
                         len(FEATURE_COLUMNS['ycbcr']) + len(FEATURE_COLUMNS['chroma']),
        "feature_breakdown": {
            "SSIM": len(FEATURE_COLUMNS['ssim']),
            "DCT": len(FEATURE_COLUMNS['dct']),
            "YCbCr": len(FEATURE_COLUMNS['ycbcr']),
            "Chroma": len(FEATURE_COLUMNS['chroma'])
        },
        "class_distribution": class_dist.to_dict(),
        "columns": df.columns.tolist(),
        "timestamp": datetime.now().isoformat()
    }
    
    metadata_path = OUTPUT_DIR / "feature_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata to: {metadata_path}\n")
    
    # ===== STEP 8: Feature Summary =====
    
    print("STEP 8: Feature Summary")
    print("-" * 70)
    
    print(f"\nAll Features ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        if col not in ['file', 'version', 'LABEL']:
            print(f"  {i:2d}. {col:30s}")
    
    print("\n" + "="*70)
    print("  UNIFIED FEATURE EXTRACTION COMPLETE")
    print("="*70)
    
    print(f"\nOutput File: {COMBINED_CSV}")
    print(f"\nFeature Breakdown:")
    print(f"  SSIM Features:   {len(FEATURE_COLUMNS['ssim']):2d}")
    print(f"  DCT Features:    {len(FEATURE_COLUMNS['dct']):2d}")
    print(f"  YCbCr Features:  {len(FEATURE_COLUMNS['ycbcr']):2d}")
    print(f"  Chroma Features: {len(FEATURE_COLUMNS['chroma']):2d}")
    print(f"  {'─' * 30}")
    print(f"  Total Features:  {metadata['total_features']:2d}")
    
    print(f"\nClass Distribution:")
    for cls, count in sorted(class_dist.items()):
        bar = '█' * (count // 10)
        print(f"  {cls}: {bar} ({count} images)")
    
    print(f"\nNext Steps:")
    print(f"  1. Review {COMBINED_CSV}")
    print(f"  2. Train Decision Tree: python src/decision_tree_classifier.py")
    print(f"  3. Expected Accuracy: 99%+ (vs 95% with SSIM alone)")
    print(f"\n" + "="*70 + "\n")
    
    log_message("Unified feature extraction completed successfully")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Extraction interrupted by user")
        log_message("Extraction interrupted", "WARNING")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        log_message(f"Error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
