#!/usr/bin/env python3
"""
PITSEC: Generate 24-Version JPEG Dataset (With All 24 Versions)

This script takes uncompressed TIFF images and compresses each one
with all 24 libjpeg versions, creating a labeled dataset for ML.

CORRECTED: Added missing version '8e' (8 + 12 + 2 + 2 = 24)

Usage:
    python src/generate_24version_dataset.py

Requirements:
    - jpeglib (pip install jpeglib)
    - All 2000+ images in data/alaska_tif/
    
Output:
    - data/compressed/ folder with 24 versions of each image
    - Dataset ready for bulk_classify.py feature extraction

Time: ~2-4 hours for 2000 images (depends on CPU)
"""

import os
import sys
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image

try:
    import jpeglib
except ImportError:
    print("ERROR: jpeglib not installed")
    print("Install with: pip install jpeglib")
    sys.exit(1)

# ===== CONFIGURATION =====

# Input and output directories
INPUT_DIR = Path("data/alaska_tif")
OUTPUT_DIR = Path("data/compressed")
LOG_FILE = "dataset_generation.log"

# All 24 libjpeg versions (from Benes 2022 paper)
# CORRECTED: Added '8e' to Cluster C1 (was missing)
VERSION_LIST = [
    # Cluster C0: Legacy upsampling family (8 versions)
    '6b', 'turbo120', 'turbo150', 'turbo160', 'turbo170',
    'turbo180', 'turbo190', 'mozjpeg201',
    
    # Cluster C1: DCT-scaling family (12 versions)
    '7', '8', '8a', '8b', '8c', '8d', '8e',
    '9', '9a', '9b', '9c', '9d',
    
    # Cluster C2: New chrominance quantization (2 versions)
    '9e', '9f',
    
    # Cluster C3: mozjpeg progressive family (2 versions)
    'mozjpeg300', 'mozjpeg403'
]

# Version clusters (for reference)
CLUSTERS = {
    'C0': ['6b', 'turbo120', 'turbo150', 'turbo160', 'turbo170', 
           'turbo180', 'turbo190', 'mozjpeg201'],
    'C1': ['7', '8', '8a', '8b', '8c', '8d', '8e',  # ADDED '8e'
           '9', '9a', '9b', '9c', '9d'],
    'C2': ['9e', '9f'],
    'C3': ['mozjpeg300', 'mozjpeg403']
}

# JPEG quality factor
QUALITY_FACTOR = 75

# ===== HELPER FUNCTIONS =====

def log_message(message, level="INFO"):
    """Write message to log file and print to console."""
    timestamp = __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] {message}"
    print(log_entry)
    with open(LOG_FILE, 'a') as f:
        f.write(log_entry + "\n")

def get_cluster_for_version(version):
    """Return cluster (C0, C1, C2, C3) for a given version."""
    for cluster, versions in CLUSTERS.items():
        if version in versions:
            return cluster
    return "UNKNOWN"

def compress_with_version(input_path, output_path, version, quality=75):
    """
    Compress a single image with specified libjpeg version.
    
    Args:
        input_path: Path to input TIFF image
        output_path: Path to save compressed JPEG
        version: libjpeg version (e.g., '6b', '7', '9e', '8e')
        quality: JPEG quality factor (1-100)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read image as numpy array
        img = Image.open(input_path)
        img_array = np.asarray(img)
        
        # Convert to RGB if needed (jpeglib prefers RGB)
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]
        
        # Use jpeglib to compress with specific version
        with jpeglib.version(version):
            # Create JPEG encoder from spatial (uncompressed) data
            jpeg = jpeglib.from_spatial(img_array)
            
            # Set quality (simplified approach)
            # jpeglib automatically handles quantization based on quality
            jpeg.quality = quality
            
            # Write to file
            jpeg.write_spatial(str(output_path))
        
        return True
        
    except Exception as e:
        log_message(f"Error compressing {input_path} with version {version}: {str(e)}", "ERROR")
        return False

def main():
    """Main pipeline: compress all images with all 24 versions."""
    
    print("\n" + "="*70)
    print("  PITSEC: Generate 24-Version JPEG Dataset")
    print("="*70 + "\n")
    
    # Clear previous log
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    
    log_message("Starting dataset generation pipeline")
    
    # ===== STEP 1: Verify Input Directory =====
    
    print("STEP 1: Verifying input directory...")
    
    if not INPUT_DIR.exists():
        log_message(f"ERROR: Input directory not found: {INPUT_DIR}", "ERROR")
        print(f"❌ Please ensure images are in: {INPUT_DIR.absolute()}")
        sys.exit(1)
    
    tiff_files = sorted(INPUT_DIR.glob("*.tif"))
    if not tiff_files:
        log_message(f"ERROR: No TIFF files found in {INPUT_DIR}", "ERROR")
        print(f"❌ No images found in {INPUT_DIR}")
        sys.exit(1)
    
    num_images = len(tiff_files)
    log_message(f"Found {num_images} TIFF images in {INPUT_DIR}")
    print(f"✓ Found {num_images} images")
    print(f"  Input: {INPUT_DIR.absolute()}\n")
    
    # ===== STEP 2: Create Output Directory =====
    
    print("STEP 2: Creating output directory...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_message(f"Output directory created/verified: {OUTPUT_DIR}")
    print(f"✓ Output: {OUTPUT_DIR.absolute()}\n")
    
    # ===== STEP 3: Compress All Images with All Versions =====
    
    print("STEP 3: Compressing images...")
    print(f"  Versions: {len(VERSION_LIST)}")
    print(f"  Images: {num_images}")
    print(f"  Total compressions: {len(VERSION_LIST) * num_images}\n")
    
    log_message(f"Starting compression: {len(VERSION_LIST)} versions × {num_images} images")
    
    compression_stats = {
        "total": len(VERSION_LIST) * num_images,
        "successful": 0,
        "failed": 0,
        "by_version": {}
    }
    
    # Progress bar for all compressions
    total_ops = len(VERSION_LIST) * num_images
    pbar = tqdm(total=total_ops, desc="Compressing", unit="image", ncols=70)
    
    # Iterate through versions
    for version in VERSION_LIST:
        compression_stats["by_version"][version] = {"success": 0, "fail": 0}
        cluster = get_cluster_for_version(version)
        
        # Iterate through images
        for tiff_file in tiff_files:
            # Generate output filename with version label
            # Example: 00001_7.jpeg (image 00001 compressed with v7)
            stem = tiff_file.stem  # e.g., "00001"
            output_filename = f"{stem}_{version}.jpeg"
            output_path = OUTPUT_DIR / output_filename
            
            # Skip if already exists
            if output_path.exists():
                compression_stats["by_version"][version]["success"] += 1
                pbar.update(1)
                continue
            
            # Compress
            success = compress_with_version(tiff_file, output_path, version, QUALITY_FACTOR)
            
            if success:
                compression_stats["successful"] += 1
                compression_stats["by_version"][version]["success"] += 1
            else:
                compression_stats["failed"] += 1
                compression_stats["by_version"][version]["fail"] += 1
            
            pbar.update(1)
    
    pbar.close()
    
    print()  # Newline after progress bar
    
    # ===== STEP 4: Verify and Report =====
    
    print("STEP 4: Verifying dataset...")
    
    # Count generated files
    jpeg_files = sorted(OUTPUT_DIR.glob("*.jpeg"))
    log_message(f"Total JPEG files generated: {len(jpeg_files)}")
    print(f"✓ Generated {len(jpeg_files)} JPEG files")
    
    # Check by version
    print("\nCompression results by version:")
    print("-" * 70)
    
    for version in VERSION_LIST:
        cluster = get_cluster_for_version(version)
        versions_files = sorted(OUTPUT_DIR.glob(f"*_{version}.jpeg"))
        stats = compression_stats["by_version"][version]
        status = "✓" if stats["fail"] == 0 else "⚠"
        print(f"{status} v{version:12} ({cluster}): {len(versions_files):4} files "
              f"({stats['success']} success, {stats['fail']} failed)")
    
    print("-" * 70)
    
    # ===== STEP 5: Generate Metadata =====
    
    print("\nSTEP 5: Generating metadata...")
    
    metadata = {
        "dataset_name": "ALASKA_v2_24Versions",
        "num_images": num_images,
        "num_versions": len(VERSION_LIST),
        "total_compressions": num_images * len(VERSION_LIST),
        "quality_factor": QUALITY_FACTOR,
        "versions": VERSION_LIST,
        "clusters": CLUSTERS,
        "input_directory": str(INPUT_DIR.absolute()),
        "output_directory": str(OUTPUT_DIR.absolute()),
        "compression_stats": compression_stats,
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }
    
    # Save metadata
    metadata_path = OUTPUT_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    log_message(f"Metadata saved to: {metadata_path}")
    print(f"✓ Metadata saved: {metadata_path}")
    
    # ===== STEP 6: Final Summary =====
    
    print("\n" + "="*70)
    print("  DATASET GENERATION COMPLETE")
    print("="*70)
    
    print(f"\nSummary:")
    print(f"  Input images:       {num_images}")
    print(f"  Versions:           {len(VERSION_LIST)}")
    print(f"  Total compressions: {total_ops}")
    print(f"  Successful:         {compression_stats['successful']}")
    print(f"  Failed:             {compression_stats['failed']}")
    print(f"  Output location:    {OUTPUT_DIR.absolute()}")
    
    print(f"\nCluster breakdown:")
    for cluster in ['C0', 'C1', 'C2', 'C3']:
        cluster_versions = CLUSTERS[cluster]
        print(f"  {cluster}: {len(cluster_versions)} versions - {', '.join(cluster_versions)}")
    
    print(f"\nDataset ready for feature extraction!")
    print(f"Next step: python src/bulk_classify.py")
    
    print("\n" + "="*70 + "\n")
    
    log_message("Dataset generation completed successfully")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Dataset generation interrupted by user")
        log_message("Dataset generation interrupted", "WARNING")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {str(e)}")
        log_message(f"Fatal error: {str(e)}", "ERROR")
        sys.exit(1)
