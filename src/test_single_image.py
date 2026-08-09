#!/usr/bin/env python3
"""
PITSEC: Test Single Image Compression

Compresses ONE image with all 24 libjpeg versions.
Use this to test which versions work BEFORE running the full dataset.

Usage:
    python test_single_image.py

Outputs:
    test_output/ folder with 24 versions of the first image
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import jpeglib
except ImportError:
    print("ERROR: jpeglib not installed")
    print("Install with: pip install jpeglib")
    sys.exit(1)

# All 24 libjpeg versions
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

CLUSTERS = {
    'C0': ['6b', 'turbo120', 'turbo150', 'turbo160', 'turbo170', 
           'turbo180', 'turbo190', 'mozjpeg201'],
    'C1': ['7', '8', '8a', '8b', '8c', '8d', '8e',
           '9', '9a', '9b', '9c', '9d'],
    'C2': ['9e', '9f'],
    'C3': ['mozjpeg300', 'mozjpeg403']
}

def get_cluster_for_version(version):
    """Return cluster for version."""
    for cluster, versions in CLUSTERS.items():
        if version in versions:
            return cluster
    return "UNKNOWN"

def test_compress_image(input_path, output_dir, version, quality=75):
    """
    Try to compress one image with a specific version.
    
    Returns:
        (success: bool, message: str)
    """
    try:
        # Read image
        img = Image.open(input_path)
        img_array = np.asarray(img)
        
        # Convert to RGB if needed
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]
        
        # Compress with version
        with jpeglib.version(version):
            jpeg = jpeglib.from_spatial(img_array)
            jpeg.quality = quality
            
            # Generate output filename
            stem = Path(input_path).stem
            output_file = output_dir / f"{stem}_{version}.jpeg"
            jpeg.write_spatial(str(output_file))
        
        # Get file size
        file_size = os.path.getsize(output_dir / f"{stem}_{version}.jpeg") / 1024  # KB
        return True, f"✓ {version:12} ({get_cluster_for_version(version)}) - {file_size:.1f} KB"
        
    except Exception as e:
        error_msg = str(e)
        if "not available" in error_msg.lower():
            return False, f"✗ {version:12} - NOT AVAILABLE"
        else:
            return False, f"✗ {version:12} - ERROR: {error_msg[:50]}"

def main():
    print("\n" + "="*70)
    print("  Testing Single Image Compression (All 24 Versions)")
    print("="*70 + "\n")
    
    # Input directory
    input_dir = Path("data/alaska_tif")
    output_dir = Path("data/test_output")
    
    # Find first image
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)
    
    tiff_files = sorted(input_dir.glob("*.tif"))
    if not tiff_files:
        print(f"❌ No TIFF files found in {input_dir}")
        sys.exit(1)
    
    test_image = tiff_files[0]
    print(f"Testing image: {test_image.name}")
    print(f"Size: {os.path.getsize(test_image) / 1024 / 1024:.1f} MB\n")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Test each version
    print("Compressing with all 24 versions...")
    print("-"*70)
    
    success_count = 0
    fail_count = 0
    results_by_cluster = {cluster: {"success": 0, "fail": 0} for cluster in CLUSTERS}
    
    for version in VERSION_LIST:
        cluster = get_cluster_for_version(version)
        success, message = test_compress_image(test_image, output_dir, version)
        
        print(message)
        
        if success:
            success_count += 1
            results_by_cluster[cluster]["success"] += 1
        else:
            fail_count += 1
            results_by_cluster[cluster]["fail"] += 1
    
    print("-"*70)
    
    # Summary
    print(f"\nSummary:")
    print(f"  Successful: {success_count}/24")
    print(f"  Failed:     {fail_count}/24")
    
    print(f"\nResults by cluster:")
    for cluster in ['C0', 'C1', 'C2', 'C3']:
        s = results_by_cluster[cluster]["success"]
        f = results_by_cluster[cluster]["fail"]
        total = len(CLUSTERS[cluster])
        pct = (s / total * 100) if total > 0 else 0
        print(f"  {cluster}: {s}/{total} available ({pct:.0f}%)")
    
    # List missing versions
    if fail_count > 0:
        print(f"\n⚠ Missing {fail_count} versions:")
        for version in VERSION_LIST:
            success, message = test_compress_image(test_image, output_dir, version)
            if not success:
                cluster = get_cluster_for_version(version)
                print(f"    - {version} ({cluster})")
        
        print(f"\nThese versions are NOT available on your system.")
        print(f"You should remove them from VERSION_LIST before running full compression.")
    else:
        print(f"\n✓ All 24 versions are available!")
    
    # Show generated files
    output_files = sorted(output_dir.glob("*.jpeg"))
    print(f"\nGenerated files: {len(output_files)} JPEGs in {output_dir}/")
    print(f"  Total size: {sum(os.path.getsize(f) for f in output_files) / 1024:.1f} KB")
    
    print("\n" + "="*70 + "\n")
    
    if fail_count == 0:
        print("✓ All versions work! Safe to run full dataset generation.\n")
    else:
        print(f"⚠ {fail_count} versions missing. Edit VERSION_LIST before full run.\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
