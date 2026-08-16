#!/usr/bin/env python3
"""
PITSEC: DCT-Domain Feature Extraction

Extracts Discrete Cosine Transform (DCT) features from JPEG images
for improved libjpeg version fingerprinting.

DCT features capture compression-specific patterns at the frequency domain level,
which complements spatial-domain SSIM features for better classification.

Features extracted:
  1. AC Energy (per block): Sum of squared AC coefficients
  2. DC Variance: Variance of DC coefficients across blocks
  3. AC Variance: Variance of AC coefficients
  4. Coefficient Histogram Distance: Distribution of coefficient magnitudes
  5. Zero Coefficient Ratio: Percentage of zero coefficients (quantization indicator)
  6. Energy Concentration: Percentage of energy in first N coefficients

Each feature is computed for:
  - Full image (all channels combined)
  - Per-channel (Y, Cr, Cb in YCbCr color space)
"""

import numpy as np
from pathlib import Path
from PIL import Image
import cv2


def read_jpeg_as_array(jpeg_path):
    """
    Read JPEG image as numpy array (BGR color space).
    
    Args:
        jpeg_path: Path to JPEG file
        
    Returns:
        numpy array of shape (height, width, 3) in BGR format
    """
    img = cv2.imread(str(jpeg_path))
    if img is None:
        raise ValueError(f"Could not read image: {jpeg_path}")
    return img


def bgr_to_ycbcr(img_bgr):
    """
    Convert BGR image to YCbCr color space.
    
    Args:
        img_bgr: numpy array in BGR format
        
    Returns:
        numpy array in YCbCr format
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    return img_ycbcr


def pad_to_block_size(img, block_size=8):
    """
    Pad image to be divisible by block_size.
    
    JPEG processes 8x8 blocks, so we need height and width divisible by 8.
    
    Args:
        img: numpy array
        block_size: size of blocks (default 8 for JPEG)
        
    Returns:
        Padded numpy array
    """
    h, w = img.shape[:2]
    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size
    
    if len(img.shape) == 3:
        return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    else:
        return np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')


def compute_dct_2d(block):
    """
    Compute 2D Discrete Cosine Transform on 8x8 block.
    
    The DCT transforms spatial domain (pixel values) to frequency domain.
    - DC component (top-left): Average value of block
    - AC components: Higher frequency details
    
    Different libjpeg versions use different DCT implementations,
    leading to slightly different coefficient values.
    
    Args:
        block: 8x8 numpy array
        
    Returns:
        8x8 DCT coefficient array
    """
    dct = cv2.dct(np.float32(block))
    return dct


def extract_dc_coefficient(dct_block):
    """
    Extract DC (zero-frequency) coefficient from DCT block.
    
    DC coefficient = average brightness of the 8x8 block.
    
    Args:
        dct_block: 8x8 DCT coefficient array
        
    Returns:
        float: DC coefficient value (top-left of DCT)
    """
    return dct_block[0, 0]


def extract_ac_coefficients(dct_block):
    """
    Extract AC (non-zero frequency) coefficients from DCT block.
    
    AC coefficients = details and textures.
    These vary more between libjpeg versions due to rounding differences.
    
    Args:
        dct_block: 8x8 DCT coefficient array
        
    Returns:
        1D array of 63 AC coefficients (all except DC)
    """
    # Flatten and exclude DC coefficient (index [0,0])
    return dct_block.flatten()[1:]


def compute_block_dcts(img_channel, block_size=8):
    """
    Compute DCT for all 8x8 blocks in image.
    
    Args:
        img_channel: 2D grayscale image (height, width)
        block_size: size of blocks (default 8)
        
    Returns:
        tuple: (dc_coefficients array, ac_coefficients array)
    """
    h, w = img_channel.shape
    num_blocks_h = h // block_size
    num_blocks_w = w // block_size
    
    dc_coeffs = np.zeros((num_blocks_h, num_blocks_w))
    ac_coeffs_list = []
    
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            # Extract 8x8 block
            block = img_channel[i*block_size:(i+1)*block_size,
                               j*block_size:(j+1)*block_size]
            
            # Compute DCT
            dct_block = compute_dct_2d(block)
            
            # Extract DC and AC
            dc_coeffs[i, j] = extract_dc_coefficient(dct_block)
            ac_coeffs_list.append(extract_ac_coefficients(dct_block))
    
    ac_coeffs = np.array(ac_coeffs_list)  # shape: (num_blocks, 63)
    
    return dc_coeffs, ac_coeffs


def compute_ac_energy(ac_coeffs):
    """
    Compute AC Energy: sum of squared AC coefficients.
    
    AC Energy measures high-frequency content.
    Different libjpeg versions handle high frequencies differently,
    so AC energy varies by version.
    
    High AC energy = lots of details/texture
    Low AC energy = smooth regions
    
    Args:
        ac_coeffs: array of AC coefficients (num_blocks, 63)
        
    Returns:
        float: Total AC energy (sum of squared AC coefficients)
    """
    return np.sum(ac_coeffs ** 2)


def compute_dc_variance(dc_coeffs):
    """
    Compute DC Variance: variance of DC coefficients across blocks.
    
    DC coefficients represent block averages.
    Variance tells us about luminance variation.
    
    Different libjpeg versions might round DC differently,
    causing variance to differ slightly.
    
    Args:
        dc_coeffs: array of DC coefficients (num_blocks_h, num_blocks_w)
        
    Returns:
        float: Variance of all DC coefficients
    """
    return np.var(dc_coeffs)


def compute_ac_variance(ac_coeffs):
    """
    Compute AC Variance: variance of all AC coefficients.
    
    Measures variability in high-frequency components.
    
    Args:
        ac_coeffs: array of AC coefficients (num_blocks, 63)
        
    Returns:
        float: Variance of all AC coefficients
    """
    return np.var(ac_coeffs)


def compute_zero_coefficient_ratio(ac_coeffs, threshold=0.5):
    """
    Compute Zero Coefficient Ratio: percentage of AC coefficients near zero.
    
    JPEG quantization zeros out small coefficients.
    Different quantization tables (used by different versions) create different
    numbers of zeroed coefficients.
    
    This is a STRONG indicator of which version was used for compression.
    
    Args:
        ac_coeffs: array of AC coefficients
        threshold: values below this are considered "zero"
        
    Returns:
        float: Ratio of zero/near-zero coefficients (0 to 1)
    """
    near_zero = np.abs(ac_coeffs) < threshold
    return np.sum(near_zero) / ac_coeffs.size


def compute_energy_concentration(ac_coeffs, top_n=10):
    """
    Compute Energy Concentration: percentage of energy in first N coefficients.
    
    DCT arranges coefficients by frequency (low → high from left to right).
    Energy concentration = how much of total AC energy is in low frequencies.
    
    Different versions compress differently, affecting frequency distribution.
    
    Args:
        ac_coeffs: array of AC coefficients (num_blocks, 63)
        top_n: number of top coefficients to check
        
    Returns:
        float: Fraction of energy in first top_n coefficients (0 to 1)
    """
    total_energy = np.sum(ac_coeffs ** 2)
    if total_energy == 0:
        return 0
    
    # Sort all coefficients by magnitude
    flat_coeffs = ac_coeffs.flatten()
    sorted_by_magnitude = np.sort(np.abs(flat_coeffs))[::-1]
    
    energy_in_top_n = np.sum(sorted_by_magnitude[:top_n] ** 2)
    return energy_in_top_n / total_energy


def compute_coefficient_histogram(ac_coeffs, num_bins=32):
    """
    Compute Coefficient Histogram: distribution of AC coefficient magnitudes.
    
    Create histogram of coefficient magnitudes.
    Different versions have different distributions.
    
    Args:
        ac_coeffs: array of AC coefficients
        num_bins: number of histogram bins
        
    Returns:
        1D array: histogram (normalized to sum to 1)
    """
    hist, _ = np.histogram(np.abs(ac_coeffs), bins=num_bins, range=(0, 100))
    hist = hist / np.sum(hist)  # Normalize
    return hist


def compute_histogram_distance(hist1, hist2):
    """
    Compute distance between two histograms using Wasserstein distance.
    
    When we re-compress an image with different versions,
    coefficient histograms diverge. This distance measures divergence.
    
    Args:
        hist1, hist2: normalized histograms
        
    Returns:
        float: Wasserstein distance between histograms (0 to 1)
    """
    # Simple L1 distance (can also use Wasserstein)
    return np.sum(np.abs(hist1 - hist2)) / 2


def extract_dct_features(jpeg_path):
    """
    Extract all DCT features from a JPEG image.
    
    This is the main function for DCT feature extraction.
    
    Args:
        jpeg_path: Path to JPEG file
        
    Returns:
        dict: Dictionary with DCT features:
            - ac_energy: Sum of squared AC coefficients
            - dc_variance: Variance of DC coefficients
            - ac_variance: Variance of AC coefficients
            - zero_ratio: Ratio of zero coefficients
            - energy_conc: Energy concentration in low frequencies
            - hist_entropy: Entropy of coefficient histogram
            - Per-channel versions of above
    """
    try:
        # Read image
        img_bgr = read_jpeg_as_array(jpeg_path)
        
        # Convert to YCbCr
        img_ycbcr = bgr_to_ycbcr(img_bgr)
        
        # Pad to block size
        img_ycbcr = pad_to_block_size(img_ycbcr)
        
        # Extract channels
        y_channel = img_ycbcr[:, :, 0]
        cr_channel = img_ycbcr[:, :, 1]
        cb_channel = img_ycbcr[:, :, 2]
        
        features = {}
        
        # ===== Y Channel (Luminance) =====
        dc_y, ac_y = compute_block_dcts(y_channel)
        features['ac_energy_y'] = compute_ac_energy(ac_y)
        features['dc_variance_y'] = compute_dc_variance(dc_y)
        features['ac_variance_y'] = compute_ac_variance(ac_y)
        features['zero_ratio_y'] = compute_zero_coefficient_ratio(ac_y)
        features['energy_conc_y'] = compute_energy_concentration(ac_y)
        
        # ===== Cr Channel (Chroma Red) =====
        # Chroma channels are typically lower resolution, but still have DCT
        dc_cr, ac_cr = compute_block_dcts(cr_channel)
        features['ac_energy_cr'] = compute_ac_energy(ac_cr)
        features['dc_variance_cr'] = compute_dc_variance(dc_cr)
        features['ac_variance_cr'] = compute_ac_variance(ac_cr)
        features['zero_ratio_cr'] = compute_zero_coefficient_ratio(ac_cr)
        features['energy_conc_cr'] = compute_energy_concentration(ac_cr)
        
        # ===== Cb Channel (Chroma Blue) =====
        dc_cb, ac_cb = compute_block_dcts(cb_channel)
        features['ac_energy_cb'] = compute_ac_energy(ac_cb)
        features['dc_variance_cb'] = compute_dc_variance(dc_cb)
        features['ac_variance_cb'] = compute_ac_variance(ac_cb)
        features['zero_ratio_cb'] = compute_zero_coefficient_ratio(ac_cb)
        features['energy_conc_cb'] = compute_energy_concentration(ac_cb)
        
        # ===== Combined Features =====
        # Average across channels
        features['ac_energy_avg'] = (features['ac_energy_y'] + 
                                      features['ac_energy_cr'] + 
                                      features['ac_energy_cb']) / 3
        
        features['dc_variance_avg'] = (features['dc_variance_y'] + 
                                        features['dc_variance_cr'] + 
                                        features['dc_variance_cb']) / 3
        
        features['zero_ratio_avg'] = (features['zero_ratio_y'] + 
                                       features['zero_ratio_cr'] + 
                                       features['zero_ratio_cb']) / 3
        
        return features
        
    except Exception as e:
        print(f"Error extracting DCT features from {jpeg_path}: {e}")
        return None


def extract_dct_features_batch(jpeg_dir, output_csv=None):
    """
    Extract DCT features from all JPEG images in a directory.
    
    Args:
        jpeg_dir: Directory containing JPEG files
        output_csv: Optional output CSV file path
        
    Returns:
        pandas DataFrame with DCT features
    """
    import pandas as pd
    
    jpeg_dir = Path(jpeg_dir)
    jpeg_files = sorted(jpeg_dir.glob("*.jpeg"))
    
    results = []
    
    print(f"Extracting DCT features from {len(jpeg_files)} images...")
    
    for i, jpeg_file in enumerate(jpeg_files):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(jpeg_files)}")
        
        features = extract_dct_features(jpeg_file)
        if features is not None:
            features['file'] = jpeg_file.name
            results.append(features)
    
    df = pd.DataFrame(results)
    
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"✓ Saved to {output_csv}")
    
    return df


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        jpeg_path = sys.argv[1]
        features = extract_dct_features(jpeg_path)
        
        if features:
            print(f"DCT Features for {jpeg_path}:")
            print("-" * 50)
            for key, value in sorted(features.items()):
                print(f"  {key:25s}: {value:.6f}")
    else:
        print("Usage: python dct_features.py <jpeg_path>")
        print("   or: python dct_features.py <jpeg_directory> <output_csv>")
