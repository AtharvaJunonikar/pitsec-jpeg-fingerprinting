# src/chroma_features.py

from pathlib import Path
from typing import Dict, List
import pandas as pd
from sklearn.model_selection import train_test_split

import cv2
import numpy as np

C0_ENCODERS = ['6b', 'turbo120', 'turbo130', 'turbo140', 'turbo150', 'turbo200', 'turbo210', 'mozjpeg101', 'mozjpeg201']  # copy c0 list
C1_ENCODERS = ['7', '8', '8a', '8b', '8c', '8d', '9', '9a', '9b', '9c', '9d']  # copy c1 list
C2_ENCODERS = ['9e', '9f']  # copy c2 list
C3_ENCODERS = ['mozjpeg300', 'mozjpeg403']  # copy c3 list


def infer_cluster_from_filename(filename: str) -> str:
    """
    Infer C0/C1/C2/C3 from the encoder tag in the filename.
    Uses the same encoder lists as bulk_classify.py.
    """
    name = filename.lower()

    for tag in C0_ENCODERS:
        if tag in name:
            return "C0"
    for tag in C1_ENCODERS:
        if tag in name:
            return "C1"
    for tag in C2_ENCODERS:
        if tag in name:
            return "C2"
    for tag in C3_ENCODERS:
        if tag in name:
            return "C3"

    raise ValueError(f"Could not infer cluster from filename: {filename}")

def _ensure_gray_or_color(image: np.ndarray) -> np.ndarray:
    """Ensure image is 3‑channel BGR."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"Unsupported image shape: {image.shape}")


def extract_chroma_features(image_path: str) -> Dict[str, float]:
    """
    Compute simple chroma‑pattern features for one JPEG image.

    Features (all from Cb and Cr channels):
      - mean, standard deviation, and variance
      - edge energy (Sobel gradients)
      - local 8x8 block variance statistics

    Returns a dict mapping feature names to float values.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    # Read image (BGR)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    bgr = _ensure_gray_or_color(bgr)

    # Convert BGR -> YCrCb (OpenCV’s order is Y, Cr, Cb)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0].astype(np.float32)
    Cr = ycrcb[:, :, 1].astype(np.float32)
    Cb = ycrcb[:, :, 2].astype(np.float32)

    features: Dict[str, float] = {}

    # 1. Global statistics
    for channel, name in [(Cb, "Cb"), (Cr, "Cr")]:
        features[f"chroma_{name}_mean"] = float(np.mean(channel))
        features[f"chroma_{name}_std"] = float(np.std(channel))
        features[f"chroma_{name}_var"] = float(np.var(channel))

    # 2. Edge energy (Sobel gradients) in chroma channels
    for channel, name in [(Cb, "Cb"), (Cr, "Cr")]:
        gx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        features[f"chroma_{name}_edge_mean"] = float(np.mean(mag))
        features[f"chroma_{name}_edge_std"] = float(np.std(mag))

    # 3. Block‑wise variance (8x8) – captures “wrinkliness”
    def block_variances(channel: np.ndarray, block: int = 8) -> np.ndarray:
        h, w = channel.shape
        h = (h // block) * block
        w = (w // block) * block
        chan = channel[:h, :w]
        chan_blocks = chan.reshape(h // block, block, w // block, block)
        # move block dims together and compute var within blocks
        chan_blocks = np.moveaxis(chan_blocks, 1, 2)  # (n_h, n_w, b, b)
        vars_flat = np.var(chan_blocks, axis=(2, 3)).ravel()
        return vars_flat

    for channel, name in [(Cb, "Cb"), (Cr, "Cr")]:
        vars_flat = block_variances(channel)
        features[f"chroma_{name}_blockvar_mean"] = float(np.mean(vars_flat))
        features[f"chroma_{name}_blockvar_std"] = float(np.std(vars_flat))
        features[f"chroma_{name}_blockvar_max"] = float(np.max(vars_flat))

    # Optionally: include Y channel basic stats (can help later)
    features["luma_Y_mean"] = float(np.mean(Y))
    features["luma_Y_std"] = float(np.std(Y))

    return features


def extract_chroma_features_batch(image_paths: List[str]) -> List[Dict[str, float]]:
    """
    Convenience wrapper: apply extract_chroma_features to many images.
    """
    return [extract_chroma_features(p) for p in image_paths]

def build_chroma_dataset(image_paths) -> pd.DataFrame:
    records = []
    for path in image_paths:
        feats = extract_chroma_features(path)
        fname = Path(path).name

        record = dict(feats)
        record["file"] = fname
        record["LABEL"] = infer_cluster_from_filename(fname)

        records.append(record)

    df = pd.DataFrame(records)
    cols = [c for c in df.columns if c not in ("file", "LABEL")] + ["file", "LABEL"]
    return df[cols]

if __name__ == "__main__":
    # When you run:
    #   python src/chroma_features.py
    # this will compute chroma features for N_IMAGES and write a CSV (and ARFF if dataframe2arff is available).
    from pathlib import Path

    # How many images to include in this small test dataset
    N_IMAGES = 5  # change to 5, 20, etc.

    root = Path("data/compressed")
    candidates = list(root.rglob("*.jpeg")) + list(root.rglob("*.jpg"))
    if not candidates:
        raise SystemExit("No .jpeg/.jpg files found under data/compressed")

    selected = [str(p) for p in candidates[:N_IMAGES]]
    print(f"Building chroma feature dataset from {len(selected)} images")

    
    df = build_chroma_dataset(selected)

    # Make sure output directory exists
    out_dir = Path("test_output")
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "chroma_features_sample.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote CSV to {csv_path}")

    # Optional ARFF export if you have dataframe2arff in your project
    try:
        from bulk_classify import dataframe2arff  # adjust path if needed

        arff_path = out_dir / "chroma_features_sample.arff"
        dataframe2arff(df, str(arff_path))
        print(f"Wrote ARFF to {arff_path}")
    except ImportError:
        print("dataframe2arff not available; skipping ARFF export")