# src/ycbcr_features.py

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd

from chroma_features import infer_cluster_from_filename  # reuse same labelling logic


def _load_ycbcr(image_path: str):
    """Load image and return Y, Cb, Cr as float32 arrays."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycrcb[:, :, 0].astype(np.float32)
    Cr = ycrcb[:, :, 1].astype(np.float32)
    Cb = ycrcb[:, :, 2].astype(np.float32)
    return Y, Cb, Cr


def _channel_stats(channel: np.ndarray, prefix: str) -> Dict[str, float]:
    """Basic stats + entropy for one channel."""
    ch = channel.reshape(-1).astype(np.float32)

    mean = float(np.mean(ch))
    std = float(np.std(ch))
    var = float(np.var(ch))

    # Normalized histogram for entropy
    hist, _ = np.histogram(ch, bins=64, range=(0, 255), density=True)
    hist = hist + 1e-12  # avoid log(0)
    entropy = float(-np.sum(hist * np.log(hist)))

    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_var": var,
        f"{prefix}_entropy": entropy,
    }


def extract_ycbcr_features(image_path: str) -> Dict[str, float]:
    """
    Compute simple Y, Cb, Cr statistics for one JPEG image.

    Features:
      - Y_mean, Y_std, Y_var, Y_entropy
      - Cb_mean, Cb_std, Cb_var, Cb_entropy
      - Cr_mean, Cr_std, Cr_var, Cr_entropy
      - simple differences: Y_minus_Cb_mean, Y_minus_Cr_mean
    """
    Y, Cb, Cr = _load_ycbcr(image_path)

    feats: Dict[str, float] = {}
    feats.update(_channel_stats(Y, "Y"))
    feats.update(_channel_stats(Cb, "Cb"))
    feats.update(_channel_stats(Cr, "Cr"))

    # Simple comparative features
    feats["Y_minus_Cb_mean"] = feats["Y_mean"] - feats["Cb_mean"]
    feats["Y_minus_Cr_mean"] = feats["Y_mean"] - feats["Cr_mean"]

    return feats


def build_ycbcr_dataset(image_paths: List[str]) -> pd.DataFrame:
    """
    Compute YCbCr features and labels for a list of images and return a DataFrame.
    """
    records = []
    for path in image_paths:
        feats = extract_ycbcr_features(path)
        fname = Path(path).name

        record = dict(feats)
        record["file"] = fname
        record["LABEL"] = infer_cluster_from_filename(fname)

        records.append(record)

    df = pd.DataFrame(records)
    cols = [c for c in df.columns if c not in ("file", "LABEL")] + ["file", "LABEL"]
    return df[cols]


if __name__ == "__main__":
    # Small test: build a sample CSV for N_IMAGES images
    N_IMAGES = 5  # change to e.g. 5 when debugging

    root = Path("data/compressed")
    candidates = list(root.rglob("*.jpeg")) + list(root.rglob("*.jpg"))
    if not candidates:
        raise SystemExit("No .jpeg/.jpg files found under data/compressed")

    selected = [str(p) for p in candidates[:N_IMAGES]]
    print(f"Building YCbCr feature dataset from {len(selected)} images")

    df = build_ycbcr_dataset(selected)

    out_dir = Path("test_output")
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "ycbcr_features_sample.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote CSV to {csv_path}")