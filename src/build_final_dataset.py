#!/usr/bin/env python3
"""Build the PITSEC final feature dataset with correct SSIM row structure.

This script reproduces the original SSIM workflow from ssim_features.py:
- Four SSIM rows per image, using source encoders ['6b', '7', '9e', 'mozjpeg300'].
- Each row has LABEL = C0, C1, C2, C3 respectively.
- DCT, YCbCr, and chroma features are computed once per image and repeated.

Run from the repository root:
    python src/build_final_dataset.py --limit 5 --workers 2

Full run:
    python src/build_final_dataset.py --workers 4
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import jpeglib
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity
from tqdm import tqdm

from chroma_features import extract_chroma_features
from dct_features import extract_dct_features
from ycbcr_features import extract_ycbcr_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "compressed"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "all_features_combined.csv"

SSIM_ENCODERS = ["6b", "7", "9e", "mozjpeg300"]
CLUSTER_LABELS = ["C0", "C1", "C2", "C3"]

FEATURE_COLUMNS = {
    "ssim": [
        "diff_C0", "diff_C1", "diff_C2", "diff_C3",
        "norm_C0", "norm_C1", "norm_C2", "norm_C3",
    ],
    "dct": [
        "ac_energy_y", "dc_variance_y", "ac_variance_y", "zero_ratio_y", "energy_conc_y",
        "ac_energy_cr", "dc_variance_cr", "ac_variance_cr", "zero_ratio_cr", "energy_conc_cr",
        "ac_energy_cb", "dc_variance_cb", "ac_variance_cb", "zero_ratio_cb", "energy_conc_cb",
    ],
    "ycbcr": [
        "Y_mean", "Y_std", "Y_var", "Y_entropy",
        "Cb_mean", "Cb_std", "Cr_mean", "Cr_std",
    ],
    "chroma": [
        "chroma_Cb_blockvar_mean", "chroma_Cb_blockvar_std",
        "chroma_Cr_blockvar_mean", "chroma_Cr_blockvar_std",
    ],
}

ALL_FEATURE_COLUMNS = [
    column
    for group in FEATURE_COLUMNS.values()
    for column in group
]
OUTPUT_COLUMNS = ALL_FEATURE_COLUMNS + ["file", "version", "LABEL"]


def compare(img1: np.ndarray, img2: np.ndarray) -> float:
    """Use the same RGB-channel SSIM calculation as ssim_features.py."""
    scores = []
    for channel in range(3):
        score, _ = structural_similarity(
            img1[:, :, channel], img2[:, :, channel], full=True
        )
        scores.append((1.0 - score) * 100.0)
    return float(sum(scores) / len(scores))


def extract_ssim_features(image_path: Path) -> Dict[str, List[float]]:
    """Reproduce the original SSIM workflow: four rows per image.

    Returns a dict with keys:
        diff_C0, diff_C1, diff_C2, diff_C3
        norm_C0, norm_C1, norm_C2, norm_C3
    Each value is a list of four floats, one per source encoder.
    """
    image = Image.open(image_path)
    image_array = np.asarray(image)

    with tempfile.TemporaryDirectory(prefix="pitsec_ssim_") as temp_dir:
        before_path = Path(temp_dir) / "before.jpeg"
        after_path = Path(temp_dir) / "after.jpeg"

        raw_rows = []

        for source_version in SSIM_ENCODERS:
            with jpeglib.version(source_version):
                encoded = jpeglib.from_spatial(image_array)
                encoded.write_spatial(str(before_path))

            before = cv2.imread(str(before_path))
            if before is None:
                raise ValueError(f"Could not read temporary image: {before_path}")

            comparisons = []
            for target_version in SSIM_ENCODERS:
                with jpeglib.version(target_version):
                    intermediate = Image.open(before_path)
                    encoded = jpeglib.from_spatial(np.asarray(intermediate))
                    encoded.write_spatial(str(after_path))

                after = cv2.imread(str(after_path))
                if after is None:
                    raise ValueError(f"Could not read temporary image: {after_path}")
                comparisons.append(compare(before, after))

            low = min(comparisons)
            high = max(comparisons)
            if high == low:
                normalized = [0.0] * len(comparisons)
            else:
                normalized = [(value - low) / (high - low) for value in comparisons]

            raw_rows.append({
                "diff": comparisons,
                "norm": normalized,
            })

    result: Dict[str, List[float]] = {
        "diff_C0": [row["diff"][0] for row in raw_rows],
        "diff_C1": [row["diff"][1] for row in raw_rows],
        "diff_C2": [row["diff"][2] for row in raw_rows],
        "diff_C3": [row["diff"][3] for row in raw_rows],
        "norm_C0": [row["norm"][0] for row in raw_rows],
        "norm_C1": [row["norm"][1] for row in raw_rows],
        "norm_C2": [row["norm"][2] for row in raw_rows],
        "norm_C3": [row["norm"][3] for row in raw_rows],
    }
    return result


def process_one(image_path_string: str) -> List[Dict[str, object]]:
    """Process one JPEG file and return four rows (one per SSIM source encoder)."""
    image_path = Path(image_path_string)
    try:
        ssim_values = extract_ssim_features(image_path)

        dct_values = extract_dct_features(image_path)
        if dct_values is None:
            raise ValueError("DCT extractor returned None")

        ycbcr_values = extract_ycbcr_features(str(image_path))
        chroma_values = extract_chroma_features(str(image_path))

        rows = []
        for label_index, label in enumerate(CLUSTER_LABELS):
            row: Dict[str, object] = {}

            row["diff_C0"] = float(ssim_values["diff_C0"][label_index])
            row["diff_C1"] = float(ssim_values["diff_C1"][label_index])
            row["diff_C2"] = float(ssim_values["diff_C2"][label_index])
            row["diff_C3"] = float(ssim_values["diff_C3"][label_index])
            row["norm_C0"] = float(ssim_values["norm_C0"][label_index])
            row["norm_C1"] = float(ssim_values["norm_C1"][label_index])
            row["norm_C2"] = float(ssim_values["norm_C2"][label_index])
            row["norm_C3"] = float(ssim_values["norm_C3"][label_index])

            for column in FEATURE_COLUMNS["dct"]:
                row[column] = float(dct_values[column])
            for column in FEATURE_COLUMNS["ycbcr"]:
                row[column] = float(ycbcr_values[column])
            for column in FEATURE_COLUMNS["chroma"]:
                row[column] = float(chroma_values[column])

            row["file"] = image_path.name
            row["version"] = SSIM_ENCODERS[label_index]
            row["LABEL"] = label
            rows.append(row)

        return rows
    except Exception as error:
        return [{
            "__error__": f"{image_path.name}: {type(error).__name__}: {error}"
        }]


def find_images(input_dir: Path) -> List[Path]:
    return sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 2) - 1)),
        help="Number of CPU processes; start with 2 or 4 on a MacBook.",
    )
    args = parser.parse_args()

    images = find_images(args.input)
    if args.limit is not None:
        images = images[:args.limit]
    if not images:
        raise SystemExit(f"No JPEG images found in {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    error_path = args.output.with_suffix(".errors.csv")
    rows = []
    errors = []

    print(f"Images: {len(images)}")
    print(f"Workers: {args.workers}")
    print(f"Output: {args.output}")
    print(f"Expected rows: {len(images) * 4}")

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_one, str(path)) for path in images]
        with tqdm(total=len(futures), desc="Extracting", unit="image") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result and "__error__" in result[0]:
                    errors.append({"error": result[0]["__error__"]})
                else:
                    rows.extend(result)
                pbar.update(1)

    elapsed = time.time() - start_time

    if not rows:
        raise SystemExit("No rows were produced. Check the error file or traceback.")

    df = pd.DataFrame(rows).reindex(columns=OUTPUT_COLUMNS)
    df = df.sort_values(["file", "LABEL"]).reset_index(drop=True)
    df.to_csv(args.output, index=False)

    if errors:
        pd.DataFrame(errors).to_csv(error_path, index=False)

    print(f"\nCompleted in {elapsed:.1f} seconds")
    print(f"Rows written: {len(df)}")
    print(f"Columns written: {len(df.columns)}")
    print(f"Feature columns: {len(ALL_FEATURE_COLUMNS)}")
    print(f"Saved: {args.output}")
    if errors:
        print(f"Failed images: {len(errors)}")
        print(f"Errors: {error_path}")


if __name__ == "__main__":
    main()