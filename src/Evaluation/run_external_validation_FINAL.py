#!/usr/bin/env python3
"""
External validation for the SSIM-based PITSEC Random Forest.

Input:
    data/external_originals/natural_*.jpg

Output:
    data/external_compressed/
    output/external_validation/external_features.csv
    output/external_validation/external_results.txt
    output/external_validation/external_confusion_matrix.csv

Each external original image is re-encoded using:
    6b          -> C0
    7           -> C1
    9e          -> C2
    mozjpeg300  -> C3

The saved Random Forest is evaluated only. It is never retrained here.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import cv2
import joblib
import jpeglib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.metrics import structural_similarity
from tqdm import tqdm
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from chroma_features import extract_chroma_features
from dct_features import extract_dct_features
from ycbcr_features import extract_ycbcr_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ORIGINAL_DIR = PROJECT_ROOT / "data" / "natural_images"
COMPRESSED_DIR = PROJECT_ROOT / "data" / "external_compressed"
OUTPUT_DIR = PROJECT_ROOT / "output" / "external_validation"

MODEL_PATH = (
    PROJECT_ROOT
    / "output"
    / "model_evaluation_results"
    / "random_forest_model.joblib"
)

MODEL_FEATURES_PATH = (
    PROJECT_ROOT
    / "output"
    / "model_evaluation_results"
    / "model_metadata.json"
)

SSIM_ENCODERS = ["6b", "7", "9e", "mozjpeg300"]
LABELS = ["C0", "C1", "C2", "C3"]

FEATURE_COLUMNS = [
    'diff_C0', 'diff_C1', 'diff_C2', 'diff_C3',
    'norm_C0', 'norm_C1', 'norm_C2', 'norm_C3',
    'ac_energy_y', 'dc_variance_y', 'ac_variance_y', 'zero_ratio_y', 'energy_conc_y',
    'ac_energy_cr', 'dc_variance_cr', 'ac_variance_cr', 'zero_ratio_cr', 'energy_conc_cr',
    'ac_energy_cb', 'dc_variance_cb', 'ac_variance_cb', 'zero_ratio_cb', 'energy_conc_cb',
    'Y_mean', 'Y_std', 'Y_var', 'Y_entropy',
    'Cb_mean', 'Cb_std', 'Cr_mean', 'Cr_std',
    'chroma_Cb_blockvar_mean', 'chroma_Cb_blockvar_std', 'chroma_Cr_blockvar_mean', 'chroma_Cr_blockvar_std'
]


def compare(img1, img2):
    """Exact SSIM comparison logic used by the training feature extractor."""
    img1_r = img1[:, :, 0]
    img1_g = img1[:, :, 1]
    img1_b = img1[:, :, 2]

    img2_r = img2[:, :, 0]
    img2_g = img2[:, :, 1]
    img2_b = img2[:, :, 2]

    (score_r, diff_r) = structural_similarity(img1_r, img2_r, full=True)
    (score_g, diff_g) = structural_similarity(img1_g, img2_g, full=True)
    (score_b, diff_b) = structural_similarity(img1_b, img2_b, full=True)

    image_diff_r = (1 - score_r) * 100
    image_diff_g = (1 - score_g) * 100
    image_diff_b = (1 - score_b) * 100
    image_diff_avg = (image_diff_r + image_diff_g + image_diff_b) / 3

    return image_diff_avg


def load_as_rgb_array(path: Path) -> np.ndarray:
    """Load any supported image and convert it to uint8 RGB."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def encode_reference_image(rgb_array: np.ndarray, encoder: str, output_path: Path) -> None:
    """Encode one RGB image with a selected jpeglib encoder version."""
    with jpeglib.version(encoder):
        jpeg = jpeglib.from_spatial(rgb_array)
        jpeg.write_spatial(str(output_path))


def create_external_recompressions(original_paths: list[Path]) -> list[dict]:
    """
    Creates exactly one JPEG variant per external original image.

    The four source encoders are distributed deterministically across the
    external originals. Each source image therefore produces exactly one
    feature row, while that row still contains the four SSIM comparisons
    (C0, C1, C2, C3).
    """
    COMPRESSED_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for index, original_path in enumerate(
        tqdm(original_paths, desc="Creating external JPEGs", unit="image")
    ):
        rgb_array = load_as_rgb_array(original_path)

        # One source encoder / class per original image.
        source_encoder = SSIM_ENCODERS[index % len(SSIM_ENCODERS)]
        label = LABELS[index % len(LABELS)]

        destination = COMPRESSED_DIR / f"{original_path.stem}_{source_encoder}.jpeg"
        encode_reference_image(rgb_array, source_encoder, destination)

        records.append({
            "original_file": original_path.name,
            "file": destination.name,
            "path": str(destination),
            "ssim_source_version": source_encoder,
            "LABEL": label,
        })

    return records


def extract_ssim_row_features(jpeg_path: Path, source_encoder: str) -> dict:
    """
    Generate the SSIM features using the exact logic from ssim_features(2).py.

    For the supplied source JPEG:
      1. Re-encode using its source encoder.
      2. Decode that temporary JPEG.
      3. Re-encode the decoded image with each of the four BSP encoders.
      4. Compare before/after using the exact three-channel SSIM calculation.
      5. Normalize the four SSIM differences using min/max normalization.
    """
    with tempfile.TemporaryDirectory(prefix="pitsec_external_ssim_") as temp_dir:
        temp_dir = Path(temp_dir)
        temp_path = temp_dir / "temp.jpeg"
        temp_path1 = temp_dir / "temp1.jpeg"

        # Equivalent to:
        # with jpeglib.version(version):
        #     b = np.asarray(im)
        #     c = jpeglib.from_spatial(b)
        #     c.write_spatial(temp_path)
        with Image.open(jpeg_path) as im:
            with jpeglib.version(source_encoder):
                b = np.asarray(im)
                c = jpeglib.from_spatial(b)
                c.write_spatial(str(temp_path))

        before = cv2.imread(str(temp_path))
        if before is None:
            raise ValueError(f"Could not read temporary image: {temp_path}")

        temp_list = []

        # Exact BSP list used by the training extractor.
        for target_encoder in SSIM_ENCODERS:
            with jpeglib.version(target_encoder):
                a = Image.open(temp_path)
                try:
                    b = np.asarray(a)
                finally:
                    a.close()

                c = jpeglib.from_spatial(b)
                c.write_spatial(str(temp_path1))

            after = cv2.imread(str(temp_path1))
            if after is None:
                raise ValueError(f"Could not read temporary image: {temp_path1}")

            image_diff_avg = compare(before, after)
            temp_list.append(image_diff_avg)

        lowest = min(temp_list)
        highest = max(temp_list)

        # This is the same normalization used in ssim_features(2).py.
        # Keep the guard to avoid division by zero for a degenerate row.
        if highest == lowest:
            normalized = [0.0] * 4
        else:
            normalized = [
                (temp_list[i] - lowest) / (highest - lowest)
                for i in range(4)
            ]

    return {
        "diff_C0": temp_list[0],
        "diff_C1": temp_list[1],
        "diff_C2": temp_list[2],
        "diff_C3": temp_list[3],
        "norm_C0": normalized[0],
        "norm_C1": normalized[1],
        "norm_C2": normalized[2],
        "norm_C3": normalized[3],
    }


def extract_feature_rows(records: list[dict]) -> pd.DataFrame:
    """Extract the same 35 training features for all external JPEG rows."""
    rows = []

    for record in tqdm(records, desc="Extracting external features", unit="image"):
        jpeg_path = Path(record["path"])

        ssim = extract_ssim_row_features(
            jpeg_path,
            record["ssim_source_version"],
        )

        dct = extract_dct_features(jpeg_path)
        if dct is None:
            raise ValueError(f"DCT extraction failed: {jpeg_path}")

        ycbcr = extract_ycbcr_features(str(jpeg_path))
        chroma = extract_chroma_features(str(jpeg_path))

        row = {}
        row.update(ssim)

        for column in FEATURE_COLUMNS[8:23]:
            row[column] = float(dct[column])

        for column in FEATURE_COLUMNS[23:31]:
            row[column] = float(ycbcr[column])

        for column in FEATURE_COLUMNS[31:]:
            row[column] = float(chroma[column])

        row.update({
            "original_file": record["original_file"],
            "file": record["file"],
            "version": record["ssim_source_version"],
            "LABEL": record["LABEL"],
        })

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise SystemExit(f"Saved model not found: {MODEL_PATH}")

    if not MODEL_FEATURES_PATH.exists():
        raise SystemExit(f"Model feature list not found: {MODEL_FEATURES_PATH}")

    original_paths = sorted(ORIGINAL_DIR.glob("cat_*.jpg"))

    if len(original_paths) != 20:
        raise SystemExit(
            f"Expected exactly 20 natural_*.jpg images in {ORIGINAL_DIR}, "
            f"found {len(original_paths)}."
        )

    print("=" * 72)
    print("PITSEC Preliminary External Validation")
    print("=" * 72)
    print(f"External originals: {len(original_paths)}")
    print(f"Expected validation rows: {len(original_paths)}")
    print("Source: Kaggle Natural Images pilot subset")
    print("Model: frozen all-features Random Forest; no retraining")

    records = create_external_recompressions(original_paths)
    records_df = pd.DataFrame(records)
    records_df.to_csv(OUTPUT_DIR / "external_recompression_manifest.csv", index=False)

    df = extract_feature_rows(records)
    df = df.reindex(columns=FEATURE_COLUMNS + ["original_file", "file", "version", "LABEL"])

    if len(df) != len(original_paths):
        raise RuntimeError(
            f"Expected exactly one feature row per original image: "
            f"{len(original_paths)} rows expected, found {len(df)}."
        )

    features_path = OUTPUT_DIR / "external_features.csv"
    df.to_csv(features_path, index=False)

    missing = int(df[FEATURE_COLUMNS].isna().sum().sum())
    infinite = int(np.isinf(df[FEATURE_COLUMNS].to_numpy()).sum())

    if missing or infinite:
        raise RuntimeError(
            f"External feature dataset contains missing={missing}, infinite={infinite}"
        )

    model = joblib.load(MODEL_PATH)
    model_features = json.loads(MODEL_FEATURES_PATH.read_text(encoding="utf-8"))

    if model_features != FEATURE_COLUMNS:
        raise RuntimeError(
            "Saved model feature order does not match the external feature schema."
        )

    predictions = model.predict(df[model_features])

    accuracy = accuracy_score(df["LABEL"], predictions)
    matrix = confusion_matrix(df["LABEL"], predictions, labels=LABELS)
    report_text = classification_report(
        df["LABEL"],
        predictions,
        labels=LABELS,
        digits=4,
    )

    results = df[["original_file", "file", "version", "LABEL"]].copy()
    results["prediction"] = predictions
    results["correct"] = results["LABEL"] == results["prediction"]

    results.to_csv(OUTPUT_DIR / "external_predictions.csv", index=False)
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(
        OUTPUT_DIR / "external_confusion_matrix.csv"
    )

    results_text = OUTPUT_DIR / "external_results.txt"

    with results_text.open("w", encoding="utf-8") as handle:
        handle.write("PITSEC Preliminary External Validation\n")
        handle.write("=" * 50 + "\n")
        handle.write("External source: Kaggle Natural Images dataset\n")
        handle.write(f"External original images: {len(original_paths)}\n")
        handle.write(f"External validation rows: {len(df)} (one row per original image)\n")
        handle.write("Reference encoders: 6b, 7, 9e, mozjpeg300\n")
        handle.write("Labels: C0, C1, C2, C3\n")
        handle.write("Model: frozen all-features Random Forest; no retraining\n")
        handle.write(f"Missing features: {missing}\n")
        handle.write(f"Infinite features: {infinite}\n")
        handle.write(f"Accuracy: {accuracy:.4%}\n\n")
        handle.write("Confusion matrix (rows=true, columns=predicted; C0,C1,C2,C3):\n")
        handle.write(str(matrix))
        handle.write("\n\nClassification report:\n")
        handle.write(report_text)

    print("\n" + "=" * 72)
    print("EXTERNAL VALIDATION RESULT")
    print("=" * 72)
    print(f"External rows: {len(df)}")
    print(f"Accuracy: {accuracy:.4%}")
    print("Confusion matrix (rows=true, columns=predicted; C0,C1,C2,C3):")
    print(matrix)
    print("\nClassification report:")
    print(report_text)
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()