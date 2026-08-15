import os
import tempfile
from pathlib import Path

import cv2
import jpeglib
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
CSV_INPUT = Path("/mnt/c/pitsec-jpeg-fingerprinting/output/all_features_combined.csv")
CSV_OUTPUT = Path("/mnt/c/pitsec-jpeg-fingerprinting/output/all_features_combined_ssim_fixed.csv")
IMAGE_DIR = Path("/mnt/c/pitsec-jpeg-fingerprinting/data/compressed")

# Optional old SSIM-only cache. If it exists, its values are
# copied into the full output before processing continues.
SSIM_CACHE = CSV_OUTPUT.with_suffix(".checkpoint.csv")

# Full CSV is saved after this many newly calculated images.
CHECKPOINT_EVERY = 100

SSIM_COLUMNS = [
    "diff_C0", "diff_C1", "diff_C2", "diff_C3",
    "norm_C0", "norm_C1", "norm_C2", "norm_C3",
]

# Exact reference versions from ssim_features.py
BSP = ["6b", "7", "9e", "mozjpeg300"]


def compare(img1, img2):
    """Copied directly from ssim_features.py."""
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


def calculate_ssim_features(image_path, source_version):
    """
    Uses the same inner SSIM calculation as ssim_features.py.
    The source_version is taken from the existing CSV row so the
    repair works for all 24 JPEG versions in the combined dataset.
    """
    temp_dir = tempfile.TemporaryDirectory()
    try:
        temp_path = os.path.join(temp_dir.name, "temp.jpeg")
        temp_path1 = os.path.join(temp_dir.name, "temp1.jpeg")

        im = Image.open(image_path)

        # Same operation as the original outer 'version' loop.
        with jpeglib.version(source_version):
            b = np.asarray(im)
            c = jpeglib.from_spatial(b)
            c.write_spatial(temp_path)

        before = cv2.imread(temp_path)
        if before is None:
            raise RuntimeError(f"cv2.imread failed: {temp_path}")

        # Exact original inner loop.
        temp_list = []
        for version2 in BSP:
            with jpeglib.version(version2):
                a = Image.open(temp_path)
                b = np.asarray(a)
                c = jpeglib.from_spatial(b)
                c.write_spatial(temp_path1)

            after = cv2.imread(temp_path1)
            if after is None:
                raise RuntimeError(f"cv2.imread failed: {temp_path1}")

            image_diff_avg = compare(before, after)
            temp_list.append(image_diff_avg)

        # Exact original normalization.
        lowest = min(temp_list)
        highest = max(temp_list)
        if highest == lowest:
            raise ZeroDivisionError(
                f"SSIM normalization denominator is zero for {image_path.name}: {temp_list}"
            )

        normalized = [
            (temp_list[i] - lowest) / (highest - lowest)
            for i in range(4)
        ]

        return temp_list + normalized
    finally:
        temp_dir.cleanup()


def save_full_csv(df, path):
    """Always save the COMPLETE 38-column dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def main():
    print("=" * 70)
    print("PITSEC: Repair SSIM columns in existing feature dataset")
    print("=" * 70)

    if not CSV_INPUT.exists():
        raise FileNotFoundError(f"Input CSV not found: {CSV_INPUT}")
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    # --------------------------------------------------------
    # Load the original full dataset.
    # --------------------------------------------------------
    df = pd.read_csv(CSV_INPUT)
    original_columns = df.columns.tolist()

    required = ["file", "version"] + SSIM_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Input rows:    {len(df)}")
    print(f"Input columns: {len(df.columns)}")

    # --------------------------------------------------------
    # If the previous 9-column cache exists, use it only as a
    # cache. It is NEVER used as the output dataset.
    # --------------------------------------------------------
    completed = set()
    if SSIM_CACHE.exists():
        cache = pd.read_csv(SSIM_CACHE)
        cache_required = ["file"] + SSIM_COLUMNS
        if all(c in cache.columns for c in cache_required):
            cache = cache.drop_duplicates("file", keep="last")
            cache_index = cache.set_index("file")
            cache_files = set(cache["file"].astype(str))

            # Only apply cache entries that exist in the original dataset.
            mask = df["file"].astype(str).isin(cache_files)
            for col in SSIM_COLUMNS:
                df.loc[mask, col] = (
                    df.loc[mask, "file"].astype(str).map(cache_index[col])
                )
            completed = cache_files & set(df["file"].astype(str))
            print(f"Loaded cached SSIM values: {len(completed)}")
        else:
            print("Ignoring incompatible SSIM cache.")

    # --------------------------------------------------------
    # If a full output already exists, resume from it.
    # --------------------------------------------------------
    if CSV_OUTPUT.exists():
        existing = pd.read_csv(CSV_OUTPUT)
        if existing.columns.tolist() == original_columns and len(existing) == len(df):
            # Existing output is authoritative for completed rows.
            existing_index = existing.set_index("file")
            common = set(df["file"].astype(str)) & set(existing["file"].astype(str))
            for col in SSIM_COLUMNS:
                values = existing_index[col]
                mask = df["file"].astype(str).isin(common)
                df.loc[mask, col] = df.loc[mask, "file"].astype(str).map(values)
            print(f"Loaded existing full output: {len(common)} rows")

    # Determine completion by presence of all 8 SSIM values.
    done_mask = df[SSIM_COLUMNS].notna().all(axis=1)
    # If the input already has old SSIM values, they are NOT considered
    # repaired unless they came from the cache/full output. Therefore, when
    # no cache/output was found, force all rows to be recalculated.
    if not SSIM_CACHE.exists() and not CSV_OUTPUT.exists():
        done_mask[:] = False

    pending = df.loc[~done_mask].copy()
    print(f"Rows requiring SSIM calculation: {len(pending)}")

    # --------------------------------------------------------
    # Calculate SSIM and save the COMPLETE dataset every checkpoint.
    # --------------------------------------------------------
    errors = []
    newly_done = 0

    for idx, row in tqdm(
        pending.iterrows(),
        total=len(pending),
        desc="SSIM",
        unit="image",
    ):
        filename = str(row["file"])
        source_version = str(row["version"])
        image_path = IMAGE_DIR / filename

        try:
            values = calculate_ssim_features(image_path, source_version)
            df.loc[idx, SSIM_COLUMNS] = values
            newly_done += 1

        except Exception as exc:
            errors.append({
                "file": filename,
                "version": source_version,
                "error": repr(exc),
            })

        if newly_done >= CHECKPOINT_EVERY:
            save_full_csv(df, CSV_OUTPUT)
            print(f"\nCheckpoint saved: {CSV_OUTPUT}")
            print(f"Completed: {int(df[SSIM_COLUMNS].notna().all(axis=1).sum())}/{len(df)}")
            newly_done = 0

    # Final full output.
    save_full_csv(df, CSV_OUTPUT)

    if errors:
        error_path = CSV_OUTPUT.with_suffix(".errors.csv")
        pd.DataFrame(errors).to_csv(error_path, index=False)
        print(f"\nFailed rows: {len(errors)}")
        print(f"Error report: {error_path}")

    # --------------------------------------------------------
    # Strict final validation.
    # --------------------------------------------------------
    if df.columns.tolist() != original_columns:
        raise RuntimeError("Column list changed unexpectedly.")

    if len(df) != len(pd.read_csv(CSV_INPUT)):
        raise RuntimeError("Row count changed unexpectedly.")

    missing_ssim = int(df[SSIM_COLUMNS].isna().any(axis=1).sum())
    if missing_ssim:
        raise RuntimeError(
            f"Output still has {missing_ssim} rows with missing SSIM values."
        )

    check = pd.read_csv(CSV_OUTPUT, nrows=5)
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Output:  {CSV_OUTPUT}")
    print(f"Rows:    {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("Column structure preserved:", df.columns.tolist() == original_columns)
    print("\nFirst 5 output rows:")
    print(check.to_string(index=False))


if __name__ == "__main__":
    main()
