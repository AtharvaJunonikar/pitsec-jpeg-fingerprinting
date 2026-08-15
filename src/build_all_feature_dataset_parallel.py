#!/usr/bin/env python3
"""
Build a single CSV dataset from all JPEG images while preserving the existing
logic inside each feature extractor.

This script intentionally does NOT modify the underlying feature logic in:
- dct_features.py
- chroma_features.py
- ycbcr_features.py

It simply orchestrates these modules and writes one combined table.

Parallelism:
- Uses ProcessPoolExecutor for CPU-bound feature extraction.
- Automatically uses the available CPU count, capped by the number of files.
- Safe for Windows because it uses top-level functions only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple
import tempfile

import pandas as pd

# Import feature functions exactly as they are implemented in the project.
from chroma_features import extract_chroma_features, infer_cluster_from_filename
from dct_features import extract_dct_features
from ycbcr_features import extract_ycbcr_features
from ssim_features import compare, bsp

try:
    import jpeglib
except ImportError as exc:  # pragma: no cover
    raise ImportError("jpeglib is required for SSIM feature extraction.") from exc


VERSION_TO_CLUSTER = {
    "6b": "C0",
    "turbo120": "C0",
    "turbo130": "C0",
    "turbo140": "C0",
    "turbo150": "C0",
    "turbo200": "C0",
    "turbo210": "C0",
    "mozjpeg101": "C0",
    "mozjpeg201": "C0",
    "7": "C1",
    "8": "C1",
    "8a": "C1",
    "8b": "C1",
    "8c": "C1",
    "8d": "C1",
    "9": "C1",
    "9a": "C1",
    "9b": "C1",
    "9c": "C1",
    "9d": "C1",
    "9e": "C2",
    "9f": "C2",
    "mozjpeg300": "C3",
    "mozjpeg403": "C3",
}


def _find_image_files(root_dir: str | Path) -> List[Path]:
    """Return all JPG/JPEG paths under directory, sorted for stable order."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
    ]
    return sorted(files)


def _parse_source_and_version(file_path: Path) -> Tuple[str, str, str]:
    """Extract source image ID, version tag, and cluster label from filename."""
    name = file_path.name
    stem = file_path.stem

    # Example: 00042_7.jpeg -> source_id = 00042, version = 7
    if "_" in stem:
        source_id, version = stem.rsplit("_", 1)
        if version in VERSION_TO_CLUSTER:
            return source_id, version, VERSION_TO_CLUSTER[version]

    # Fallback if filenames are not versioned using the project schema
    source_id = stem
    version = "unknown"
    cluster = infer_cluster_from_filename(name)
    return source_id, version, cluster


def _compute_ssim_features_for_file(file_path: str | Path) -> dict:
    """Compute the project's SSIM recompression features using the existing logic."""
    path = Path(file_path)
    image = path

    # Preserve the original SSIM pipeline: compare the source image against each benchmark version
    # and compute 4 diff values + 4 normalized values. The logic comes from ssim_features.py.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        temp_path = tmpdir / "temp.jpeg"
        temp_path1 = tmpdir / "temp1.jpeg"

        results = []
        for version in bsp:
            file_cluster = None
            if version in ["6b", "turbo120", "turbo130", "turbo140", "turbo150", "turbo200", "turbo210", "mozjpeg101", "mozjpeg201"]:
                file_cluster = "C0"
            elif version in ["7", "8", "8a", "8b", "8c", "8d", "9", "9a", "9b", "9c", "9d"]:
                file_cluster = "C1"
            elif version in ["9e", "9f"]:
                file_cluster = "C2"
            elif version in ["mozjpeg300", "mozjpeg403"]:
                file_cluster = "C3"
            else:
                file_cluster = version

            with jpeglib.version(version):
                b = __import__("numpy").asarray(__import__("PIL").Image.open(image))
                c = jpeglib.from_spatial(b)
                c.write_spatial(str(temp_path))

            before = __import__("cv2").imread(str(temp_path))
            for version2 in bsp:
                with jpeglib.version(version2):
                    a = __import__("PIL").Image.open(temp_path)
                    b = __import__("numpy").asarray(a)
                    c = jpeglib.from_spatial(b)
                    c.write_spatial(str(temp_path1))
                after = __import__("cv2").imread(str(temp_path1))
                image_diff_avg = compare(before, after)
                results.append(image_diff_avg)

            # Keep the original pipeline in ssim_features.py consistent.
            # It computes min/max over the 4 benchmark comparisons and then normalizes the 4 values.
            if len(results) >= 4:
                pass

        # Reproduce the exact logic from ssim_features.py for this image.
        # The original script performs the 4 comparisons for each benchmark version and then appends
        # normalized values to the same list. We do the equivalent here for the 4 benchmark versions.
        benchmark_results = []
        for version in bsp:
            temp_list = []
            with jpeglib.version(version):
                b = __import__("numpy").asarray(__import__("PIL").Image.open(image))
                c = jpeglib.from_spatial(b)
                c.write_spatial(str(temp_path))
            before = __import__("cv2").imread(str(temp_path))
            for version2 in bsp:
                with jpeglib.version(version2):
                    a = __import__("PIL").Image.open(temp_path)
                    b = __import__("numpy").asarray(a)
                    c = jpeglib.from_spatial(b)
                    c.write_spatial(str(temp_path1))
                after = __import__("cv2").imread(str(temp_path1))
                temp_list.append(compare(before, after))

            lowest = min(temp_list)
            highest = max(temp_list)
            benchmark_results.extend(temp_list)
            benchmark_results.extend([(temp_list[i] - lowest) / (highest - lowest) for i in range(4)])

        # The original script stores exactly 8 values in the order: diff_C0..C3, norm_C0..C3
        # using the benchmark ordering ['6b', '7', '9e', 'mozjpeg300'].
        diff_values = benchmark_results[:4]
        norm_values = benchmark_results[4:8]
        return {
            "diff_C0": diff_values[0],
            "diff_C1": diff_values[1],
            "diff_C2": diff_values[2],
            "diff_C3": diff_values[3],
            "norm_C0": norm_values[0],
            "norm_C1": norm_values[1],
            "norm_C2": norm_values[2],
            "norm_C3": norm_values[3],
        }


def _extract_features_for_file(file_path: str | Path) -> dict:
    """Compute all feature families for one image, preserving their internal logic."""
    path = Path(file_path)

    dct_features = extract_dct_features(str(path)) or {}
    chroma_features = extract_chroma_features(str(path))
    ycbcr_features = extract_ycbcr_features(str(path))
    ssim_features = _compute_ssim_features_for_file(path)

    combined = {}
    combined.update(dct_features)
    combined.update(chroma_features)
    combined.update(ycbcr_features)
    combined.update(ssim_features)

    source_id, version, cluster = _parse_source_and_version(path)
    combined["file"] = path.name
    combined["source_id"] = source_id
    combined["version"] = version
    combined["LABEL"] = cluster

    return combined


def _worker_safe_files(files: Iterable[str]) -> List[dict]:
    """Batch wrapper used by the executor."""
    records = []
    for file_path in files:
        records.append(_extract_features_for_file(file_path))
    return records


def build_feature_dataset(
    input_dir: str,
    output_csv: str,
    limit: int | None = None,
    workers: int | None = None,
) -> pd.DataFrame:
    """Extract features for all images and write a combined CSV."""
    files = _find_image_files(input_dir)

    if limit is not None:
        files = files[:limit]

    if not files:
        raise FileNotFoundError(f"No JPEG images found under {input_dir}")

    if workers is None:
        workers = min(os.cpu_count() or 1, len(files))
    workers = max(1, min(workers, len(files)))

    print(f"Found {len(files)} images")
    print(f"Using {workers} worker processes")

    records: List[dict] = []
    chunk_size = max(1, math.ceil(len(files) / workers))

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        tasks = [
            executor.submit(_worker_safe_files, [str(p) for p in files[i : i + chunk_size]])
            for i in range(0, len(files), chunk_size)
        ]

        for future in concurrent.futures.as_completed(tasks):
            try:
                batch = future.result()
                records.extend(batch)
            except Exception as exc:  # pragma: no cover - runtime visibility
                print(f"Worker failed: {exc}", file=sys.stderr)
                raise

    if not records:
        raise RuntimeError("No feature rows were generated.")

    df = pd.DataFrame(records)
    if "LABEL" not in df.columns:
        raise RuntimeError("No LABEL column was produced. Feature extraction failed unexpectedly.")

    # Ensure stable column ordering.
    fixed_cols = ["file", "source_id", "version", "LABEL"]
    feature_cols = [c for c in df.columns if c not in fixed_cols]
    df = df[fixed_cols + feature_cols]

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} rows to {output_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract project feature families from JPEGs and save a CSV dataset.")
    parser.add_argument("--input-dir", type=str, default="data/compressed", help="Directory containing JPEG files")
    parser.add_argument("--output-csv", type=str, default="output/all_features.csv", help="Where to save the feature CSV")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for a quick test run")
    parser.add_argument("--workers", type=int, default=None, help="Override the number of worker processes")
    args = parser.parse_args()

    build_feature_dataset(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        limit=args.limit,
        workers=args.workers,
    )
