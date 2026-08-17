#!/usr/bin/env python3
"""Leakage-safe Random Forest experiments for the SSIM-based PITSEC dataset.

The target is LABEL: C0, C1, C2, C3. All rows sharing the same image_id
(the text before the final underscore in `file`) stay in the same split.

Run from the project root:
    python src/train_ssim_random_forest.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit


#PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = Path("/mnt/c/pitsec-jpeg-fingerprinting/output/all_features_combined.csv")
OUTPUT_DIR = Path("/mnt/c/pitsec-jpeg-fingerprinting/output/random_forest_results")
RANDOM_STATE = 42

SSIM_FEATURES = [
    "diff_C0", "diff_C1", "diff_C2", "diff_C3",
    "norm_C0", "norm_C1", "norm_C2", "norm_C3",
]

DCT_FEATURES = [
    "ac_energy_y", "dc_variance_y", "ac_variance_y", "zero_ratio_y", "energy_conc_y",
    "ac_energy_cr", "dc_variance_cr", "ac_variance_cr", "zero_ratio_cr", "energy_conc_cr",
    "ac_energy_cb", "dc_variance_cb", "ac_variance_cb", "zero_ratio_cb", "energy_conc_cb",
]

YCBCR_FEATURES = [
    "Y_mean", "Y_std", "Y_var", "Y_entropy",
    "Cb_mean", "Cb_std", "Cr_mean", "Cr_std",
]

CHROMA_FEATURES = [
    "chroma_Cb_blockvar_mean", "chroma_Cb_blockvar_std",
    "chroma_Cr_blockvar_mean", "chroma_Cr_blockvar_std",
]

EXPERIMENTS = {
    "ssim_only": SSIM_FEATURES,
    "ssim_dct": SSIM_FEATURES + DCT_FEATURES,
    "ssim_ycbcr_chroma": SSIM_FEATURES + YCBCR_FEATURES + CHROMA_FEATURES,
    "all_features": SSIM_FEATURES + DCT_FEATURES + YCBCR_FEATURES + CHROMA_FEATURES,
}


def write_report(path: Path, name: str, features: list[str], accuracy: float,
                 matrix, report_text: str, importance: pd.DataFrame,
                 train_rows: int, test_rows: int, train_groups: int, test_groups: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"Experiment: {name}\n")
        handle.write(f"Features: {len(features)}\n")
        handle.write(f"Train rows: {train_rows}\n")
        handle.write(f"Test rows: {test_rows}\n")
        handle.write(f"Train image IDs: {train_groups}\n")
        handle.write(f"Test image IDs: {test_groups}\n")
        handle.write(f"Accuracy: {accuracy:.6f}\n\n")
        handle.write("Confusion matrix (rows=true, columns=predicted; C0,C1,C2,C3):\n")
        handle.write(str(matrix))
        handle.write("\n\nClassification report:\n")
        handle.write(report_text)
        handle.write("\nTop feature importances:\n")
        handle.write(importance.head(20).to_string(index=False))
        handle.write("\n")


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"Dataset not found: {CSV_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    required = set(sum(EXPERIMENTS.values(), [])) | {"file", "LABEL"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    # Example: 00001_turbo210.jpeg -> 00001.
    # This keeps all 24 JPEG versions and all 4 SSIM rows of one source image together.
    df["image_id"] = df["file"].str.rsplit("_", n=1).str[0]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, df["LABEL"], groups=df["image_id"]))
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    overlap = set(train_df["image_id"]) & set(test_df["image_id"])
    if overlap:
        raise RuntimeError(f"Data leakage detected: {len(overlap)} shared image IDs")

    print("=" * 72)
    print("PITSEC SSIM Random Forest Classification")
    print("=" * 72)
    print(f"Dataset rows: {len(df)}")
    print(f"Dataset image IDs: {df['image_id'].nunique()}")
    print(f"Training rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Training image IDs: {train_df['image_id'].nunique()}")
    print(f"Test image IDs: {test_df['image_id'].nunique()}")
    print(f"Shared image IDs: {len(overlap)}")
    print("Target labels:")
    print(train_df["LABEL"].value_counts().sort_index().to_string())

    summaries = []
    json_results = {}

    for name, features in EXPERIMENTS.items():
        print("\n" + "=" * 72)
        print(f"Experiment: {name} ({len(features)} features)")
        print("=" * 72)

        model = RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1,
        )

        start = time.time()
        model.fit(train_df[features], train_df["LABEL"])
        elapsed = time.time() - start

        predictions = model.predict(test_df[features])
        accuracy = accuracy_score(test_df["LABEL"], predictions)
        labels = ["C0", "C1", "C2", "C3"]
        matrix = confusion_matrix(test_df["LABEL"], predictions, labels=labels)
        report_text = classification_report(test_df["LABEL"], predictions, labels=labels, digits=4)
        report_dict = classification_report(
            test_df["LABEL"], predictions, labels=labels, digits=4, output_dict=True
        )

        importance = (
            pd.DataFrame({"feature": features, "importance": model.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        importance.to_csv(OUTPUT_DIR / f"{name}_feature_importance.csv", index=False)
        pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
            OUTPUT_DIR / f"{name}_confusion_matrix.csv"
        )
        write_report(
            OUTPUT_DIR / f"{name}_report.txt",
            name,
            features,
            accuracy,
            matrix,
            report_text,
            importance,
            len(train_df),
            len(test_df),
            train_df["image_id"].nunique(),
            test_df["image_id"].nunique(),
        )

        # Save the all-feature model for later inference; feature order is stored alongside it.
        if name == "all_features":
            joblib.dump(model, OUTPUT_DIR / "all_features_random_forest.joblib")
            (OUTPUT_DIR / "all_features_model_features.json").write_text(
                json.dumps(features, indent=2), encoding="utf-8"
            )

        print(f"Training time: {elapsed:.1f} seconds")
        print(f"Test accuracy: {accuracy:.4%}")
        print("Confusion matrix (rows=true, columns=predicted; C0,C1,C2,C3):")
        print(matrix)
        print("Top 10 feature importances:")
        print(importance.head(10).to_string(index=False))

        summaries.append({
            "experiment": name,
            "n_features": len(features),
            "test_accuracy": accuracy,
            "training_seconds": elapsed,
        })
        json_results[name] = {
            "n_features": len(features),
            "test_accuracy": accuracy,
            "training_seconds": elapsed,
            "classification_report": report_dict,
        }

    summary = pd.DataFrame(summaries).sort_values("test_accuracy", ascending=False)
    summary.to_csv(OUTPUT_DIR / "experiment_summary.csv", index=False)
    (OUTPUT_DIR / "experiment_results.json").write_text(
        json.dumps(json_results, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    print(summary.to_string(index=False))
    print(f"\nSaved reports, confusion matrices, feature importances, and model to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
