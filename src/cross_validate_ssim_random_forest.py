#!/usr/bin/env python3
"""5-fold group cross-validation for the SSIM-based PITSEC Random Forest.

All rows from one original image ID remain in exactly one fold.
This evaluates the all-features model without image-level leakage.

Run from the project root:
    python src/cross_validate_ssim_random_forest.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "output" / "all_features_combined.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "random_forest_cv_results"
RANDOM_STATE = 42
LABELS = ["C0", "C1", "C2", "C3"]

FEATURES = [
    "diff_C0", "diff_C1", "diff_C2", "diff_C3",
    "norm_C0", "norm_C1", "norm_C2", "norm_C3",
    "ac_energy_y", "dc_variance_y", "ac_variance_y", "zero_ratio_y", "energy_conc_y",
    "ac_energy_cr", "dc_variance_cr", "ac_variance_cr", "zero_ratio_cr", "energy_conc_cr",
    "ac_energy_cb", "dc_variance_cb", "ac_variance_cb", "zero_ratio_cb", "energy_conc_cb",
    "Y_mean", "Y_std", "Y_var", "Y_entropy",
    "Cb_mean", "Cb_std", "Cr_mean", "Cr_std",
    "chroma_Cb_blockvar_mean", "chroma_Cb_blockvar_std",
    "chroma_Cr_blockvar_mean", "chroma_Cr_blockvar_std",
]


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"Dataset not found: {CSV_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    df["image_id"] = df["file"].str.rsplit("_", n=1).str[0]

    missing = sorted(set(FEATURES + ["LABEL"]) - set(df.columns))
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    groups = df["image_id"]
    cv = GroupKFold(n_splits=5)
    fold_results = []
    combined_matrix = pd.DataFrame(0, index=LABELS, columns=LABELS)
    all_importances = []

    print("=" * 72)
    print("PITSEC Random Forest: 5-Fold Group Cross-Validation")
    print("=" * 72)
    print(f"Rows: {len(df)}")
    print(f"Original image IDs: {groups.nunique()}")
    print(f"Features: {len(FEATURES)}")
    print("Folds: 5")

    for fold, (train_idx, test_idx) in enumerate(cv.split(df, df["LABEL"], groups), start=1):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        overlap = set(train_df["image_id"]) & set(test_df["image_id"])
        if overlap:
            raise RuntimeError(f"Fold {fold}: found {len(overlap)} shared image IDs")

        print("\n" + "-" * 72)
        print(f"Fold {fold}/5")
        print(f"Train image IDs: {train_df['image_id'].nunique()}")
        print(f"Test image IDs: {test_df['image_id'].nunique()}")
        print(f"Shared image IDs: {len(overlap)}")

        model = RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE + fold,
            verbose=0,
        )

        start = time.time()
        model.fit(train_df[FEATURES], train_df["LABEL"])
        elapsed = time.time() - start

        predictions = model.predict(test_df[FEATURES])
        accuracy = accuracy_score(test_df["LABEL"], predictions)
        matrix = confusion_matrix(test_df["LABEL"], predictions, labels=LABELS)
        combined_matrix += pd.DataFrame(matrix, index=LABELS, columns=LABELS)

        importances = pd.DataFrame({
            "fold": fold,
            "feature": FEATURES,
            "importance": model.feature_importances_,
        })
        all_importances.append(importances)

        fold_results.append({
            "fold": fold,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_image_ids": train_df["image_id"].nunique(),
            "test_image_ids": test_df["image_id"].nunique(),
            "shared_image_ids": len(overlap),
            "accuracy": accuracy,
            "training_seconds": elapsed,
        })

        print(f"Accuracy: {accuracy:.4%}")
        print(f"Training time: {elapsed:.1f} seconds")

    fold_df = pd.DataFrame(fold_results)
    importances_df = pd.concat(all_importances, ignore_index=True)
    importance_summary = (
        importances_df.groupby("feature", as_index=False)["importance"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    class_report = classification_report(
        [], [], labels=LABELS, zero_division=0, output_dict=True
    )
    del class_report

    normalized_matrix = combined_matrix.div(combined_matrix.sum(axis=1), axis=0)

    fold_df.to_csv(OUTPUT_DIR / "fold_results.csv", index=False)
    combined_matrix.to_csv(OUTPUT_DIR / "combined_confusion_matrix_counts.csv")
    normalized_matrix.to_csv(OUTPUT_DIR / "combined_confusion_matrix_normalized.csv")
    importances_df.to_csv(OUTPUT_DIR / "feature_importance_by_fold.csv", index=False)
    importance_summary.to_csv(OUTPUT_DIR / "feature_importance_summary.csv", index=False)

    summary = {
        "features": len(FEATURES),
        "folds": 5,
        "mean_accuracy": float(fold_df["accuracy"].mean()),
        "std_accuracy": float(fold_df["accuracy"].std(ddof=1)),
        "min_accuracy": float(fold_df["accuracy"].min()),
        "max_accuracy": float(fold_df["accuracy"].max()),
        "mean_training_seconds": float(fold_df["training_seconds"].mean()),
        "shared_image_ids_total": int(fold_df["shared_image_ids"].sum()),
    }
    (OUTPUT_DIR / "cross_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 72)
    print(fold_df[["fold", "accuracy", "training_seconds", "shared_image_ids"]].to_string(index=False))
    print(f"\nMean accuracy: {summary['mean_accuracy']:.4%}")
    print(f"Standard deviation: {summary['std_accuracy']:.4%}")
    print(f"Accuracy range: {summary['min_accuracy']:.4%} to {summary['max_accuracy']:.4%}")
    print(f"Total shared image IDs across folds: {summary['shared_image_ids_total']}")
    print("\nCombined confusion matrix (rows=true, columns=predicted):")
    print(combined_matrix)
    print("\nTop 15 mean feature importances:")
    print(importance_summary.head(15).to_string(index=False))
    print(f"\nSaved cross-validation results to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
