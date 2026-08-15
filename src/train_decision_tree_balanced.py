#!/usr/bin/env python3
"""
Train a decision tree on the combined JPEG fingerprinting features.

This script is designed for the project CSV created by:
    src/build_all_feature_dataset_parallel.py

It follows the correct ML practice for this dataset:
- split by source image ID, not by row
- keep all versions for each source image together
- optionally balance the training set while keeping the test set realistic
- evaluate cluster classification (C0-C3)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


FEATURE_COLUMNS = [
    "ac_energy_y",
    "dc_variance_y",
    "ac_variance_y",
    "zero_ratio_y",
    "energy_conc_y",
    "ac_energy_cr",
    "dc_variance_cr",
    "ac_variance_cr",
    "zero_ratio_cr",
    "energy_conc_cr",
    "ac_energy_cb",
    "dc_variance_cb",
    "ac_variance_cb",
    "zero_ratio_cb",
    "energy_conc_cb",
    "ac_energy_avg",
    "dc_variance_avg",
    "zero_ratio_avg",
    "chroma_Cb_mean",
    "chroma_Cb_std",
    "chroma_Cb_var",
    "chroma_Cr_mean",
    "chroma_Cr_std",
    "chroma_Cr_var",
    "chroma_Cb_edge_mean",
    "chroma_Cb_edge_std",
    "chroma_Cr_edge_mean",
    "chroma_Cr_edge_std",
    "chroma_Cb_blockvar_mean",
    "chroma_Cb_blockvar_std",
    "chroma_Cb_blockvar_max",
    "chroma_Cr_blockvar_mean",
    "chroma_Cr_blockvar_std",
    "chroma_Cr_blockvar_max",
    "luma_Y_mean",
    "luma_Y_std",
    "Y_mean",
    "Y_std",
    "Y_var",
    "Y_entropy",
    "Cb_mean",
    "Cb_std",
    "Cb_var",
    "Cb_entropy",
    "Cr_mean",
    "Cr_std",
    "Cr_var",
    "Cr_entropy",
    "Y_minus_Cb_mean",
    "Y_minus_Cr_mean",
    "diff_C0",
    "diff_C1",
    "diff_C2",
    "diff_C3",
    "norm_C0",
    "norm_C1",
    "norm_C2",
    "norm_C3",
]


def split_by_source_image(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split by source image ID so all compressed versions of the same image stay together."""
    source_ids = df["source_id"].drop_duplicates().tolist()
    train_ids, test_ids = train_test_split(
        source_ids,
        test_size=test_size,
        random_state=random_state,
    )

    train_mask = df["source_id"].isin(train_ids)
    test_mask = df["source_id"].isin(test_ids)

    return df.loc[train_mask].copy(), df.loc[test_mask].copy()


def balance_training_set(train_df: pd.DataFrame, random_state: int = 42):
    """Downsample majority classes to the size of the minority class.

    Important: this is done only on the training split, not on the full dataset.
    """
    min_count = train_df["LABEL"].value_counts().min()

    balanced_frames = []
    for label, group in train_df.groupby("LABEL", sort=True):
        n = min(len(group), min_count)
        sampled = group.sample(n=n, random_state=random_state)
        balanced_frames.append(sampled)

    return pd.concat(balanced_frames, ignore_index=True)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """Print standard evaluation metrics."""
    pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, pred))
    print("Macro F1:", f1_score(y_test, pred, average="macro"))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred, labels=["C0", "C1", "C2", "C3"]))
    print("\nClassification Report:\n")
    print(classification_report(y_test, pred, labels=["C0", "C1", "C2", "C3"], zero_division=0))


def main():
    parser = argparse.ArgumentParser(description="Train a decision tree on the JPEG fingerprinting dataset.")
    parser.add_argument("--csv", type=str, default="output/all_features.csv", help="Path to the combined feature CSV")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of source images kept for test")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--balance-train", action="store_true", help="Downsample the training set to balance class counts")
    parser.add_argument("--class-weight-balanced", action="store_true", help="Use class_weight='balanced' in the decision tree")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "LABEL" not in df.columns:
        raise ValueError("The CSV must contain a LABEL column.")

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing[:10]}")

    train_df, test_df = split_by_source_image(df, test_size=args.test_size, random_state=args.random_state)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["LABEL"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["LABEL"]

    if args.balance_train:
        train_df = balance_training_set(train_df, random_state=args.random_state)
        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df["LABEL"]

    clf_kwargs = {"random_state": args.random_state}
    if args.class_weight_balanced:
        clf_kwargs["class_weight"] = "balanced"

    model = DecisionTreeClassifier(**clf_kwargs)
    model.fit(X_train, y_train)

    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))
    print("Train label distribution:\n", y_train.value_counts().to_dict())
    print("Test label distribution:\n", y_test.value_counts().to_dict())
    print("\n=== Decision Tree Evaluation ===")
    evaluate_model(model, X_test, y_test)

    # Also print feature importance for interpretability.
    print("\nTop 10 important features:\n")
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    print(importances.sort_values(ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
