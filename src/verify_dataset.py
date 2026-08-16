#!/usr/bin/env python3
"""Quick verification script for the combined feature dataset."""

import pandas as pd
from pathlib import Path

output_dir = Path(__file__).resolve().parent.parent / "output"
csv_path = output_dir / "all_features_combined.csv"

if not csv_path.exists():
    raise SystemExit(f"CSV not found: {csv_path}")

df = pd.read_csv(csv_path)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nRows per file:")
print(df.groupby("file").size())

print("\nLabels:")
print(df["LABEL"].value_counts())

print("\nVersions:")
print(df["version"].value_counts())