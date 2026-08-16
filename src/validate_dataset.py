import numpy as np
import pandas as pd
from pathlib import Path

CSV_PATH = Path("output/all_features_combined.csv")

FEATURE_COLUMNS = [
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

NON_SSIM_COLUMNS = FEATURE_COLUMNS[8:]

df = pd.read_csv(CSV_PATH)

df["image_id"] = df["file"].str.rsplit("_", n=1).str[0]
df["jpeg_version"] = (
    df["file"]
    .str.rsplit("_", n=1)
    .str[1]
    .str.replace(".jpeg", "", regex=False)
)

print("=" * 60)
print("DATASET VALIDATION")
print("=" * 60)

print("\nRows:", len(df))
print("Columns:", len(df.columns))
print("Unique original image IDs:", df["image_id"].nunique())
print("Unique JPEG files:", df["file"].nunique())

print("\nMissing feature values:", int(df[FEATURE_COLUMNS].isna().sum().sum()))
print("Infinite feature values:", int(np.isinf(df[FEATURE_COLUMNS].to_numpy()).sum()))

print("\nRows per JPEG file:")
print(df.groupby("file").size().value_counts().sort_index())

print("\nJPEG versions per original image:")
print(df.groupby("image_id")["jpeg_version"].nunique().value_counts().sort_index())

print("\nRows per original image:")
print(df.groupby("image_id").size().value_counts().sort_index())

print("\nClass labels:")
print(df["LABEL"].value_counts().sort_index())

print("\nSSIM source versions:")
print(df["version"].value_counts().sort_index())

within_file_variation = df.groupby("file")[NON_SSIM_COLUMNS].nunique()
inconsistent = within_file_variation[within_file_variation.gt(1).any(axis=1)]

print("\nFiles with inconsistent DCT/YCbCr/chroma values across 4 SSIM rows:")
print(len(inconsistent))

print("\nFeature ranges:")
print(df[FEATURE_COLUMNS].describe().T[["min", "max", "mean", "std"]])

if len(inconsistent) == 0:
    print("\nPASS: DCT, YCbCr, and chroma features are identical across the four SSIM rows of every JPEG file.")
else:
    print("\nWARNING: Some files have inconsistent repeated non-SSIM features.")

print("=" * 60)