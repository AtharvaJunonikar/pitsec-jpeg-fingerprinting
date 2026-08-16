# Experiment 1 — SSIM-Based JPEG Library Fingerprinting

## Purpose

This experiment extends the original SSIM-based JPEG fingerprinting workflow with three additional feature groups:

- **SSIM:** recompression-difference features.
- **DCT:** frequency-domain statistics calculated from 8×8 image blocks.
- **YCbCr:** luminance and chrominance distribution statistics.
- **Chroma:** local 8×8 chroma block-variance statistics.

The objective is to classify four SSIM reference encoder clusters:

| SSIM source encoder | Label |
|---|---|
| `6b` | `C0` |
| `7` | `C1` |
| `9e` | `C2` |
| `mozjpeg300` | `C3` |

The project uses an explainable non-deep-learning classifier: a Random Forest.

---

## Workflow overview

```text
Original images
      │
      ├── Compress with 24 JPEG-library versions
      │       │
      │       └── 60,000 JPEG files from 2,500 original images
      │
      ├── Create four SSIM reference-source cases per JPEG file
      │       │
      │       └── Labels C0, C1, C2, C3
      │
      ├── Extract features
      │       ├── 8 SSIM features
      │       ├── 15 DCT features
      │       ├── 8 YCbCr features
      │       └── 4 chroma features
      │
      ├── Create combined dataset
      │       │
      │       └── 240,000 rows and 35 numeric feature columns
      │
      ├── Validate dataset structure and feature integrity
      │
      ├── Train and evaluate Random Forest models
      │       ├── Feature-ablation experiments
      │       └── 5-fold group cross-validation
      │
      └── Evaluate frozen model on unseen public external images
```

---

## Dataset construction

### Input structure

- **Original images:** 2,500
- **JPEG-library versions per original image:** 24
- **Compressed JPEG files:** 60,000
- **SSIM rows per JPEG file:** 4
- **Final combined rows:** 240,000

The row count is:

\[
2{,}500 \times 24 \times 4 = 240{,}000
\]

Each JPEG file generates four rows because the original SSIM workflow re-encodes the image using four reference encoders (`6b`, `7`, `9e`, `mozjpeg300`). The row label is the cluster associated with that SSIM source encoder.

### Final dataset columns

The generated CSV has 38 saved columns:

- 35 numeric features.
- `file`: JPEG filename.
- `version`: SSIM source encoder version.
- `LABEL`: target class (`C0`, `C1`, `C2`, or `C3`).

The validation script adds temporary `image_id` and `jpeg_version` columns in memory, so it may report 40 columns. Those two helper columns are not saved in the original CSV.

---

## Feature groups

### SSIM — 8 features

```text
diff_C0, diff_C1, diff_C2, diff_C3,
norm_C0, norm_C1, norm_C2, norm_C3
```

These measure the structural-similarity difference after reference recompression and are the baseline fingerprint features.

### DCT — 15 features

For Y, Cr, and Cb channels, the pipeline extracts:

```text
ac_energy, dc_variance, ac_variance, zero_ratio, energy_conc
```

DCT features capture image-frequency and compression-related behavior in 8×8 blocks.

### YCbCr — 8 features

```text
Y_mean, Y_std, Y_var, Y_entropy,
Cb_mean, Cb_std, Cr_mean, Cr_std
```

These describe luminance and chrominance distributions.

### Chroma — 4 features

```text
chroma_Cb_blockvar_mean, chroma_Cb_blockvar_std,
chroma_Cr_blockvar_mean, chroma_Cr_blockvar_std
```

These measure local chroma variability across 8×8 blocks.

---

## Scripts

### Dataset generation and validation

| Script | What it does |
|---|---|
| `build_final_dataset.py` | Generates the 35-feature combined dataset using CPU multiprocessing. |
| `verify_dataset.py` | Prints row count, label distribution, version distribution, and rows per file. |
| `validate_dataset.py` | Verifies feature integrity, no missing/infinite values, 24 JPEG versions per original image, 4 rows per JPEG file, and non-SSIM consistency across SSIM rows. |

### Model training and evaluation

| Script | What it does |
|---|---|
| `train_ssim_random_forest.py` | Trains Random Forest ablation models: SSIM only, SSIM+DCT, SSIM+YCbCr+chroma, and all features. |
| `cross_validate_ssim_random_forest.py` | Runs leakage-safe 5-fold group cross-validation for the all-feature Random Forest. |

### External validation

| Script | What it does |
|---|---|
| `select_natural_images_pilot.py` | Selects a deterministic 20-image public-image pilot subset and creates a manifest. |
| `run_external_validation.py` | Recompresses external images with the four reference encoders, extracts the same 35 features, and evaluates the frozen Random Forest without retraining. |

---

## How to run

Run all commands from the project root.

### Build the full feature dataset

```bash
python src/experiment_1/build_final_dataset.py --workers 4
```

### Verify the basic dataset structure

```bash
python src/experiment_1/verify_dataset.py
```

### Run complete data validation

```bash
python src/experiment_1/validate_dataset.py
```

### Train Random Forest experiments

```bash
python src/experiment_1/train_ssim_random_forest.py
```

### Run 5-fold group cross-validation

```bash
python src/experiment_1/cross_validate_ssim_random_forest.py
```

### Select the 20-image external pilot

```bash
python src/experiment_1/select_natural_images_pilot.py
```

### Run frozen-model external validation

```bash
python src/experiment_1/run_external_validation.py
```

> If the external-validation scripts remain in `src/external_validation_scripts/` rather than `src/experiment_1/`, run them from that original path instead.

---

## Leakage-safe evaluation

A random row-level split would be invalid because related samples from the same original image could appear in both train and test sets.

The project therefore uses `image_id` as a group identifier. For example, all rows related to original image `00001` stay together:

```text
00001_6b.jpeg
00001_7.jpeg
...
00001_turbo210.jpeg
```

Each original image contributes:

\[
24 \text{ JPEG versions} \times 4 \text{ SSIM rows} = 96 \text{ rows}
\]

`GroupShuffleSplit` and `GroupKFold` ensure that no `image_id` occurs in both training and test data.

---

## Output locations

Generated files are intentionally stored outside this source folder:

| Directory | Contents |
|---|---|
| `output/` | Combined dataset, validation outputs, model artifacts, and reports. |
| `output/random_forest_results/` | Ablation-study summaries, feature importance, confusion matrices, reports, and saved model. |
| `output/random_forest_cv_results/` | Cross-validation fold results, combined confusion matrix, and feature-importance summary. |
| `output/external_validation/` | External pilot features, predictions, manifests, confusion matrix, and report. |
| `data/external_originals/` | Selected external pilot images and selection manifest. |
| `data/external_compressed/` | Reference-compressed variants of external pilot images. |

---

## Reproducibility notes

- Random seed: `42`.
- Random Forest: 300 trees, `max_features="sqrt"`, `min_samples_leaf=2`, and all available CPU cores.
- The external model is loaded from the internal all-feature Random Forest and is **not retrained** on external images.
- Preserve the generated output folders locally; they are excluded from Git because the main dataset and model files are large.

---

## Interpretation

The experiment is designed to test whether additional DCT, YCbCr, and chroma features improve the original SSIM-based fingerprinting approach. See `RESULTS.md` for measured internal and external validation results.
