# Experiment 1 — Results and Interpretation

## Executive summary

Experiment 1 created a combined SSIM, DCT, YCbCr, and chroma feature dataset for SSIM-based JPEG-library fingerprint classification.

The final Random Forest model achieved:

- **Internal 5-fold group cross-validation:** **82.41% ± 0.45%** mean accuracy.
- **Preliminary external-content pilot:** **78.75%** accuracy on 20 unseen public images.

The results show that SSIM is the primary source of predictive information, while DCT, YCbCr, and chroma features improve performance beyond the original SSIM-only baseline.

---

## 1. Dataset validation

### Expected dataset design

| Item | Expected value | Observed value | Status |
|---|---:|---:|---|
| Original image IDs | 2,500 | 2,500 | Pass |
| JPEG versions per original image | 24 | 24 for every image ID | Pass |
| JPEG files | 60,000 | 60,000 | Pass |
| SSIM rows per JPEG file | 4 | 4 for every JPEG file | Pass |
| Rows per original image | 96 | 96 for every image ID | Pass |
| Total feature rows | 240,000 | 240,000 | Pass |
| Numeric feature columns | 35 | 35 | Pass |
| Saved CSV columns | 38 | 38 | Pass |
| Missing feature values | 0 | 0 | Pass |
| Infinite feature values | 0 | 0 | Pass |
| Non-SSIM consistency across 4 SSIM rows | 0 inconsistent files | 0 inconsistent files | Pass |

The total number of rows is correct:

\[
2{,}500 \text{ original images} \times 24 \text{ JPEG versions} \times 4 \text{ SSIM cases} = 240{,}000 \text{ rows}
\]

### What the consistency result means

For each JPEG file, DCT, YCbCr, and chroma features are calculated once and repeated across its four SSIM reference-label rows. The validation result:

```text
Files with inconsistent DCT/YCbCr/chroma values across 4 SSIM rows: 0
```

means that all repeated non-SSIM feature values are correctly aligned with their corresponding JPEG file.

### Feature-range checks

The validation found feature values in plausible ranges:

- `norm_C0` through `norm_C3`: values from 0 to 1.
- `zero_ratio_*`: values from 0 to 1.
- `energy_conc_*`: values from 0 to 1.
- Y/Cb/Cr means: within 8-bit image intensity ranges.
- No missing or infinite values occurred.

This does not prove that every feature is optimal, but it confirms that the generated numeric data is structurally valid and usable for model training.

---

## 2. Random Forest setup

### Target classes

| SSIM source encoder | Target label |
|---|---|
| `6b` | C0 |
| `7` | C1 |
| `9e` | C2 |
| `mozjpeg300` | C3 |

### Model configuration

| Parameter | Value |
|---|---|
| Classifier | Random Forest |
| Number of trees | 300 |
| Feature sampling | Square root of feature count per split |
| Minimum samples per leaf | 2 |
| Class balancing | `balanced_subsample` |
| Random seed | 42 |
| Parallelism | All available CPU cores |

### Leakage prevention

The evaluation was performed by grouping all rows from the same original image ID together. Therefore, the train and test sets never contain related rows from the same original image.

For the 80/20 split:

```text
Training image IDs: 2000
Test image IDs: 500
Shared image IDs: 0
```

This is important because a row-level random split would allow highly related JPEG versions and SSIM rows to occur in both training and test data, making the reported accuracy unrealistically high.

---

## 3. Feature-ablation results

| Experiment | Feature count | Test accuracy | Interpretation |
|---|---:|---:|---|
| SSIM only | 8 | 77.66% | Original SSIM-based baseline |
| SSIM + YCbCr + chroma | 20 | 81.29% | Color-domain statistics add useful signal |
| SSIM + DCT | 23 | 82.17% | DCT provides the largest incremental improvement |
| All features | 35 | **82.46%** | Best single-split result |

### What this means

The full model improves over the SSIM-only baseline by:

\[
82.46\% - 77.66\% = 4.80 \text{ percentage points}
\]

This supports the project goal of extending the original SSIM-based method with additional JPEG-related feature modalities.

The gain from adding DCT features is larger than the gain from adding YCbCr and chroma features alone. The all-feature model provides the best result, although its improvement over SSIM+DCT is modest:

\[
82.46\% - 82.17\% = 0.29 \text{ percentage points}
\]

---

## 4. Internal cross-validation

### 5-fold group cross-validation

Each fold trained on 2,000 original image IDs and tested on 500 previously unseen original image IDs. Every fold reported zero shared image IDs between train and test data.

| Fold | Accuracy | Shared image IDs |
|---:|---:|---:|
| 1 | 82.53% | 0 |
| 2 | 81.70% | 0 |
| 3 | 82.27% | 0 |
| 4 | 82.70% | 0 |
| 5 | 82.85% | 0 |

### Cross-validation summary

| Metric | Result |
|---|---:|
| Mean accuracy | **82.41%** |
| Standard deviation | **0.45%** |
| Minimum accuracy | 81.70% |
| Maximum accuracy | 82.85% |
| Shared image IDs across all folds | 0 |

### What this means

The standard deviation of 0.45 percentage points is small. The model’s performance is therefore stable across five different groups of unseen original images.

The mean cross-validation accuracy is very close to the 82.46% accuracy from the earlier all-feature single split. This agreement indicates that the single-split result was not an accidental favorable partition.

---

## 5. Internal class behavior

The combined cross-validation confusion matrix was:

```text
       predicted
true      C0     C1     C2     C3
C0     39872  15564      2   4562
C1      6759  49427     22   3792
C2         0      2  59968     30
C3      5495   5986      0  48519
```

### Interpretation

- **C2** is extremely separable: 59,968 of 60,000 C2 samples were classified correctly.
- **C0, C1, and C3** have more overlap.
- The largest internal confusion is **C0 predicted as C1**.
- C3 is confused with both C0 and C1.

This should be reported as an empirical property of the chosen SSIM reference clusters and feature set. It is not evidence of a dataset-structure error because class balance, source grouping, and feature consistency were all validated.

---

## 6. Feature importance

The highest mean Random Forest feature importances in group cross-validation were:

| Rank | Feature | Mean importance |
|---:|---|---:|
| 1 | `norm_C0` | 0.1241 |
| 2 | `norm_C1` | 0.1118 |
| 3 | `norm_C2` | 0.1013 |
| 4 | `diff_C3` | 0.1002 |
| 5 | `norm_C3` | 0.0976 |
| 6 | `diff_C1` | 0.0736 |
| 7 | `diff_C2` | 0.0687 |
| 8 | `diff_C0` | 0.0676 |
| 9 | `ac_variance_y` | 0.0182 |
| 10 | `ac_energy_y` | 0.0174 |

Additional DCT and chroma features among the higher-ranked non-SSIM features include:

```text
chroma_Cr_blockvar_std
zero_ratio_y
chroma_Cb_blockvar_std
ac_energy_cr
ac_variance_cr
```

### What this means

SSIM features provide most of the classification signal. DCT features provide the strongest complementary contribution, especially luminance-channel AC variance and AC energy. Chroma features contribute smaller but non-zero additional information.

---

## 7. Preliminary external validation

### Protocol

A public external-content pilot was performed using 20 images selected from the Kaggle Natural Images dataset.

- External original images: 20.
- Training data used: none of these images.
- Reference re-encodings per external image: 4.
- External rows: 80.
- Model: frozen all-feature Random Forest trained on internal data.
- Retraining on external images: no.

The reference encoding and labels were:

| Encoder | Label |
|---|---|
| `6b` | C0 |
| `7` | C1 |
| `9e` | C2 |
| `mozjpeg300` | C3 |

### External results

| Metric | Result |
|---|---:|
| External validation rows | 80 |
| Overall accuracy | **78.75%** |
| Macro precision | 80.77% |
| Macro recall | 78.75% |
| Macro F1 score | 78.20% |

External confusion matrix:

```text
       predicted
true    C0  C1  C2  C3
C0      17   2   0   1
C1      10   9   0   1
C2       0   0  20   0
C3       2   1   0  17
```

Per-class recall:

| Class | Recall | Meaning |
|---|---:|---|
| C0 | 85.0% | Strong external detection |
| C1 | 45.0% | Main external weakness; often predicted as C0 |
| C2 | 100.0% | Perfect on this 20-image pilot |
| C3 | 85.0% | Strong external detection |

### Comparison with internal cross-validation

| Evaluation | Accuracy |
|---|---:|
| Internal 5-fold group cross-validation | 82.41% ± 0.45% |
| Preliminary external-content pilot | 78.75% |

The external accuracy is lower by:

\[
82.41\% - 78.75\% = 3.66 \text{ percentage points}
\]

### What this means

The model retained useful performance on a separate public image source without being retrained, but performance dropped relative to the internal image-group cross-validation result.

This is expected in forensic and JPEG-fingerprinting settings because image content, camera pipeline, image size, prior compression, and color processing can change the distribution of features. The pilot particularly confirms that C1 and C0 are the most difficult clusters to separate under content-source shift.

### Important limitation

This external test is a **preliminary external-content robustness pilot**, not final camera-native forensic validation:

- It uses only 20 external original images.
- The source images may have unknown prior JPEG processing.
- The project PPT specifies that final validation data should be selected together with the supervisor.

For final work, validate on at least 100 camera-native images from a supervisor-approved source such as a camera-forensics dataset or newly captured images from distinct devices.

---

## 8. Final conclusion

Experiment 1 is complete for internal development and preliminary external evaluation.

The final outcome is:

> A leakage-safe, SSIM-based Random Forest JPEG fingerprint classifier using 35 combined SSIM, DCT, YCbCr, and chroma features. The model achieved **82.41% ± 0.45%** in 5-fold group cross-validation and **78.75%** on a preliminary public external-content pilot. Additional features improved accuracy by **4.80 percentage points** compared with the SSIM-only baseline.

The immediate next steps for future work are:

1. Use a larger supervisor-approved camera-native external validation set.
2. Evaluate at least 100 external originals.
3. Analyze C0/C1/C3 confusion in relation to JPEG parameters, chroma subsampling, and library-version behavior.
4. Include the workflow, results, limitations, literature references, and parameter settings in the scientific report.
