# Chroma Wrinkle Feature Extraction (`chroma_features.py`)

## 1. High‑level idea

JPEG fingerprinting relies on tiny, systematic artifacts left by the encoder. Besides double‑compression patterns (handled in `bulk_classify.py`), **chroma channels (Cb, Cr)** also contain useful "wrinkliness" patterns that depend on:

- how chroma is sub‑sampled
- how chroma DCT coefficients are quantized
- how up/down‑sampling is implemented in different libjpeg versions

The `chroma_features.py` module extracts **hand‑crafted chroma descriptors** from each JPEG image. These descriptors summarize:

- overall chroma intensity statistics
- local chroma edge strength
- block‑wise chroma variance ("wrinkles")
- a small amount of luma context

We use them as an additional feature family, **complementary** to the SSIM‑based double‑compression matrix features from `bulk_classify.py`.

Each image is represented by **one feature row** with a **cluster label C0–C3**.

---

## 2. Pipeline summary

For each JPEG in `data/compressed/`:

1. Load the image as BGR (`cv2.imread`).
2. Convert to **YCrCb** color space (`cv2.COLOR_BGR2YCrCb`).
3. Extract channels:
   - `Y`  – luma (brightness)
   - `Cr` – red‑difference chroma
   - `Cb` – blue‑difference chroma
4. Compute for Cb/Cr/Y:
   - global intensity statistics
   - edge energy statistics
   - 8×8 block‑wise variance statistics
5. Infer the image’s **cluster label** (`C0`, `C1`, `C2`, `C3`) from its filename using the same encoder→cluster mapping as `bulk_classify.py`.
6. Write one row to `test_output/chroma_features_sample.csv` containing:
   - all numeric features
   - `file`
   - `LABEL`

No model is trained in this script; it only builds a **structured dataset** that other scripts can use.

---

## 3. Detailed feature definitions

Let `Cb` and `Cr` be the chroma channels as 2D float arrays, and `Y` the luma channel.

### 3.1 Global chroma statistics

For each of `Cb` and `Cr` we compute:

- `chroma_Cb_mean`, `chroma_Cr_mean`  
  `chroma_Cb_mean = mean(Cb)`  
  `chroma_Cr_mean = mean(Cr)`

- `chroma_Cb_std`, `chroma_Cr_std`  
  `chroma_Cb_std = std(Cb)`  
  `chroma_Cr_std = std(Cr)`

- `chroma_Cb_var`, `chroma_Cr_var`  
  `chroma_Cb_var = var(Cb)`  
  `chroma_Cr_var = var(Cr)`

These summarize the **overall chroma intensity distribution**. Different libjpeg versions (e.g., with different chroma quantization tables) can produce systematically different chroma contrasts and spreads.

### 3.2 Chroma edge energy (Sobel gradients)

For each of `Cb` and `Cr`:

1. Compute horizontal and vertical gradients using Sobel filters:

   ```python
   gx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
   gy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
   mag = np.sqrt(gx * gx + gy * gy)
   ```

2. From the gradient magnitude `mag` compute:

   - `chroma_Cb_edge_mean`, `chroma_Cr_edge_mean`:  
     `chroma_Cb_edge_mean = mean(mag_Cb)`  
     `chroma_Cr_edge_mean = mean(mag_Cr)`
   - `chroma_Cb_edge_std`, `chroma_Cr_edge_std`:  
     `chroma_Cb_edge_std = std(mag_Cb)`  
     `chroma_Cr_edge_std = std(mag_Cr)`

These measure how **textured** the chroma channels are in terms of edges. Changes in sub‑sampling and reconstruction filters influence chroma edges differently from luma edges, and those differences can be characteristic of encoder families.

### 3.3 Block‑wise chroma variance ("wrinkles")

To approximate **chroma wrinkles** (local irregularities introduced by chroma sub‑sampling and quantization), we examine local variance over 8×8 blocks.

For each of `Cb` and `Cr`:

1. Crop the channel to a multiple of 8 in width/height.
2. Reshape it into non‑overlapping 8×8 blocks.
3. For each block, compute its variance.
4. Collect all block variances into a flat array `vars_flat`.
5. From `vars_flat` compute:

   - `chroma_Cb_blockvar_mean`, `chroma_Cr_blockvar_mean`  
     `= mean(block_variances)`
   - `chroma_Cb_blockvar_std`, `chroma_Cr_blockvar_std`  
     `= std(block_variances)`
   - `chroma_Cb_blockvar_max`, `chroma_Cr_blockvar_max`  
     `= max(block_variances)`

These features capture both the **average level** of chroma "wrinkliness" and its **spread and extremes**. Empirically, different libjpeg families (e.g., with new chroma quantization or progressive coding) can produce distinct patterns in chroma block variance.

### 3.4 Basic luma statistics

For luma:

- `luma_Y_mean = mean(Y)`
- `luma_Y_std  = std(Y)`

We include these as simple contextual features. They allow downstream analysis to check whether chroma patterns differ from luma behavior (for example, a version mainly affecting chroma but not luma).

---

## 4. Label assignment: mapping to C0–C3

The label in `chroma_features_sample.csv` is a **cluster label** from the four libjpeg families defined in the project:

- **C0:** legacy upsampling family (e.g., `6b`, `turbo120`, `turbo121`, `mozjpeg201`, ...)
- **C1:** DCT‑scaling family (e.g., `7`, `8`, `8a`, `9`, `9a`, `9b`, `9c`, `9d`)
- **C2:** new chrominance quantization (e.g., `9e`, `9f`)
- **C3:** mozjpeg progressive family (e.g., `mozjpeg300`, `mozjpeg403`, `mozjpeg101`)

In `chroma_features.py` we keep the same lists (copied from `bulk_classify.py`), and infer the cluster from the filename:

```python
def infer_cluster_from_filename(filename: str) -> str:
    name = filename.lower()

    for tag in C0_ENCODERS:
        if tag in name:
            return "C0"
    for tag in C1_ENCODERS:
        if tag in name:
            return "C1"
    for tag in C2_ENCODERS:
        if tag in name:
            return "C2"
    for tag in C3_ENCODERS:
        if tag in name:
            return "C3"

    raise ValueError(f"Could not infer cluster from filename: {filename}")
```

Example:

- `02020_mozjpeg300.jpeg` → contains `"mozjpeg300"` → in `C3_ENCODERS` → `LABEL = C3`.
- `01644_8d.jpeg`        → contains `"8d"`         → in `C1_ENCODERS` → `LABEL = C1`.

We verified label consistency by comparing this `LABEL` with the labels used in `output_addLayer.csv` from `bulk_classify.py`.

---

## 5. What this achieves / why it is useful

### 5.1 Complementary signal to SSIM matrix

The double‑compression SSIM matrix focuses on **how images respond to recompression** under different encoder assumptions. It is very powerful, but it mainly captures how luma and chroma behave together when you recompress.

Chroma features add **direct, per‑image descriptors** of the chroma channels themselves, without re‑running a full double‑compression matrix:

- They can highlight families where **chroma behavior** changes more strongly than luma (for example, new chroma quantization, different chroma upsampling).
- They provide **interpretability**: we can say things like, "C3 images tend to have higher chroma block variance and edge energy than C1 images at similar luma conditions."

### 5.2 Lightweight and reusable

Compared to computing a full 4×4 recompression matrix for every feature set:

- Chroma features require only **one pass per image**, no nested loops over encoder versions.
- They are cheap to compute and can be applied to new images without recompression if needed (in this project we use the already‑compressed JPEGs).

This makes them a good candidate for:

- fast exploratory analysis
- feature importance inspection (e.g., via decision trees or random forests)
- combining with heavier SSIM‑based features for improved accuracy

### 5.3 Explanation to the professor / reviewers

When presenting this work, we can justify chroma features as follows:

- JPEG artifacts are not only in luma; **chroma channels also carry encoder‑specific traces** due to sub‑sampling and quantization strategies.
- Our features explicitly target:
  - global chroma distribution
  - chroma edge strength
  - local chroma irregularity at the 8×8 block scale ("wrinkles")
  - with clear, interpretable statistics.
- We are not blindly throwing deep features at the problem; we are **designing features that are directly linked to the known JPEG processing pipeline** (YCbCr transform, sub‑sampling, DCT block structure).
- These features are then available to any classifier we choose later and can be analyzed for importance, which is important for explanation and forensics.

---

## 6. How to run and where the output goes

From the project root:

```bash
python src/chroma_features.py
```

What it does:

- Takes the first `N_IMAGES` in `data/compressed/` (controlled by `N_IMAGES` in the `__main__` block).
- Computes all chroma features.
- Writes:

```text
test_output/chroma_features_sample.csv
```

Columns:

- 18 numeric feature columns:
  - `chroma_Cb_mean`, `chroma_Cb_std`, `chroma_Cb_var`
  - `chroma_Cr_mean`, `chroma_Cr_std`, `chroma_Cr_var`
  - `chroma_Cb_edge_mean`, `chroma_Cb_edge_std`
  - `chroma_Cr_edge_mean`, `chroma_Cr_edge_std`
  - `chroma_Cb_blockvar_mean`, `chroma_Cb_blockvar_std`, `chroma_Cb_blockvar_max`
  - `chroma_Cr_blockvar_mean`, `chroma_Cr_blockvar_std`, `chroma_Cr_blockvar_max`
  - `luma_Y_mean`, `luma_Y_std`
- `file`  – filename of the JPEG
- `LABEL` – cluster label (`C0`, `C1`, `C2`, `C3`)

Optionally, if `dataframe2arff` is available and imported, it also writes:

```text
test_output/chroma_features_sample.arff
```

for use in Weka.

---

## 7. Integration with the rest of the project

- This module **does not replace** `bulk_classify.py`; it **extends** the feature space.
- Later, we can join:
  - SSIM‑based divergence features (`output_addLayer.csv`)
  - chroma features (`chroma_features_sample.csv`)
  - other feature families (YCbCr statistics, DCT‑domain features)

  by merging on the `file` column and using a common `LABEL`.

- The combined dataset can then be used to:
  - train explainable models (decision trees, random forests)
  - inspect which features (for example, which chroma statistics) are most informative for distinguishing the libjpeg clusters.


