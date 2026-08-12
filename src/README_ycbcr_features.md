# YCbCr Feature Extraction (`ycbcr_features.py`)

## 1. High‑level idea

The JPEG encoder operates in the **YCbCr color space**:

- `Y`  – luma (brightness)
- `Cb` – blue‑difference chroma
- `Cr` – red‑difference chroma

Different libjpeg versions (and families C0–C3) use slightly different:

- quantization tables for Y vs Cb/Cr
- chroma subsampling schemes
- rounding implementations

These choices can change the **global distributions** of Y, Cb, and Cr values in subtle but systematic ways. The `ycbcr_features.py` module extracts simple, interpretable statistics from each channel to capture those differences.

We use these YCbCr statistics as an additional feature family, complementary to:

- the SSIM‑based double‑compression matrix features (`bulk_classify.py`)
- the local chroma "wrinkle" features (`chroma_features.py`)

Each image becomes **one feature row** with a **cluster label C0–C3**.

---

## 2. Pipeline summary

For each JPEG in `data/compressed/`:

1. Load the image as BGR with OpenCV (`cv2.imread`).
2. Convert it to **YCrCb** color space (`cv2.COLOR_BGR2YCrCb`).
3. Extract channels:
   - `Y`  – luma
   - `Cr` – red chroma
   - `Cb` – blue chroma
4. For each channel (Y, Cb, Cr) compute:
   - mean
   - standard deviation
   - variance
   - histogram‑based entropy
5. Compute two simple **comparative features**:
   - `Y_minus_Cb_mean`
   - `Y_minus_Cr_mean`
6. Infer the image’s **cluster label** (C0, C1, C2, C3) from the filename, using the same encoder→cluster mapping as in `chroma_features.py` / `bulk_classify.py`.
7. Write one row per image to `test_output/ycbcr_features_sample.csv` with:
   - all numeric features
   - `file`
   - `LABEL`

This script only builds a feature dataset; model training happens elsewhere.

---

## 3. Detailed feature definitions

Let `Y`, `Cb`, `Cr` be the three channels as 2D float arrays.

### 3.1 Per‑channel statistics

For each channel we compute:

- `<ch>_mean`  – average pixel value
- `<ch>_std`   – standard deviation of pixel values
- `<ch>_var`   – variance of pixel values
- `<ch>_entropy` – entropy of a 64‑bin histogram

Where `<ch>` is one of `Y`, `Cb`, `Cr`.

Implementation (simplified):

```python
ch = channel.reshape(-1).astype(np.float32)

mean = float(np.mean(ch))
std = float(np.std(ch))
var = float(np.var(ch))

hist, _ = np.histogram(ch, bins=64, range=(0, 255), density=True)
hist = hist + 1e-12  # avoid log(0)
entropy = float(-np.sum(hist * np.log(hist)))
```

These statistics capture:

- **mean**  – overall brightness / chroma bias (e.g., how strong the chroma components are on average)
- **std / var** – how spread out the values are (contrast in that channel)
- **entropy** – how complex or uniform the distribution is

Since libjpeg families may treat Y and Cb/Cr differently (e.g. stronger quantization for chroma), these statistics can help distinguish encoder clusters.

### 3.2 Comparative Y vs Cb/Cr features

We also compute two simple differences:

- `Y_minus_Cb_mean = Y_mean - Cb_mean`
- `Y_minus_Cr_mean = Y_mean - Cr_mean`

These measure how bright the image is relative to its chroma levels. For example:

- If `Y_minus_Cb_mean` is large positive, Y is much brighter than Cb on average.
- If it is small or negative, chroma amplitude is comparable to or larger than luma.

Different encoder families (and quantization tables) can change this relationship in subtle ways.

---

## 4. Label assignment: mapping to C0–C3

The label in `ycbcr_features_sample.csv` is again the **cluster label** for the libjpeg family that produced the JPEG:

- **C0:** legacy upsampling family (e.g., `6b`, `turbo120`, `turbo121`, `mozjpeg201`, ...)
- **C1:** DCT‑scaling family (e.g., `7`, `8`, `8a`, `9`, `9a`, `9b`, `9c`, `9d`)
- **C2:** new chrominance quantization (e.g., `9e`, `9f`)
- **C3:** mozjpeg progressive family (e.g., `mozjpeg300`, `mozjpeg403`, `mozjpeg101`)

We do **not** re‑implement the mapping logic here. Instead, we **reuse** the helper from `chroma_features.py`:

```python
from chroma_features import infer_cluster_from_filename
```

and in `build_ycbcr_dataset` we call:

```python
fname = Path(path).name
record["file"] = fname
record["LABEL"] = infer_cluster_from_filename(fname)
```

This guarantees that YCbCr features and chroma features use exactly the same encoder→cluster mapping as `bulk_classify.py`.

Example mappings:

- `01374_9d.jpeg`          → contains `"9d"`          → cluster `C1`
- `01486_mozjpeg101.jpeg`  → contains `"mozjpeg101"`  → cluster `C0` (per project lists)
- `00033_mozjpeg403.jpeg`  → contains `"mozjpeg403"`  → cluster `C3`
- `02020_mozjpeg300.jpeg`  → contains `"mozjpeg300"`  → cluster `C3`

We checked consistency by comparing `LABEL` here with the labels in `output_addLayer.csv`.

---

## 5. What this achieves / why it is useful

### 5.1 Capturing global Y vs Cb/Cr behavior

The YCbCr features summarize **how each encoder family shapes the overall Y, Cb, Cr distributions**, without focusing on local block structures:

- If an encoder strongly quantizes chroma, `Cb_std`, `Cr_std`, and their entropies may be systematically lower.
- If an encoder family changes luma quantization while keeping chroma similar, `Y_*` statistics will shift while `Cb_*` and `Cr_*` remain stable.
- The `Y_minus_Cb_mean` and `Y_minus_Cr_mean` features highlight whether luma and chroma amplitudes are aligned or skewed.

This is complementary to:

- **SSIM matrix features** (which look at double‑compression response), and
- **chroma wrinkle features** (which look at local block‑wise irregularities).

### 5.2 Simplicity and interpretability

All YCbCr features are:

- simple aggregates (mean, variance, entropy, differences)
- easy to compute
- straightforward to explain to non‑experts

This makes them ideal for:

- sanity‑checking the dataset
- quick exploratory analysis
- use in explainable models (decision trees, random forests) where we can inspect which Y/Cb/Cr stats are most important.

### 5.3 Lightweight feature family

Compared to recomputing a full double‑compression matrix for every feature extension, YCbCr features are very cheap:

- one pass over the image per channel
- one histogram per channel

They add **little computational overhead** but provide additional signal about encoder behavior.

---

## 6. How to run and where the output goes

From the project root:

```bash
python src/ycbcr_features.py
```

In the `__main__` block you can control how many images are processed by changing:

```python
N_IMAGES = 10
```

When you run the script it will:

1. Find the first `N_IMAGES` JPEGs under `data/compressed/`.
2. Compute Y, Cb, Cr feature statistics for each.
3. Infer the encoder cluster label from the filename.
4. Write the resulting table to:

```text
test_output/ycbcr_features_sample.csv
```

Columns include:

- `Y_mean`, `Y_std`, `Y_var`, `Y_entropy`
- `Cb_mean`, `Cb_std`, `Cb_var`, `Cb_entropy`
- `Cr_mean`, `Cr_std`, `Cr_var`, `Cr_entropy`
- `Y_minus_Cb_mean`, `Y_minus_Cr_mean`
- `file`
- `LABEL`

This CSV is structured so it can be directly merged (on `file`) with:

- chroma feature CSV (`chroma_features_sample.csv`)
- SSIM‑based feature CSV (`output_addLayer.csv`)
- other feature families.

---

## 7. Integration with the rest of the project

- `ycbcr_features.py` is a **standalone feature extractor**; it does not depend on the double‑compression pipeline.
- It **reuses** the same label mapping as `chroma_features.py` / `bulk_classify.py`, ensuring all feature sets agree on the cluster label.
- The resulting CSV can be joined with other feature tables on `file` and `LABEL` to build a **combined dataset** for training ML models.

This gives us another interpretable perspective on the encoder fingerprints:

> While the SSIM matrix captures how images react to recompression, and chroma wrinkles capture local chroma irregularities, the YCbCr features summarize how each encoder family shapes the global brightness and chroma distributions. Together, these feature families provide a richer and more explainable representation of JPEG library fingerprints.
