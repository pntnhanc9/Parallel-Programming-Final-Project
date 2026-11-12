# Parallel Image Stitching (Panorama)

> **Course**: Parallel Programming – Mid-term Project  
> **Group**: 03  

---

## 📌 Project Overview
This project implements an end-to-end panoramic image stitching pipeline from scratch in C++/CUDA. The system takes two overlapping photos, detects distinctive feature points, establishes correspondences, estimates the projective transformation (homography), and blends them into a seamless panorama.

---

## 🔄 The 6-Stage Pipeline

| Stage | Algorithm / Method Chosen | Key Highlights |
|---|---|---|
| **1. Preprocessing** | ITU-R BT.601 Grayscale + 2-Pass Separable Gaussian Blur | $\sigma=2.0, r=3$; $2K$ vs $K^2$ filter ops; Neumann clamp boundary |
| **2. Keypoint Detection** | FAST-9 + Harris Corner Score + $3\times3$ NMS | 4-point cardinal early rejection; deterministic Top-$K$ (3000) |
| **3. Feature Description** | Intensity Centroid + Rotated BRIEF (rBRIEF) | 256-bit binary descriptor (32 bytes); LCG seed 2026; rotation-invariant |
| **4. Feature Matching** | Brute-force Hamming Distance | Hardware `__popc` + XOR; Lowe's ratio test ($0.75$); mutual cross-check |
| **5. Homography** | RANSAC with 4-Point Normalized DLT | Deterministic sampling; Hartley normalization; $8\times8$ Gaussian elimination |
| **6. Warping & Blending** | Inverse Homography + Bilinear Sampling + Feathering | Analytical $3\times3$ matrix inversion; linear distance-to-border blending |

---

## 📁 Repository Structure

```text
├── dataset/
├── src/
│   ├── preprocess.cu      # Stage 1: Grayscale & Separable Gaussian Blur
│   ├── keypoint.cu        # Stage 2: FAST-9 & Harris keypoint detection
│   ├── describe.cu        # Stage 3: Intensity centroid & rBRIEF descriptor
│   ├── matching.cu        # Stage 4: Hamming distance brute-force matcher
│   ├── homography.cu      # Stage 5: RANSAC 4-point DLT estimation
│   ├── warp.cu            # Stage 6: Inverse warping & linear feathering
│   ├── main.cu            # CLI entry point, benchmark harness & timing
│   ├── stages.cuh         # Pipeline stage interfaces
│   ├── types.cuh          # Core data structures (Image, Keypoint, etc.)
│   ├── utils.cuh          # CLI parsing, timing & image I/O helpers
│   ├── utils.cu           # stb_image wrapper & utility implementations
│   ├── cuda_utils.cuh     # CUDA timers & error-checking macros
│   └── build.sh           # Build script using nvcc
├── main.ipynb             # Full experiment, visualization & report notebook
└── README.md              # Project documentation
```

---

## 🚀 Quick Start

### 1. Build from Source
Requirements: `CUDA Toolkit >= 11.0` (with `nvcc`) or `g++ >= 9.0` supporting C++17.

```bash
# Build the binary ./src/stitch
bash src/build.sh

# Clean build artifacts
bash src/build.sh clean
```

### 2. Run Image Stitching
```bash
./src/stitch <left_image> <right_image> <output_image> --version 0 [--runs N]

# Example run with sample images:
./src/stitch aqua_L.jpg aqua_R.jpg output.jpg --version 0 --runs 5
```
* `--version 0`: CPU sequential baseline.
* `--runs N`: Number of timed runs (includes 1 warm-up run) to compute median, min, max execution times.

### 3. Run Experiments via Notebook
Open [`main.ipynb`](main.ipynb) in **Google Colab** (with GPU runtime) or local Jupyter to view step-by-step visualizations, homography inlier plots, and execution time breakdowns across resolution tiers.

---

## 📊 Datasets

The pipeline is benchmarked across three resolution tiers to evaluate scaling behavior:

| Resolution Tier | Typical Resolution | Planned Role |
|---|---|---|
| **Small** | `1280 x 720` (720p) | Fast functional sanity check & low-overhead baseline |
| **Medium** | `1920 x 1080` (1080p) | Standard benchmark comparison |
| **Large** | `3740 x 2160` (4K) | High-resolution compute & memory-bandwidth stress test |

---

## 👥 Team Information (Group 03)

| Student Name | Student ID | Primary Responsibilities |
|---|:---:|---|
| **Nguyen Hoang Minh Tam** | `23127017` | Standardize codebase & datasets; Implement `homography.cu`, `warp.cu` |
| **Nguyen Manh Cuong** | `23127034` | Dataset synthesis; Implement `keypoint.cu`, `matching.cu`; Benchmark scripts & report |
| **Phan Nhat Thien Nhan** | `23127096` | Organize repository; Implement `preprocess.cu`, `describe.cu`; Pipeline docs |
