# Parallel Image Stitching (Panorama)

> **Course**: Parallel Programming – Midterm Project  
> **Group**: 03  

---

## Project Overview
This project implements an end-to-end panoramic image stitching pipeline from scratch in C++/CUDA. The system takes two overlapping photos, detects distinctive feature points, establishes correspondences, estimates the projective transformation (homography), and blends them into a seamless panorama.

---

## Repository Structure

```text
├── dataset/
├── src/
│   ├── preprocess.cu           # Stage 1: Grayscale & Separable Gaussian Blur
│   ├── keypoint.cu             # Stage 2: FAST-9 & Harris keypoint detection
│   ├── describe.cu             # Stage 3: Intensity centroid & rBRIEF descriptor
│   ├── matching.cu             # Stage 4: Hamming distance brute-force matcher
│   ├── homography.cu           # Stage 5: RANSAC 4-point DLT estimation
│   ├── warp.cu                 # Stage 6: Inverse warping & linear feathering
│   ├── main.cu                 # CLI entry point, benchmark harness & timing
│   ├── stages.cuh              # Pipeline stage interfaces
│   ├── types.cuh               # Core data structures (Image, Keypoint, etc.)
│   ├── utils.cuh               # CLI parsing, timing & image I/O helpers
│   ├── utils.cu                # stb_image wrapper & utility implementations
│   └── build.sh                # Build script using nvcc
├── Group03_Midterm.ipynb       # Full experiment, visualization & report notebook
└── README.md                   # Project documentation
```

---

## Quick Start

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
Open our notebook in **Google Colab** (with GPU runtime) or local Jupyter to view step-by-step visualizations, homography inlier plots, and execution time breakdowns across resolution tiers.

---

## Team Information (Group 03)

| Student Name | Student ID | Primary Responsibilities |
|---|:---:|---|
| **Nguyen Hoang Minh Tam** | `23127017` | Standardize codebase & datasets; Implement `homography.cu`, `warp.cu` |
| **Nguyen Manh Cuong** | `23127034` | Dataset synthesis; Implement `keypoint.cu`, `matching.cu`; Benchmark scripts & report |
| **Phan Nhat Thien Nhan** | `23127096` | Organize repository; Implement `preprocess.cu`, `describe.cu`; Pipeline docs |
