# Panorama Image Stitching (CUDA)

**Course:** CSC14120 – Parallel Programming (Final Project)  
**Institution:** Faculty of Information Technology, University of Science, VNU-HCM  
**Presentation Video:** [no link (yet)]()

---

Given two overlapping RGB photographs of the same scene, combine them into a single wide-angle image. The stitching process follows a six-stage process
---

## Team Information

| Student ID | Full Name | Role / Contributions |
| :--- | :--- | :--- |
| 23127017 | Nguyễn Hoàng Minh Tâm | `v0::preprocess`, `v0::describe`
| 23127034 | Nguyễn Mạnh Cường`    | `v0::keypoint`, `v0::match`
| 23127096 | Phan Nhật Thiện Nhân  | `v0::homography`, `v0::warping`

---

## Pipeline & Implemented Versions
| Version | Name | Description |
| :--- | :--- | :--- |
| `V0` | Sequential | Single-thread C/C++ reference implementation for correctness and speedup baseline. |
<!-- | `V1` | Basic CUDA | Straightforward GPU kernels for all stages using global memory. |
| `V2` | Optimized CUDA 1 | `[Main optimization idea and targeted bottleneck]` |
| `V3` | Optimized CUDA 2 | `[Main optimization idea and targeted bottleneck]` | -->

---

## Compilation Commands

```bash
# Example compilation command (update flags and source files to match your build)
[nvcc src/*.cu src/*.cpp -o stitch]
```

## Execution Instructions

All versions are invoked through a single executable using the `--version` flag:

```bash
./stitch <left_image> <right_image> <output_image> --version <0|1|2|...> [--runs N]
```

### Arguments

* `<left_image>`: Path to the left RGB input image (Image A).
* `<right_image>`: Path to the right RGB input image (Image B, warped onto Image A's frame).
* `<output_image>`: Path where the stitched panorama image will be saved.
* `--version <0|1|2|...>`: Pipeline version to run (`0` = Sequential `V0`, `1` = Basic CUDA `V1`, `2+` = Optimized CUDA versions).
* `--runs N`: Number of timed runs to average after 1 warm-up run (default: `5`).

### Example Commands

```bash
# Run V0 (Sequential baseline) on Medium resolution (1920x1080)
./stitch data/site1_1080p_left.png data/site1_1080p_right.png out_v0.png --version 0 --runs 5

# Run V1 (Basic CUDA) on Large resolution (3840x2160)
./stitch data/site1_4k_left.png data/site1_4k_right.png out_v1.png --version 1 --runs 5

# Run V2 (Optimized CUDA) on Large resolution (3840x2160)
./stitch data/site1_4k_left.png data/site1_4k_right.png out_v2.png --version 2 --runs 5
```

---

## Expected Outputs

1. **Stitched Panorama Image:** Saved to `<output_image>`, visually consistent with the `V0` reference output.
2. **Console Timing Report:** Printed after `1` warm-up run and `N` timed runs (excluding disk I/O), including:
   * Active GPU device name
   * Per-stage execution time (`ms`)
   * Total kernel time (`ms`)
   * Host–device memory transfer time (`ms`)
   * End-to-end pipeline time (`ms`)

### Sample Console Output

```text
GPU: [NVIDIA Tesla T4]
Version: V1 | Resolution: 1920x1080 | Runs: 5 (1 warm-up)
------------------------------------------------------------
Stage 1 (Preprocessing):        [X.XX] ms
Stage 2 (Keypoint Detection):   [X.XX] ms
Stage 3 (Description):          [X.XX] ms
Stage 4 (Matching):             [X.XX] ms
Stage 5 (Homography / RANSAC):  [X.XX] ms
Stage 6 (Warping & Blending):   [X.XX] ms
------------------------------------------------------------
Total Kernel Time:              [X.XX] ms
Host-Device Transfer Time:      [X.XX] ms
End-to-End Time (excl. I/O):    [X.XX] ms
Output saved to out_v1.png
```

---

## Dataset & Benchmarking

* **Sites:** 3 standard scene pairs (1 pair per site) + 1 challenging failure-analysis pair.
* **Resolutions:** Each 16:9 pair is evaluated at Small (`1280×720`), Medium (`1920×1080`), and Large (`3840×2160`).
* **Timing Methodology:** `std::chrono::high_resolution_clock` for `V0`; CUDA events (`cudaEventRecord` / `cudaEventElapsedTime`) for GPU stages.