/*
 * File: preprocess.cu
 * Author: Phan Nhat Thien Nhan (Phase 1)
 * 
 * General Purpose:
 *   Preprocess the input color image in the image stitching pipeline:
 *   1. Convert 8-bit interleaved RGB image to single-channel float Grayscale.
 *   2. Smooth the grayscale image using a 2-pass Separable Gaussian Blur:
 *      - Pass 1 (Horizontal): 1D Gaussian convolution along image rows.
 *      - Pass 2 (Vertical): 1D Gaussian convolution along image columns.
 * 
 * Inputs:
 *   - rgb: Source color Image (interleaved 3-channel uint8: R, G, B).
 *   - p: Parameters struct (p.blurSigma = 2.0f, p.blurRadius = 3).
 * 
 * Outputs:
 *   - blurred: Smoothed single-channel float GrayImage (values in [0.0f, 255.0f]).
 */

#include "stages.cuh"
#include "types.cuh"
#include <cmath>
#include <vector>

/*
 * General Purpose:
 *   Clamp an integer index into the valid boundary [lo, hi] (Clamp-to-edge).
 * 
 * Inputs:
 *   - val: Target index to be tested.
 *   - lo: Lower bound (inclusive).
 *   - hi: Upper bound (inclusive).
 * 
 * Outputs:
 *   - Clamped index within [lo, hi].
 */
static inline int clampInt(int val, int lo, int hi) {
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}

/*
 * General Purpose:
 *   Generate a normalized discrete 1D Gaussian filter kernel.
 * 
 * Inputs:
 *   - sigma: Standard deviation of the Gaussian distribution (p.blurSigma).
 *   - radius: Kernel radius (p.blurRadius; kernel size is 2 * radius + 1).
 * 
 * Outputs:
 *   - kernel: Output vector containing normalized float weights summing to 1.0f.
 */
static void generateGaussianKernel1d(float sigma, int radius, std::vector<float>& kernel) {
    int kernelSize = 2 * radius + 1;
    kernel.resize(kernelSize);

    float sum = 0.0f;
    float twoSigmaSquare = 2.0f * sigma * sigma;

    for (int k = -radius; k <= radius; ++k) {
        float weight = std::exp(-static_cast<float>(k * k) / twoSigmaSquare);
        kernel[k + radius] = weight;
        sum += weight;
    }

    // Normalize weights to sum to 1.0 (preserves overall image brightness)
    for (int i = 0; i < kernelSize; ++i) {
        kernel[i] /= sum;
    }
}

namespace v0 {

/*
 * General Purpose:
 *   Sequential CPU Baseline (V0) for the preprocessing stage:
 *   Converts RGB to grayscale using standard ITU-R BT.601 luminance weights,
 *   followed by separable 1D horizontal and vertical Gaussian smoothing.
 * 
 * Inputs:
 *   - rgb: Input 8-bit RGB Image (interleaved, row-major).
 *   - p: Pipeline parameters (blurSigma, blurRadius).
 * 
 * Outputs:
 *   - blurred: Output smoothed GrayImage (single channel float, row-major).
 */
void preprocess(const Image& rgb, const Params& p, GrayImage& blurred) {
    int width = rgb.width;
    int height = rgb.height;

    blurred.width = width;
    blurred.height = height;
    blurred.data.clear();

    if (width <= 0 || height <= 0 || rgb.data.empty()) {
        return;
    }

    blurred.data.resize(width * height);

    // =========================================================================
    // Step 1: Convert RGB to Grayscale (Float)
    // Formula (ITU-R BT.601): Gray = 0.299*R + 0.587*G + 0.114*B
    // =========================================================================
    std::vector<float> gray(width * height);

    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int rgbIdx = (r * width + c) * 3;
            float redVal   = static_cast<float>(rgb.data[rgbIdx + 0]);
            float greenVal = static_cast<float>(rgb.data[rgbIdx + 1]);
            float blueVal  = static_cast<float>(rgb.data[rgbIdx + 2]);

            gray[r * width + c] = 0.299f * redVal + 0.587f * greenVal + 0.114f * blueVal;
        }
    }

    // =========================================================================
    // Step 2: Separable Gaussian Blur (2 passes of 1D convolution)
    // =========================================================================
    std::vector<float> kernel;
    generateGaussianKernel1d(p.blurSigma, p.blurRadius, kernel);
    int radius = p.blurRadius;

    // 2a. Pass 1 (Horizontal): convolve along rows into temporary buffer
    std::vector<float> tempBuffer(width * height);

    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            float sum = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int sampleCol = clampInt(c + k, 0, width - 1);
                sum += gray[r * width + sampleCol] * kernel[k + radius];
            }
            tempBuffer[r * width + c] = sum;
        }
    }

    // 2b. Pass 2 (Vertical): convolve along columns into blurred.data
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            float sum = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int sampleRow = clampInt(r + k, 0, height - 1);
                sum += tempBuffer[sampleRow * width + c] * kernel[k + radius];
            }
            blurred.data[r * width + c] = sum;
        }
    }
}

} // namespace v0
