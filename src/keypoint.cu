/*
 * File: keypoint.cu
 * Author: Phan Nhat Thien Nhan (Phase 2)
 * 
 * General Purpose:
 *   Detect salient, repeatable corner keypoints in the smoothed grayscale image:
 *   1. Detect candidate corners using the FAST-9 segment test on a 16-pixel
 *      Bresenham circle of radius 3.
 *   2. Compute the Harris corner response score for each FAST-9 candidate over
 *      a 3x3 gradient window.
 *   3. Apply 3x3 Non-Maximum Suppression (NMS) within the valid border margin.
 *   4. Sort surviving keypoints deterministically by score descending (ties broken
 *      by (y, x) ascending) and keep the top p.maxKeypoints.
 * 
 * Inputs:
 *   - blurred: Smoothed single-channel float GrayImage from Stage 1.
 *   - p: Parameters struct (p.fastThreshold = 20, p.fastArc = 9, p.nmsRadius = 1,
 *        p.maxKeypoints = 3000, p.borderMargin = 20).
 * 
 * Outputs:
 *   - kps: Vector of detected Keypoint structs with x, y, score filled.
 */

#include "stages.cuh"
#include "types.cuh"
#include <algorithm>
#include <cmath>
#include <vector>

// Number of pixels on the discrete Bresenham circle of radius 3
constexpr int FAST_RING_SIZE = 16;

// Empirical Harris sensitivity constant k
constexpr float HARRIS_K = 0.04f;

// Relative (dx, dy) offsets of the 16-pixel Bresenham circle of radius 3
static const int kFastCircleX[FAST_RING_SIZE] = {
     0,  1,  2,  3,  3,  3,  2,  1,
     0, -1, -2, -3, -3, -3, -2, -1
};

static const int kFastCircleY[FAST_RING_SIZE] = {
    -3, -3, -2, -1,  0,  1,  2,  3,
     3,  3,  2,  1,  0, -1, -2, -3
};

/*
 * General Purpose:
 *   Test whether pixel (x, y) is a FAST corner candidate with a contiguous arc
 *   of at least arcLen pixels all brighter than (Ip + threshold) or all darker
 *   than (Ip - threshold) on the 16-pixel Bresenham circle of radius 3.
 * 
 * Inputs:
 *   - img: Pointer to row-major float image data.
 *   - width: Image width in pixels.
 *   - x, y: Candidate pixel coordinates.
 *   - threshold: Intensity difference threshold (p.fastThreshold).
 *   - arcLen: Minimum contiguous arc length (p.fastArc).
 * 
 * Outputs:
 *   - true if (x, y) passes the FAST segment test, false otherwise.
 */
static bool isFastCorner(const float* img, int width, int x, int y, float threshold, int arcLen) {
    const float centerVal = img[y * width + x];
    const float hiThresh  = centerVal + threshold;
    const float loThresh  = centerVal - threshold;

    // Cardinal early rejection check at indices 0, 4, 8, 12
    if (arcLen >= 8) {
        const float v0  = img[(y - 3) * width + x];
        const float v8  = img[(y + 3) * width + x];
        const float v4  = img[y * width + (x + 3)];
        const float v12 = img[y * width + (x - 3)];

        const bool canBeBrighter = (v0 > hiThresh || v8 > hiThresh) && (v4 > hiThresh || v12 > hiThresh);
        const bool canBeDarker   = (v0 < loThresh || v8 < loThresh) && (v4 < loThresh || v12 < loThresh);

        if (!canBeBrighter && !canBeDarker) {
            return false;
        }
    }

    // Check for a contiguous run of length >= arcLen around the 16-pixel ring
    int runBrighter = 0;
    int runDarker   = 0;
    const int totalSteps = FAST_RING_SIZE + arcLen - 1;

    for (int i = 0; i < totalSteps; ++i) {
        int idx = i & 15;
        float val = img[(y + kFastCircleY[idx]) * width + (x + kFastCircleX[idx])];

        if (val > hiThresh) {
            ++runBrighter;
            runDarker = 0;
            if (runBrighter >= arcLen) return true;
        } else if (val < loThresh) {
            ++runDarker;
            runBrighter = 0;
            if (runDarker >= arcLen) return true;
        } else {
            runBrighter = 0;
            runDarker   = 0;
        }
    }

    return false;
}

/*
 * General Purpose:
 *   Compute the Harris corner response score at pixel (x, y) over a 3x3 window
 *   using central-difference spatial gradients.
 * 
 * Inputs:
 *   - img: Pointer to row-major float image data.
 *   - width: Image width in pixels.
 *   - x, y: Pixel coordinates.
 * 
 * Outputs:
 *   - Harris response score R = det(M) - k * trace(M)^2.
 */
static float computeHarrisScore(const float* img, int width, int x, int y) {
    float sxx = 0.0f;
    float syy = 0.0f;
    float sxy = 0.0f;

    for (int dy = -1; dy <= 1; ++dy) {
        int py = y + dy;
        int rowOff = py * width;

        for (int dx = -1; dx <= 1; ++dx) {
            int px = x + dx;

            float ix = 0.5f * (img[rowOff + (px + 1)] - img[rowOff + (px - 1)]);
            float iy = 0.5f * (img[(py + 1) * width + px] - img[(py - 1) * width + px]);

            sxx += ix * ix;
            syy += iy * iy;
            sxy += ix * iy;
        }
    }

    float det   = sxx * syy - sxy * sxy;
    float trace = sxx + syy;
    return det - HARRIS_K * trace * trace;
}

namespace v0 {

/*
 * General Purpose:
 *   Sequential CPU Baseline (V0) for the keypoint detection stage:
 *   Detects FAST-9 corners, scores them with the Harris response, applies 3x3
 *   Non-Maximum Suppression within the valid border margin, and retains the top
 *   p.maxKeypoints sorted deterministically.
 * 
 * Inputs:
 *   - blurred: Smoothed single-channel float GrayImage from Stage 1.
 *   - p: Pipeline parameters.
 * 
 * Outputs:
 *   - kps: Detected keypoints sorted by score descending, ties broken by (y, x) ascending.
 */
void detectKeypoints(const GrayImage& blurred, const Params& p, std::vector<Keypoint>& kps) {
    kps.clear();

    const int width  = blurred.width;
    const int height = blurred.height;

    const int minSafeMargin = 4;
    const int borderMargin  = std::max(p.borderMargin, minSafeMargin);
    const int nmsRadius     = std::max(p.nmsRadius, 0);
    const int detectMargin  = std::max(borderMargin - nmsRadius, minSafeMargin);

    if (blurred.empty() || width <= 2 * borderMargin || height <= 2 * borderMargin) {
        return;
    }

    const float* img = blurred.data.data();
    const float fastThresh = static_cast<float>(p.fastThreshold);

    // =========================================================================
    // Step 1: FAST-9 Corner Detection & Harris Response Scoring
    // =========================================================================
    std::vector<uint8_t> isCorner(width * height, 0);
    std::vector<float>   scores(width * height, 0.0f);

    for (int y = detectMargin; y < height - detectMargin; ++y) {
        for (int x = detectMargin; x < width - detectMargin; ++x) {
            if (isFastCorner(img, width, x, y, fastThresh, p.fastArc)) {
                isCorner[y * width + x] = 1;
                scores[y * width + x]   = computeHarrisScore(img, width, x, y);
            }
        }
    }

    // =========================================================================
    // Step 2: 3x3 Non-Maximum Suppression (NMS) & Border Filtering
    // =========================================================================
    for (int y = borderMargin; y < height - borderMargin; ++y) {
        for (int x = borderMargin; x < width - borderMargin; ++x) {
            int idx = y * width + x;
            if (!isCorner[idx]) {
                continue;
            }

            float centerScore = scores[idx];
            bool isLocalMax = true;

            for (int dy = -nmsRadius; dy <= nmsRadius && isLocalMax; ++dy) {
                int ny = y + dy;
                for (int dx = -nmsRadius; dx <= nmsRadius; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    int nx = x + dx;
                    int nIdx = ny * width + nx;

                    if (isCorner[nIdx] && scores[nIdx] >= centerScore) {
                        isLocalMax = false;
                        break;
                    }
                }
            }

            if (isLocalMax) {
                Keypoint kp;
                kp.x     = static_cast<float>(x);
                kp.y     = static_cast<float>(y);
                kp.score = centerScore;
                kp.angle = 0.0f;
                kps.push_back(kp);
            }
        }
    }

    // =========================================================================
    // Step 3: Deterministic Sorting & Top-K Selection
    // =========================================================================
    std::sort(kps.begin(), kps.end(), [](const Keypoint& a, const Keypoint& b) {
        if (a.score != b.score) return a.score > b.score;
        if (a.y != b.y)         return a.y < b.y;
        return a.x < b.x;
    });

    if (p.maxKeypoints > 0 && static_cast<int>(kps.size()) > p.maxKeypoints) {
        kps.resize(p.maxKeypoints);
    }
}

} // namespace v0
