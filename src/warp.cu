/*
 * File: warp.cu
 * Author: Phan Nhat Thien Nhan (Phase 6)
 * 
 * General Purpose:
 *   Warp image B into image A's coordinate frame and blend both images onto a
 *   panorama canvas:
 *   1. Compute the canvas bounding box enclosing image A and the 4 corners of
 *      image B warped by H_BtoA.
 *   2. Invert H_BtoA to obtain H_AtoB for inverse bilinear warping.
 *   3. Copy pixels of A in place, inverse-warp pixels of B with bilinear
 *      interpolation, and blend overlapping pixels using linear feathering
 *      (if p.feather is enabled).
 * 
 * Inputs:
 *   - A: Left/reference 8-bit RGB Image.
 *   - B: Right/target 8-bit RGB Image.
 *   - H_BtoA: 3x3 Homography matrix mapping points from B -> A.
 *   - p: Parameters struct (p.feather = true).
 * 
 * Outputs:
 *   - panorama: Output stitched 8-bit RGB Image.
 *   - Returns true on success, false if H_BtoA is singular or canvas is invalid.
 */

#include "stages.cuh"
#include "types.cuh"
#include "utils.cuh"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// Sanity bound on output canvas dimensions to guard against degenerate warps
constexpr int MAX_PANORAMA_DIM = 16384;

/*
 * General Purpose:
 *   Compute the analytical inverse of a 3x3 homography matrix H.
 * 
 * Inputs:
 *   - H: Input 3x3 Homography matrix.
 *   - H_inv: Output inverted 3x3 Homography matrix.
 * 
 * Outputs:
 *   - true if H is invertible, false if singular.
 */
static bool invertHomography3x3(const Homography& H, Homography& H_inv) {
    const double* h = H.h;

    const double c00 =  (h[4] * h[8] - h[5] * h[7]);
    const double c01 = -(h[3] * h[8] - h[5] * h[6]);
    const double c02 =  (h[3] * h[7] - h[4] * h[6]);

    const double c10 = -(h[1] * h[8] - h[2] * h[7]);
    const double c11 =  (h[0] * h[8] - h[2] * h[6]);
    const double c12 = -(h[0] * h[7] - h[1] * h[6]);

    const double c20 =  (h[1] * h[5] - h[2] * h[4]);
    const double c21 = -(h[0] * h[5] - h[2] * h[3]);
    const double c22 =  (h[0] * h[4] - h[1] * h[3]);

    const double det = h[0] * c00 + h[1] * c01 + h[2] * c02;
    if (std::fabs(det) < 1e-12) {
        return false;
    }

    const double invDet = 1.0 / det;

    H_inv.h[0] = c00 * invDet;
    H_inv.h[1] = c10 * invDet;
    H_inv.h[2] = c20 * invDet;

    H_inv.h[3] = c01 * invDet;
    H_inv.h[4] = c11 * invDet;
    H_inv.h[5] = c21 * invDet;

    H_inv.h[6] = c02 * invDet;
    H_inv.h[7] = c12 * invDet;
    H_inv.h[8] = c22 * invDet;

    return true;
}

/*
 * General Purpose:
 *   Sample an RGB pixel from an image at continuous coordinates (x, y) using
 *   2D bilinear interpolation.
 * 
 * Inputs:
 *   - img: Source 8-bit RGB Image.
 *   - x, y: Continuous coordinates inside [0, width - 1] x [0, height - 1].
 *   - outRgb: Output 3-element float RGB array.
 * 
 * Outputs:
 *   - Interpolated RGB values written to outRgb.
 */
static inline void sampleBilinearRGB(const Image& img, double x, double y, float outRgb[3]) {
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, img.width - 1);
    const int y1 = std::min(y0 + 1, img.height - 1);

    const float alpha = static_cast<float>(x - static_cast<double>(x0));
    const float beta  = static_cast<float>(y - static_cast<double>(y0));

    const float w00 = (1.0f - alpha) * (1.0f - beta);
    const float w10 = alpha * (1.0f - beta);
    const float w01 = (1.0f - alpha) * beta;
    const float w11 = alpha * beta;

    const uint8_t* p00 = &img.data[(y0 * img.width + x0) * 3];
    const uint8_t* p10 = &img.data[(y0 * img.width + x1) * 3];
    const uint8_t* p01 = &img.data[(y1 * img.width + x0) * 3];
    const uint8_t* p11 = &img.data[(y1 * img.width + x1) * 3];

    for (int c = 0; c < 3; ++c) {
        outRgb[c] = w00 * static_cast<float>(p00[c])
                  + w10 * static_cast<float>(p10[c])
                  + w01 * static_cast<float>(p01[c])
                  + w11 * static_cast<float>(p11[c]);
    }
}

/*
 * General Purpose:
 *   Compute a linear distance-to-border feathering weight for a pixel at (x, y).
 * 
 * Inputs:
 *   - x, y: Pixel coordinates within the image.
 *   - width, height: Image dimensions.
 * 
 * Outputs:
 *   - Non-negative feathering weight.
 */
static inline float computeFeatherWeight(double x, double y, int width, int height) {
    const double dx = std::min(x + 1.0, static_cast<double>(width) - x);
    const double dy = std::min(y + 1.0, static_cast<double>(height) - y);
    return static_cast<float>(std::max(0.0, dx) * std::max(0.0, dy));
}

namespace v0 {

/*
 * General Purpose:
 *   Sequential CPU Baseline (V0) for the image warping and blending stage:
 *   Computes the bounding box of A and warped B, inverse-warps B using bilinear
 *   interpolation, copies A in place, and blends overlapping pixels.
 * 
 * Inputs:
 *   - A: Left/reference 8-bit RGB Image.
 *   - B: Right/target 8-bit RGB Image.
 *   - H_BtoA: Homography mapping B -> A.
 *   - p: Pipeline parameters (p.feather).
 * 
 * Outputs:
 *   - panorama: Stitched output Image.
 *   - Returns true on success, false if the warp is degenerate.
 */
bool warpAndBlend(const Image& A, const Image& B, const Homography& H_BtoA,
                  const Params& p, Image& panorama) {
    panorama.width  = 0;
    panorama.height = 0;
    panorama.data.clear();

    if (A.empty() || B.empty()) {
        return false;
    }

    Homography H_AtoB;
    if (!invertHomography3x3(H_BtoA, H_AtoB)) {
        return false;
    }

    // =========================================================================
    // Step 1: Compute Panorama Canvas Bounding Box
    // =========================================================================
    double minX = 0.0;
    double maxX = static_cast<double>(A.width - 1);
    double minY = 0.0;
    double maxY = static_cast<double>(A.height - 1);

    const double cornersBX[4] = { 0.0, static_cast<double>(B.width - 1),
                                  static_cast<double>(B.width - 1), 0.0 };
    const double cornersBY[4] = { 0.0, 0.0,
                                  static_cast<double>(B.height - 1), static_cast<double>(B.height - 1) };

    for (int i = 0; i < 4; ++i) {
        const double xB = cornersBX[i];
        const double yB = cornersBY[i];
        const double w  = H_BtoA.h[6] * xB + H_BtoA.h[7] * yB + H_BtoA.h[8];
        if (w <= 1e-6) {
            return false;
        }

        double wx, wy;
        applyHomography(H_BtoA, xB, yB, wx, wy);

        minX = std::min(minX, wx);
        maxX = std::max(maxX, wx);
        minY = std::min(minY, wy);
        maxY = std::max(maxY, wy);
    }

    const int xStart = static_cast<int>(std::floor(minX));
    const int xEnd   = static_cast<int>(std::ceil(maxX));
    const int yStart = static_cast<int>(std::floor(minY));
    const int yEnd   = static_cast<int>(std::ceil(maxY));

    const int outW = xEnd - xStart + 1;
    const int outH = yEnd - yStart + 1;

    if (outW <= 0 || outH <= 0 || outW > MAX_PANORAMA_DIM || outH > MAX_PANORAMA_DIM) {
        return false;
    }

    panorama.width  = outW;
    panorama.height = outH;
    panorama.data.assign(static_cast<size_t>(outW) * outH * 3, 0);

    // =========================================================================
    // Step 2: Inverse Warp B, Copy A, and Blend Overlap Region
    // =========================================================================
    const double* hInv = H_AtoB.h;

    for (int r = 0; r < outH; ++r) {
        const int yA = yStart + r;
        const bool yInA = (yA >= 0 && yA < A.height);

        for (int c = 0; c < outW; ++c) {
            const int xA = xStart + c;
            const bool inA = yInA && (xA >= 0 && xA < A.width);

            const double wB = hInv[6] * xA + hInv[7] * yA + hInv[8];
            bool inB = false;
            double xB = 0.0, yB = 0.0;

            if (wB > 1e-6) {
                const double invWB = 1.0 / wB;
                xB = (hInv[0] * xA + hInv[1] * yA + hInv[2]) * invWB;
                yB = (hInv[3] * xA + hInv[4] * yA + hInv[5]) * invWB;

                if (xB >= 0.0 && xB <= static_cast<double>(B.width - 1) &&
                    yB >= 0.0 && yB <= static_cast<double>(B.height - 1)) {
                    inB = true;
                }
            }

            if (!inA && !inB) {
                continue;
            }

            uint8_t* dstPx = &panorama.data[(static_cast<size_t>(r) * outW + c) * 3];

            if (inA && !inB) {
                const uint8_t* srcA = &A.data[(static_cast<size_t>(yA) * A.width + xA) * 3];
                dstPx[0] = srcA[0];
                dstPx[1] = srcA[1];
                dstPx[2] = srcA[2];
            } else if (!inA && inB) {
                float rgbB[3];
                sampleBilinearRGB(B, xB, yB, rgbB);
                for (int ch = 0; ch < 3; ++ch) {
                    dstPx[ch] = static_cast<uint8_t>(clampInt(static_cast<int>(std::round(rgbB[ch])), 0, 255));
                }
            } else {
                const uint8_t* srcA = &A.data[(static_cast<size_t>(yA) * A.width + xA) * 3];

                if (!p.feather) {
                    dstPx[0] = srcA[0];
                    dstPx[1] = srcA[1];
                    dstPx[2] = srcA[2];
                } else {
                    float rgbB[3];
                    sampleBilinearRGB(B, xB, yB, rgbB);

                    const float wA = computeFeatherWeight(static_cast<double>(xA), static_cast<double>(yA), A.width, A.height);
                    const float wB_weight = computeFeatherWeight(xB, yB, B.width, B.height);
                    const float weightSum = wA + wB_weight;

                    const float alphaA = (weightSum > 1e-6f) ? (wA / weightSum) : 0.5f;
                    const float alphaB = 1.0f - alphaA;

                    for (int ch = 0; ch < 3; ++ch) {
                        float blended = alphaA * static_cast<float>(srcA[ch]) + alphaB * rgbB[ch];
                        dstPx[ch] = static_cast<uint8_t>(clampInt(static_cast<int>(std::round(blended)), 0, 255));
                    }
                }
            }
        }
    }

    return true;
}

} // namespace v0
