/*
 * File: homography.cu
 * Author: Phan Nhat Thien Nhan (Phase 5)
 * 
 * General Purpose:
 *   Estimate the 3x3 planar homography matrix H (mapping image B to image A)
 *   using RANSAC and Normalized Direct Linear Transform (DLT):
 *   1. Pre-generate p.ransacIters 4-point match index samples deterministically
 *      from p.ransacSeed using a custom 32-bit LCG.
 *   2. Estimate candidate 3x3 homographies from 4 non-collinear correspondences
 *      using Hartley-normalized DLT.
 *   3. Count inliers whose reprojection error in image A is below p.ransacThresh
 *      (ties broken by lowest iteration index).
 *   4. Optionally re-estimate H over all consensus inliers if p.refitInliers is set.
 * 
 * Inputs:
 *   - kpsA, kpsB: Keypoint vectors for image A and image B.
 *   - matches: Vector of descriptor matches from Stage 4.
 *   - p: Parameters struct (p.ransacIters = 2000, p.ransacThresh = 3.0,
 *        p.ransacSeed = 1234, p.minInliers = 10, p.refitInliers = true).
 * 
 * Outputs:
 *   - H: Estimated 3x3 Homography struct mapping B -> A.
 *   - inlierMask: Binary mask (size = matches.size()), 1 if inlier, 0 otherwise.
 *   - numInliers: Total number of inliers found.
 */

#include "stages.cuh"
#include "types.cuh"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// Matched 2D point correspondence between image B and image A
struct PointPair2D {
    double xB, yB;
    double xA, yA;
};

/*
 * General Purpose:
 *   Minimal, fully deterministic 32-bit Linear Congruential Generator (LCG).
 *   Ensures identical pseudo-random sequences across all compilers, OS, and architectures.
 * 
 * Inputs:
 *   - state: Current PRNG state (modified in-place).
 * 
 * Outputs:
 *   - Next 32-bit unsigned pseudo-random integer.
 */
static inline uint32_t lcgNextRansac(uint32_t& state) {
    // Numerical Recipes / 64-bit LCG constants
    state = state * 1664525u + 1013904223u;
    return state;
}

/*
 * General Purpose:
 *   Check whether 4 point correspondences are geometrically non-degenerate
 *   (no 3 points collinear in either image A or image B).
 * 
 * Inputs:
 *   - pts: Array of 4 matched 2D point pairs.
 * 
 * Outputs:
 *   - true if no 3 points are collinear, false otherwise.
 */
static bool isQuadNonCollinear(const PointPair2D pts[4]) {
    constexpr double kMinArea2 = 1e-2;
    static const int kTriplets[4][3] = {
        {0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}
    };

    for (int t = 0; t < 4; ++t) {
        const PointPair2D& p1 = pts[kTriplets[t][0]];
        const PointPair2D& p2 = pts[kTriplets[t][1]];
        const PointPair2D& p3 = pts[kTriplets[t][2]];

        double crossB = (p2.xB - p1.xB) * (p3.yB - p1.yB) - (p2.yB - p1.yB) * (p3.xB - p1.xB);
        double crossA = (p2.xA - p1.xA) * (p3.yA - p1.yA) - (p2.yA - p1.yA) * (p3.xA - p1.xA);

        if (std::fabs(crossB) < kMinArea2 || std::fabs(crossA) < kMinArea2) {
            return false;
        }
    }
    return true;
}

/*
 * General Purpose:
 *   Solve an 8x8 linear system M * x = b using Gaussian elimination with partial pivoting.
 * 
 * Inputs:
 *   - M: 8x8 matrix of coefficients (modified in-place).
 *   - b: 8-element right-hand side vector (modified in-place).
 *   - x: 8-element output solution vector.
 * 
 * Outputs:
 *   - true if a unique solution exists, false if the matrix is singular.
 */
static bool solveLinearSystem8x8(double M[8][8], double b[8], double x[8]) {
    constexpr double kPivotEps = 1e-12;

    for (int col = 0; col < 8; ++col) {
        int pivotRow = col;
        double maxVal = std::fabs(M[col][col]);

        for (int row = col + 1; row < 8; ++row) {
            double val = std::fabs(M[row][col]);
            if (val > maxVal) {
                maxVal = val;
                pivotRow = row;
            }
        }

        if (maxVal < kPivotEps) {
            return false;
        }

        if (pivotRow != col) {
            for (int k = col; k < 8; ++k) {
                std::swap(M[col][k], M[pivotRow][k]);
            }
            std::swap(b[col], b[pivotRow]);
        }

        const double invPivot = 1.0 / M[col][col];
        for (int row = col + 1; row < 8; ++row) {
            double factor = M[row][col] * invPivot;
            M[row][col] = 0.0;
            for (int k = col + 1; k < 8; ++k) {
                M[row][k] -= factor * M[col][k];
            }
            b[row] -= factor * b[col];
        }
    }

    for (int row = 7; row >= 0; --row) {
        double sum = b[row];
        for (int k = row + 1; k < 8; ++k) {
            sum -= M[row][k] * x[k];
        }
        x[row] = sum / M[row][row];
    }

    return true;
}

/*
 * General Purpose:
 *   Estimate a 3x3 Homography H mapping B -> A from N >= 4 correspondences
 *   using Hartley-normalized Direct Linear Transform (DLT).
 * 
 * Inputs:
 *   - pairs: Array of N point correspondences (xB, yB) <-> (xA, yA).
 *   - numPairs: Number of correspondences (N >= 4).
 *   - H: Output 3x3 Homography matrix.
 * 
 * Outputs:
 *   - true if DLT succeeds, false if the configuration is degenerate.
 */
static bool solveNormalizedDLT(const PointPair2D* pairs, int numPairs, Homography& H) {
    if (numPairs < 4) {
        return false;
    }

    // Compute centroids in B and A
    double cxB = 0.0, cyB = 0.0;
    double cxA = 0.0, cyA = 0.0;
    for (int i = 0; i < numPairs; ++i) {
        cxB += pairs[i].xB;
        cyB += pairs[i].yB;
        cxA += pairs[i].xA;
        cyA += pairs[i].yA;
    }
    const double invN = 1.0 / static_cast<double>(numPairs);
    cxB *= invN; cyB *= invN;
    cxA *= invN; cyA *= invN;

    // Compute mean radial distances from centroids
    double meanDistB = 0.0;
    double meanDistA = 0.0;
    for (int i = 0; i < numPairs; ++i) {
        meanDistB += std::hypot(pairs[i].xB - cxB, pairs[i].yB - cyB);
        meanDistA += std::hypot(pairs[i].xA - cxA, pairs[i].yA - cyA);
    }
    meanDistB *= invN;
    meanDistA *= invN;

    if (meanDistB < 1e-9 || meanDistA < 1e-9) {
        return false;
    }

    constexpr double kSqrt2 = 1.4142135623730951;
    const double sB = kSqrt2 / meanDistB;
    const double sA = kSqrt2 / meanDistA;

    // Assemble 8x8 normal equations (J^T * J) * h = J^T * r
    double JTJ[8][8] = {{0.0}};
    double JTr[8]    = {0.0};

    for (int i = 0; i < numPairs; ++i) {
        const double uB = (pairs[i].xB - cxB) * sB;
        const double vB = (pairs[i].yB - cyB) * sB;
        const double uA = (pairs[i].xA - cxA) * sA;
        const double vA = (pairs[i].yA - cyA) * sA;

        const double row1[8] = { uB, vB, 1.0, 0.0, 0.0, 0.0, -uA * uB, -uA * vB };
        const double rhs1    = uA;

        const double row2[8] = { 0.0, 0.0, 0.0, uB, vB, 1.0, -vA * uB, -vA * vB };
        const double rhs2    = vA;

        for (int r = 0; r < 8; ++r) {
            JTr[r] += row1[r] * rhs1 + row2[r] * rhs2;
            for (int c = r; c < 8; ++c) {
                JTJ[r][c] += row1[r] * row1[c] + row2[r] * row2[c];
            }
        }
    }

    for (int r = 1; r < 8; ++r) {
        for (int c = 0; c < r; ++c) {
            JTJ[r][c] = JTJ[c][r];
        }
    }

    double hNorm[9] = {0.0};
    if (!solveLinearSystem8x8(JTJ, JTr, hNorm)) {
        return false;
    }
    hNorm[8] = 1.0;

    // Denormalize H = T_A^{-1} * H_norm * T_B
    const double TB[9] = {
        sB,  0.0, -sB * cxB,
        0.0, sB,  -sB * cyB,
        0.0, 0.0,  1.0
    };
    const double invSA = 1.0 / sA;
    const double TA_inv[9] = {
        invSA, 0.0,   cxA,
        0.0,   invSA, cyA,
        0.0,   0.0,   1.0
    };

    double temp[9] = {0.0};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            temp[r * 3 + c] = hNorm[r * 3 + 0] * TB[0 * 3 + c]
                            + hNorm[r * 3 + 1] * TB[1 * 3 + c]
                            + hNorm[r * 3 + 2] * TB[2 * 3 + c];
        }
    }

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            H.h[r * 3 + c] = TA_inv[r * 3 + 0] * temp[0 * 3 + c]
                           + TA_inv[r * 3 + 1] * temp[1 * 3 + c]
                           + TA_inv[r * 3 + 2] * temp[2 * 3 + c];
        }
    }

    if (std::fabs(H.h[8]) < 1e-12) {
        return false;
    }
    const double invH8 = 1.0 / H.h[8];
    for (int i = 0; i < 9; ++i) {
        H.h[i] *= invH8;
    }

    return true;
}

/*
 * General Purpose:
 *   Test whether a point correspondence (xB, yB) <-> (xA, yA) is an inlier
 *   under homography H within squared Euclidean threshold threshSq.
 * 
 * Inputs:
 *   - H: 3x3 Homography matrix mapping B -> A.
 *   - pair: Point correspondence (xB, yB, xA, yA).
 *   - threshSq: Squared reprojection threshold (p.ransacThresh^2).
 * 
 * Outputs:
 *   - true if forward-projected point in A is within threshSq, false otherwise.
 */
static inline bool isMatchInlier(const Homography& H, const PointPair2D& pair, double threshSq) {
    const double* h = H.h;
    const double w = h[6] * pair.xB + h[7] * pair.yB + h[8];
    if (w <= 1e-7) {
        return false;
    }

    const double invW   = 1.0 / w;
    const double projXA = (h[0] * pair.xB + h[1] * pair.yB + h[2]) * invW;
    const double projYA = (h[3] * pair.xB + h[4] * pair.yB + h[5]) * invW;

    const double dx = projXA - pair.xA;
    const double dy = projYA - pair.yA;
    return (dx * dx + dy * dy) < threshSq;
}

namespace v0 {

/*
 * General Purpose:
 *   Pre-generate p.ransacIters samples of 4 distinct match indices using a fixed
 *   PRNG seed (p.ransacSeed) so all versions test the exact same hypotheses.
 * 
 * Inputs:
 *   - numMatches: Number of available matches from Stage 4.
 *   - p: Parameters struct (p.ransacIters, p.ransacSeed).
 * 
 * Outputs:
 *   - samples: Output vector of size 4 * p.ransacIters containing match indices.
 */
void generateRansacSamples(int numMatches, const Params& p, std::vector<int>& samples) {
    samples.clear();

    if (numMatches < 4 || p.ransacIters <= 0) {
        return;
    }

    samples.resize(static_cast<size_t>(p.ransacIters) * 4);
    uint32_t state = p.ransacSeed;

    for (int iter = 0; iter < p.ransacIters; ++iter) {
        int* quad = &samples[static_cast<size_t>(iter) * 4];

        for (int k = 0; k < 4; ++k) {
            int candidate = -1;
            bool duplicate = true;

            while (duplicate) {
                candidate = static_cast<int>(lcgNextRansac(state) % static_cast<uint32_t>(numMatches));
                duplicate = false;
                for (int j = 0; j < k; ++j) {
                    if (quad[j] == candidate) {
                        duplicate = true;
                        break;
                    }
                }
            }
            quad[k] = candidate;
        }
    }
}

/*
 * General Purpose:
 *   Sequential CPU Baseline (V0) for homography estimation:
 *   Estimates H mapping B -> A via RANSAC with 4-point DLT, identifies inliers,
 *   and optionally refits H over all inliers.
 * 
 * Inputs:
 *   - kpsA, kpsB: Keypoints for image A and image B.
 *   - matches: Feature correspondences from Stage 4.
 *   - p: Pipeline parameters.
 * 
 * Outputs:
 *   - H: Estimated 3x3 Homography (B -> A).
 *   - inlierMask: Binary mask (1 = inlier, 0 = outlier).
 *   - numInliers: Number of inliers supporting H.
 *   - Returns true if numInliers >= p.minInliers, false otherwise.
 */
bool estimateHomography(const std::vector<Keypoint>& kpsA,
                        const std::vector<Keypoint>& kpsB,
                        const std::vector<Match>& matches, const Params& p,
                        Homography& H, std::vector<uint8_t>& inlierMask, int& numInliers) {
    const int numMatches = static_cast<int>(matches.size());
    inlierMask.assign(numMatches, 0);
    numInliers = 0;

    if (numMatches < 4 || numMatches < p.minInliers) {
        return false;
    }

    std::vector<PointPair2D> allPairs(numMatches);
    for (int i = 0; i < numMatches; ++i) {
        const Keypoint& kpA = kpsA[matches[i].idxA];
        const Keypoint& kpB = kpsB[matches[i].idxB];
        allPairs[i] = PointPair2D{
            static_cast<double>(kpB.x), static_cast<double>(kpB.y),
            static_cast<double>(kpA.x), static_cast<double>(kpA.y)
        };
    }

    std::vector<int> samples;
    generateRansacSamples(numMatches, p, samples);

    const double threshSq = p.ransacThresh * p.ransacThresh;
    int bestInliers = -1;
    Homography bestH;

    // =========================================================================
    // Step 1: RANSAC Loop (4-point DLT + Reprojection Inlier Counting)
    // =========================================================================
    for (int iter = 0; iter < p.ransacIters; ++iter) {
        const int* quadIdx = &samples[static_cast<size_t>(iter) * 4];
        PointPair2D quad[4] = {
            allPairs[quadIdx[0]],
            allPairs[quadIdx[1]],
            allPairs[quadIdx[2]],
            allPairs[quadIdx[3]]
        };

        if (!isQuadNonCollinear(quad)) {
            continue;
        }

        Homography candH;
        if (!solveNormalizedDLT(quad, 4, candH)) {
            continue;
        }

        int count = 0;
        for (int i = 0; i < numMatches; ++i) {
            if (isMatchInlier(candH, allPairs[i], threshSq)) {
                ++count;
            }
        }

        // Tie on inlier count -> lowest iteration index wins
        if (count > bestInliers) {
            bestInliers = count;
            bestH       = candH;
        }
    }

    if (bestInliers < p.minInliers) {
        return false;
    }

    // =========================================================================
    // Step 2: Extract Final Inlier Mask from Best Hypothesis
    // =========================================================================
    std::vector<PointPair2D> inlierPairs;
    inlierPairs.reserve(bestInliers);
    numInliers = 0;

    for (int i = 0; i < numMatches; ++i) {
        if (isMatchInlier(bestH, allPairs[i], threshSq)) {
            inlierMask[i] = 1;
            ++numInliers;
            inlierPairs.push_back(allPairs[i]);
        }
    }

    H = bestH;

    if (numInliers < p.minInliers) {
        return false;
    }

    // =========================================================================
    // Step 3: Optional Least-Squares DLT Refit on All Inliers
    // =========================================================================
    if (p.refitInliers && static_cast<int>(inlierPairs.size()) >= 4) {
        Homography refitH;
        if (solveNormalizedDLT(inlierPairs.data(), static_cast<int>(inlierPairs.size()), refitH)) {
            H = refitH;
        }
    }

    return true;
}

} // namespace v0
