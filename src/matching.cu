/*
 * File: matching.cu
 * Author: Phan Nhat Thien Nhan (Phase 4)
 * 
 * General Purpose:
 *   Brute-force binary descriptor matching between image A and image B:
 *   1. Compute the 256-bit Hamming distance between descriptor pairs from A and B.
 *   2. Find the 1st and 2nd nearest neighbors in B for each descriptor in A.
 *   3. Filter ambiguous matches using Lowe's ratio test (p.ratioThresh).
 *   4. Optionally enforce mutual nearest-neighbor cross-check (p.crossCheck).
 *   5. Output verified matches sorted by idxA ascending.
 * 
 * Inputs:
 *   - descA: Vector of 256-bit descriptors from image A.
 *   - descB: Vector of 256-bit descriptors from image B.
 *   - p: Parameters struct (p.ratioThresh = 0.75f, p.crossCheck = true).
 * 
 * Outputs:
 *   - matches: Output vector of Match structs (idxA, idxB, dist) sorted by idxA ascending.
 */

#include "stages.cuh"
#include "types.cuh"
#include <cstdint>
#include <limits>
#include <vector>

/*
 * General Purpose:
 *   Count the number of set bits (1s) in a 32-bit unsigned integer.
 * 
 * Inputs:
 *   - x: 32-bit unsigned integer.
 * 
 * Outputs:
 *   - Number of bits set to 1 in [0, 32].
 */
static inline int popcount32(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(x);
#else
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0F0F0F0Fu;
    return static_cast<int>((x * 0x01010101u) >> 24);
#endif
}

/*
 * General Purpose:
 *   Compute the 256-bit Hamming distance between two binary descriptors.
 * 
 * Inputs:
 *   - a: First 256-bit Descriptor (8 x uint32_t).
 *   - b: Second 256-bit Descriptor (8 x uint32_t).
 * 
 * Outputs:
 *   - Total Hamming distance in [0, 256].
 */
static inline int descriptorHammingDist(const Descriptor& a, const Descriptor& b) {
    int dist = 0;
    for (int w = 0; w < 8; ++w) {
        dist += popcount32(a.w[w] ^ b.w[w]);
    }
    return dist;
}

namespace v0 {

/*
 * General Purpose:
 *   Sequential CPU Baseline (V0) for feature descriptor matching:
 *   Finds nearest-neighbor matches from A to B using Hamming distance, applies
 *   Lowe's ratio test and optional cross-checking, and outputs matches ordered
 *   by idxA ascending.
 * 
 * Inputs:
 *   - descA: Descriptors from image A.
 *   - descB: Descriptors from image B.
 *   - p: Pipeline parameters (ratioThresh, crossCheck).
 * 
 * Outputs:
 *   - matches: Verified feature matches sorted by idxA ascending.
 */
void matchDescriptors(const std::vector<Descriptor>& descA,
                      const std::vector<Descriptor>& descB,
                      const Params& p, std::vector<Match>& matches) {
    matches.clear();

    const int numA = static_cast<int>(descA.size());
    const int numB = static_cast<int>(descB.size());

    if (numA == 0 || numB < 2) {
        return;
    }

    // =========================================================================
    // Step 1: Brute-Force Hamming Search (A -> B) & Reverse Tracking (B -> A)
    // =========================================================================
    std::vector<int> bestDistBA;
    std::vector<int> bestIdxAForB;
    if (p.crossCheck) {
        bestDistBA.assign(numB, std::numeric_limits<int>::max());
        bestIdxAForB.assign(numB, -1);
    }

    std::vector<Match> candidates;
    candidates.reserve(numA);

    for (int i = 0; i < numA; ++i) {
        int bestDist   = std::numeric_limits<int>::max();
        int secondDist = std::numeric_limits<int>::max();
        int bestIdxB   = -1;

        for (int j = 0; j < numB; ++j) {
            int d = descriptorHammingDist(descA[i], descB[j]);

            // Track 1st and 2nd nearest neighbors in B for A[i]
            if (d < bestDist) {
                secondDist = bestDist;
                bestDist   = d;
                bestIdxB   = j;
            } else if (d < secondDist) {
                secondDist = d;
            }

            // Simultaneously track nearest neighbor in A for B[j] (for cross-check)
            if (p.crossCheck && d < bestDistBA[j]) {
                bestDistBA[j]   = d;
                bestIdxAForB[j] = i;
            }
        }

        // Lowe's ratio test: keep only distinctive matches
        if (bestIdxB >= 0 &&
            static_cast<float>(bestDist) < p.ratioThresh * static_cast<float>(secondDist)) {
            Match m;
            m.idxA = i;
            m.idxB = bestIdxB;
            m.dist = bestDist;
            candidates.push_back(m);
        }
    }

    // =========================================================================
    // Step 2: Optional Mutual Nearest-Neighbor Cross-Check Filter
    // =========================================================================
    if (!p.crossCheck) {
        matches = std::move(candidates);
        return;
    }

    matches.reserve(candidates.size());
    for (size_t k = 0; k < candidates.size(); ++k) {
        const Match& m = candidates[k];
        if (bestIdxAForB[m.idxB] == m.idxA) {
            matches.push_back(m);
        }
    }
}

} // namespace v0
