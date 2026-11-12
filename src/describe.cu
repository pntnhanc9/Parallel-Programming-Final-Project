/*
 * File: describe.cu
 * Author: Phan Nhat Thien Nhan (Phase 3)
 * 
 * General Purpose:
 *   Feature descriptor extraction for the image stitching pipeline:
 *   1. Generate a fixed, deterministic 256-pair BRIEF pattern using a custom PRNG.
 *   2. Compute the dominant orientation angle for each keypoint using the
 *      Intensity Centroid method (Rosin, 1999) over a circular patch.
 *   3. Extract a 256-bit Steered/Rotated BRIEF (rBRIEF) descriptor for each keypoint.
 *   4. Filter out any keypoints whose rotated sampling coordinates fall outside
 *      the image bounds, maintaining strict 1-to-1 alignment between kps[i] and desc[i].
 * 
 * Inputs:
 *   - blurred: Smoothed single-channel float GrayImage from Stage 1.
 *   - p: Parameters struct (p.patchRadius = 15, p.briefSeed = 2026).
 *   - kps: Vector of keypoints detected from Stage 2 (with x, y, score).
 * 
 * Outputs:
 *   - kps: Updated vector with kps[i].angle filled; invalid keypoints removed.
 *   - desc: Output vector of 256-bit descriptors (8 x uint32_t) aligned with kps.
 */

#include "stages.cuh"
#include "types.cuh"
#include <cmath>
#include <cstdint>
#include <vector>

// Number of sample pairs for a 256-bit BRIEF descriptor
constexpr int NUM_BRIEF_PAIRS = 256;

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
static inline uint32_t lcgNext(uint32_t& state) {
		// Numerical Recipes / 64-bit LCG constants
		state = state * 1664525u + 1013904223u;
		return state;
}

namespace v0 {

/*
 * General Purpose:
 *   Generate the fixed 256-pair BRIEF sampling pattern from p.briefSeed.
 *   The pattern is cached and generated only once (or regenerated if the seed changes).
 * 
 * Inputs:
 *   - p: Parameters struct containing p.briefSeed and p.patchRadius.
 * 
 * Outputs:
 *   - Pointer to an array of 256 BriefPair elements.
 */
const BriefPair* briefPattern(const Params& p) {
		static BriefPair pattern[NUM_BRIEF_PAIRS];
		static int cachedSeed = -1;

		if (cachedSeed == p.briefSeed) {
				return pattern;
		}

		uint32_t state = static_cast<uint32_t>(p.briefSeed);
		int radius = p.patchRadius;
		int diameter = 2 * radius + 1;

		for (int i = 0; i < NUM_BRIEF_PAIRS; ++i) {
				// Sample offsets uniformly in [-radius, radius]
				int8_t x1 = static_cast<int8_t>(static_cast<int>(lcgNext(state) % diameter) - radius);
				int8_t y1 = static_cast<int8_t>(static_cast<int>(lcgNext(state) % diameter) - radius);
				int8_t x2 = static_cast<int8_t>(static_cast<int>(lcgNext(state) % diameter) - radius);
				int8_t y2 = static_cast<int8_t>(static_cast<int>(lcgNext(state) % diameter) - radius);

				pattern[i] = BriefPair{x1, y1, x2, y2};
		}

		cachedSeed = p.briefSeed;
		return pattern;
}

/*
 * General Purpose:
 *   Compute keypoint orientation using the Intensity Centroid method over
 *   a circular patch of radius p.patchRadius.
 * 
 * Inputs:
 *   - blurred: Smoothed single-channel float image.
 *   - cx: Rounded integer X coordinate of keypoint center.
 *   - cy: Rounded integer Y coordinate of keypoint center.
 *   - radius: Patch radius (p.patchRadius).
 *   - angle: Output orientation angle in radians [-pi, pi].
 * 
 * Outputs:
 *   - true if the patch is valid and angle is computed, false if out-of-bounds.
 */
static bool computeKeypointAngle(const GrayImage& blurred, int cx, int cy, int radius, float& angle) {
		int width = blurred.width;
		int height = blurred.height;

		// Reject keypoints whose circular patch would exceed image borders
		if (cx - radius < 0 || cx + radius >= width ||
				cy - radius < 0 || cy + radius >= height) {
				return false;
		}

		float m10 = 0.0f; // Moment along X
		float m01 = 0.0f; // Moment along Y
		int radiusSq = radius * radius;

		for (int v = -radius; v <= radius; ++v) {
				int py = cy + v;
				int rowOffset = py * width;

				for (int u = -radius; u <= radius; ++u) {
						if (u * u + v * v <= radiusSq) {
								int px = cx + u;
								float intensity = blurred.data[rowOffset + px];

								m10 += static_cast<float>(u) * intensity;
								m01 += static_cast<float>(v) * intensity;
						}
				}
		}

		angle = std::atan2(m01, m10);
		return true;
}

/*
 * General Purpose:
 *   Extract a 256-bit rotated BRIEF descriptor for a keypoint given its angle.
 * 
 * Inputs:
 *   - blurred: Smoothed single-channel float image.
 *   - cx: Keypoint X coordinate.
 *   - cy: Keypoint Y coordinate.
 *   - angle: Keypoint orientation in radians.
 *   - pattern: Array of 256 canonical BRIEF sample pairs.
 *   - desc: Output 256-bit Descriptor struct.
 * 
 * Outputs:
 *   - true if all 256 rotated sample pairs fall inside the image, false otherwise.
 */
static bool extractRotatedBrief(const GrayImage& blurred, int cx, int cy, float angle,
															 const BriefPair* pattern, Descriptor& desc) {
		int width = blurred.width;
		int height = blurred.height;

		float cosAngle = std::cos(angle);
		float sinAngle = std::sin(angle);

		for (int w = 0; w < 8; ++w) {
				desc.w[w] = 0;
		}

		for (int i = 0; i < NUM_BRIEF_PAIRS; ++i) {
				const BriefPair& pair = pattern[i];

				// Rotate point 1 by keypoint angle
				float rx1 = static_cast<float>(pair.x1) * cosAngle - static_cast<float>(pair.y1) * sinAngle;
				float ry1 = static_cast<float>(pair.x1) * sinAngle + static_cast<float>(pair.y1) * cosAngle;
				int p1x = cx + static_cast<int>(std::round(rx1));
				int p1y = cy + static_cast<int>(std::round(ry1));

				// Rotate point 2 by keypoint angle
				float rx2 = static_cast<float>(pair.x2) * cosAngle - static_cast<float>(pair.y2) * sinAngle;
				float ry2 = static_cast<float>(pair.x2) * sinAngle + static_cast<float>(pair.y2) * cosAngle;
				int p2x = cx + static_cast<int>(std::round(rx2));
				int p2y = cy + static_cast<int>(std::round(ry2));

				// Boundary safety check for both sample points
				if (p1x < 0 || p1x >= width || p1y < 0 || p1y >= height ||
						p2x < 0 || p2x >= width || p2y < 0 || p2y >= height) {
						return false;
				}

				float val1 = blurred.data[p1y * width + p1x];
				float val2 = blurred.data[p2y * width + p2x];

				// Set bit i to 1 if intensity(p1) < intensity(p2)
				if (val1 < val2) {
						int wordIdx = i / 32;
						int bitIdx = i % 32;
						desc.w[wordIdx] |= (1u << bitIdx);
				}
		}

		return true;
}

/*
 * General Purpose:
 *   Sequential CPU Baseline (V0) for feature descriptor computation:
 *   Computes orientation angles and 256-bit rotated BRIEF descriptors for keypoints.
 *   Removes invalid keypoints from kps to preserve strict 1-to-1 alignment with desc.
 * 
 * Inputs:
 *   - blurred: Smoothed single-channel float image from Stage 1.
 *   - p: Parameters struct.
 *   - kps: Input keypoints vector from Stage 2.
 * 
 * Outputs:
 *   - kps: Filtered keypoints vector with angle computed.
 *   - desc: Output descriptors vector aligned with kps.
 */
void computeDescriptors(const GrayImage& blurred, const Params& p,
												std::vector<Keypoint>& kps, std::vector<Descriptor>& desc) {
		desc.clear();

		if (blurred.empty() || kps.empty()) {
				kps.clear();
				return;
		}

		const BriefPair* pattern = briefPattern(p);

		std::vector<Keypoint> validKps;
		std::vector<Descriptor> validDesc;
		validKps.reserve(kps.size());
		validDesc.reserve(kps.size());

		for (size_t i = 0; i < kps.size(); ++i) {
				Keypoint kp = kps[i];
				int cx = static_cast<int>(std::round(kp.x));
				int cy = static_cast<int>(std::round(kp.y));

				float angle = 0.0f;
				// Step 1: Compute dominant orientation
				if (!computeKeypointAngle(blurred, cx, cy, p.patchRadius, angle)) {
						continue; // Cannot compute angle safely near boundary
				}
				kp.angle = angle;

				// Step 2: Extract rotated BRIEF descriptor
				Descriptor d;
				if (!extractRotatedBrief(blurred, cx, cy, kp.angle, pattern, d)) {
						continue; // Rotated sample points fell outside the image
				}

				validKps.push_back(kp);
				validDesc.push_back(d);
		}

		// Keep kps and desc strictly aligned
		kps = std::move(validKps);
		desc = std::move(validDesc);
}

} // namespace v0
