/*
 * File: stages.cuh
 * Authors: Nguyen Hoang Minh Tam, Nguyen Manh Cuong, Phan Nhat Thien Nhan
 * External sources: none
 *
 * Rules:
 *   - Each stage lives in its own .cu file and may be developed/tested alone.
 *   - V0 = namespace v0, plain single-thread C++ (no OpenMP / std::thread / SIMD / CUDA).
 *   - Later versions (v1, v2, ...) get their own namespaces with the same stage split.
 *   - All parameters come from Params; never hard-code a constant inside a stage.
 *   - Stages must be deterministic: no time-based seeds, no std::*_distribution
 */
#pragma once
#include "types.cuh"

namespace v0 {

// ---- Stage 1: preprocess.cu
// RGB -> grayscale
// (0.299R+0.587G+0.114B) -> separable Gaussian blur. 
// Border handling: clamp-to-edge. Output has the same size as the input.
void preprocess(const Image &rgb, const Params &p, GrayImage &blurred);

// ---- Stage 2: keypoint.cu
// FAST-9 corners -> Harris score -> 3x3 NMS -> keep the top p.maxKeypoints by score. 
// Only keypoints at least p.borderMargin away from every border are returned. 
// Result is sorted by score descending, ties broken by (y, x) ascending (determinism). 
// Fills x, y, score. (angle is filled by stage 3.)
void detectKeypoints(const GrayImage &blurred, const Params &p, std::vector<Keypoint> &kps);

// ---- Stage 3: describe.cu
// Computes kps[i].angle (intensity centroid, radius p.patchRadius) 
// and a 256-bit rotated-BRIEF descriptor for each keypoint. 
// Keypoints that cannot be described are REMOVED from kps, so kps[i] <-> desc[i] always stay aligned.
void computeDescriptors(const GrayImage &blurred, const Params &p, std::vector<Keypoint> &kps, std::vector<Descriptor> &desc);

// The fixed 256-pair BRIEF pattern, generated once from p.briefSeed with an own PRNG. 
// Shared by every version (V1+ copies it to constant memory).
const BriefPair *briefPattern(const Params &p); // returns 256 pairs

// ---- Stage 4: matching.cu
// Brute-force Hamming nearest neighbour A->B, 
// Lowe ratio test (p.ratioThresh), optional cross-check (p.crossCheck). 
// Output sorted by idxA ascending.
void matchDescriptors(const std::vector<Descriptor> &descA, const std::vector<Descriptor> &descB, const Params &p, std::vector<Match> &matches);

// ---- Stage 5: homography.cu
// Pre-generates p.ransacIters samples of 4 distinct match indices (fixed seed), output size = 4*ransacIters. 
// Every version must call this, so all versions test the very same hypotheses.
void generateRansacSamples(int numMatches, const Params &p, std::vector<int> &samples);

// RANSAC with 4-point DLT. H maps B -> A. inlierMask[i] = 1 if matches[i] is an inlier. 
// If p.refitInliers, H is re-estimated from all inliers. Tie on inlier count -> lowest iteration index wins. 
// Returns false if fewer than p.minInliers inliers are found.
bool estimateHomography(const std::vector<Keypoint> &kpsA, const std::vector<Keypoint> &kpsB, const std::vector<Match> &matches, 
	const Params &p, Homography &H, std::vector<uint8_t> &inlierMask, int &numInliers);

// ---- Stage 6: warp.cu
// Canvas = bounding box of A and of the 4 corners of B warped by H. 
// Inverse warp (bilinear) of B, A copied in place, feathering in the overlap if p.feather. 
// Pixels covered by neither image are black.
bool warpAndBlend(const Image &A, const Image &B, const Homography &H_BtoA, const Params &p, Image &panorama);

} // namespace v0