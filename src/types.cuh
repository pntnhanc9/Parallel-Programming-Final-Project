/*
 * File: types.cuh
 * Authors: Nguyen Hoang Minh Tam, Nguyen Manh Cuong, Phan Nhat Thien Nhan
 * External sources: none
 *
 * Shared data types and pipeline parameters used by ALL versions (V0, V1, ...).
 * Plain C++ only -- no CUDA includes -- so V0 stays a pure CPU baseline.
 * Changing a struct here affects every module: discuss with the team first.
 */
#pragma once
#include <cstdint>
#include <vector>
#include <string>

// 8-bit RGB image, interleaved (R,G,B,R,G,B,...), row-major, always 3 channels.
struct Image {
		int width = 0, height = 0;
		std::vector<uint8_t> data;                       // size = width*height*3
		bool empty() const { return data.empty(); }
};

// Single-channel float image, values in [0,255], row-major.
struct GrayImage {
		int width = 0, height = 0;
		std::vector<float> data;                         // size = width*height
		bool empty() const { return data.empty(); }
};

// Position is in pixel coordinates of the image the keypoint was detected in.
struct Keypoint {
		float x = 0, y = 0;
		float score = 0;                                 // Harris response (used for top-K)
		float angle = 0;                                 // radians, intensity-centroid orientation
};

// 256-bit binary descriptor (rBRIEF). Hamming distance = sum of popcount(a.w[i]^b.w[i]).
struct Descriptor { uint32_t w[8] = {0}; };

// One pair of sample points of the BRIEF pattern, offsets relative to the keypoint.
struct BriefPair { int8_t x1, y1, x2, y2; };

// A match between keypoint idxA of image A and keypoint idxB of image B.
// idxA/idxB index the *final* keypoint/descriptor arrays (after border filtering).
struct Match { int idxA = -1, idxB = -1; int dist = 0; };

// 3x3 projective transform, row-major, maps points of image B into image A's frame:
//   [x' y' w]^T = H [x y 1]^T ,   (x'/w , y'/w) is the position in A.
struct Homography {
		double h[9] = {1,0,0, 0,1,0, 0,0,1};
};

// Every version must use exactly these parameters (fair timing + comparable output).
struct Params {
		// Stage 1: grayscale + separable Gaussian
		float blurSigma     = 2.0f;
		int   blurRadius    = 3;                         // 7 taps
		// Stage 2: FAST-9 + Harris score + NMS + top-K
		int   fastThreshold = 20;
		int   fastArc       = 9;
		int   nmsRadius     = 1;                         // 3x3 non-maximum suppression
		int   maxKeypoints  = 3000;                      // top-K by Harris score
		int   borderMargin  = 20;                        // keep keypoints this far from the border
		// Stage 3: oriented BRIEF
		int   patchRadius   = 15;                        // radius for intensity-centroid angle
		int   briefSeed     = 2026;                      // seed of the fixed 256-pair pattern
		// Stage 4: brute-force Hamming matching
		float ratioThresh   = 0.75f;                     // Lowe ratio test
		bool  crossCheck    = true;
		// Stage 5: RANSAC + 4-point DLT
		int   ransacIters   = 2000;
		double ransacThresh = 3.0;                       // pixels (reprojection error in A)
		uint32_t ransacSeed = 1234;                      // samples are pre-generated -> identical in all versions
		int   minInliers    = 10;
		bool  refitInliers  = true;                      // final least-squares DLT on all inliers
		// Stage 6: warp + blend
		bool  feather       = true;                      // linear feathering in the overlap
};

// Time (ms) of each stage. Stages 1-3 are summed over both images.
enum Stage { ST_PREPROCESS, ST_DETECT, ST_DESCRIBE, ST_MATCH, ST_RANSAC, ST_WARP, ST_COUNT };
static const char* const kStageNames[ST_COUNT] = {
		"1 preprocess", "2 keypoints", "3 describe", "4 match", "5 ransac", "6 warp+blend"};

struct StageTimes {
		double ms[ST_COUNT] = {0};
		double endToEndMs = 0;                           // whole pipeline, image I/O excluded
		double stageSum() const { double s = 0; for (int i = 0; i < ST_COUNT; ++i) s += ms[i]; return s; }
};

// Diagnostics of one pair stitch (for the report: "every number comes from an output cell").
struct PairInfo {
		int numKeypointsA = 0, numKeypointsB = 0;
		int numMatches = 0, numInliers = 0;
		Homography H;
};