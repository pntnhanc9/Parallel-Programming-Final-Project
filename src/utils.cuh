/*
 * File: utils.cuh
 * Authors: Nguyen Hoang Minh Tam
 * External sources: stb_image.h / stb_image_write.h (public domain, Sean Barrett, https://github.com/nothings/stb).
 *
 * CLI parsing, image I/O, CPU timer, and small helper functions.
 */
#pragma once
#include "types.cuh"
#include <chrono>
#include <string>
#include <vector>

// ---------------------------------------------------------------- command line
// Command line usage:
//   ./stitch <left_image> <right_image> <output_image> --version <N> [--runs N]
//
// <left_image> is image A (the reference);
// <right_image> is image B (warped onto A's frame);
// --version defaults to 0; 
// --runs defaults to 5.

// Highest pipeline version that is implemented. parseArgs rejects anything above it.
constexpr int kMaxVersion = 0;

struct CliOptions {
	std::string leftPath;   // image A
	std::string rightPath;  // image B
	std::string outputPath;
	int version = 0;
	int runs = 5; // timed runs (a warm-up run is always done first)
};

// Returns false on error (message in `err`).
bool parseArgs(int argc, char **argv, CliOptions &opt, std::string &err);
void printUsage(const char *exe);

// ---------------------------------------------------------------- image I/O

// Both use stb (implementation compiled once in stb_impl.cpp).
bool loadImage(const std::string &path, Image &img); // any stb format -> RGB8
bool saveImage(const std::string &path, const Image &img); // by extension: .png/.jpg/.jpeg/.bmp (default png)

// ---------------------------------------------------------------- timing

// V0 timer: std::chrono::high_resolution_clock
struct CpuTimer {
	std::chrono::high_resolution_clock::time_point t0;
	void start() { t0 = std::chrono::high_resolution_clock::now(); }
	double stopMs() const {
			return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
	}
};

// ---------------------------------------------------------------- evaluation

void applyHomography(const Homography &H, double x, double y, double &ox, double &oy);
// Mean distance (pixels) between the 4 image corners of a w x h image mapped by
// `est` and by `gt`. Used for the synthetic-pair correctness check.
double homographyCornerError(const Homography &est, const Homography &gt, int w, int h);

// ---------------------------------------------------------------- small helpers

// Clamp an integer to [lo, hi].
int clampInt(int val, int lo, int hi);
