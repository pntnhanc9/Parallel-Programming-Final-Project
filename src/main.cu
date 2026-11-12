/*
 * File: main.cu
 * Authors: Nguyen Manh Cuong
 * External sources: none
 *
 * Entry point: parse CLI, load the two images, run the selected pipeline
 * version (1 warm-up + N timed runs), print per-stage and end-to-end timing,
 * and save the panorama.
 *
 * Command line: ./stitch <left_image> <right_image> <output_image> --version 0 [--runs N]
 */
#include "stages.cuh"
#include "utils.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static std::string deviceName(int version) {
	switch (version) {
	case 0:
		return "CPU (single thread)";
	default:
		return "unknown";
	}
}

// ------------------------------------------------------------------------ V0
// Stitches A (left / reference) and B (right). Per-stage times go into T.
static bool runV0(const Image &A, const Image &B, const Params &P, Image &out, StageTimes &T, PairInfo &info) {
	CpuTimer t;

	GrayImage gA, gB;
	t.start();
	v0::preprocess(A, P, gA);
	v0::preprocess(B, P, gB);
	T.ms[ST_PREPROCESS] = t.stopMs();

	std::vector<Keypoint> kA, kB;
	t.start();
	v0::detectKeypoints(gA, P, kA);
	v0::detectKeypoints(gB, P, kB);
	T.ms[ST_DETECT] = t.stopMs();

	std::vector<Descriptor> dA, dB;
	t.start();
	v0::computeDescriptors(gA, P, kA, dA);
	v0::computeDescriptors(gB, P, kB, dB);
	T.ms[ST_DESCRIBE] = t.stopMs();
	info.numKeypointsA = (int)kA.size();
	info.numKeypointsB = (int)kB.size();

	std::vector<Match> matches;
	t.start();
	v0::matchDescriptors(dA, dB, P, matches);
	T.ms[ST_MATCH] = t.stopMs();
	info.numMatches = (int)matches.size();

	std::vector<uint8_t> inliers;
	t.start();
	const bool okH = v0::estimateHomography(kA, kB, matches, P, info.H, inliers, info.numInliers);
	T.ms[ST_RANSAC] = t.stopMs();
	if (!okH) {
		fprintf(stderr, "Homography estimation failed (%d matches, %d inliers)\n", info.numMatches, info.numInliers);
		return false;
	}

	t.start();
	const bool okW = v0::warpAndBlend(A, B, info.H, P, out);
	T.ms[ST_WARP] = t.stopMs();
	return okW;
}

// Runs one full pipeline of the selected version and fills T / info.
static bool runPipeline(int version, const Image &A, const Image &B, const Params &P, Image &pano, StageTimes &T, PairInfo &info) {
	T = StageTimes();
	info = PairInfo();
	CpuTimer total;
	total.start();
	bool ok = false;
	switch (version) {
	case 0:
		ok = runV0(A, B, P, pano, T, info);
		break;
	default:
		fprintf(stderr, "Version %d is not implemented\n", version);
		return false;
	}
	T.endToEndMs = total.stopMs();
	return ok;
}

// ---------------------------------------------------------------------- report
static void reportRow(const char *name, const std::vector<double> &v) {
	double sum = 0, mn = 1e300, mx = 0;
	for (double x : v) {
		sum += x;
		mn = std::min(mn, x);
		mx = std::max(mx, x);
	}
	const double mean = sum / v.size();
	double var = 0;
	for (double x : v)
		var += (x - mean) * (x - mean);
	const double sd = std::sqrt(var / v.size());
	printf("%-14s %11.3f %11.3f %11.3f %11.3f\n", name, mean, mn, mx, sd);
}

int main(int argc, char **argv) {
	CliOptions opt;
	std::string err;
	if (!parseArgs(argc, argv, opt, err)) {
		if (!err.empty())
			fprintf(stderr, "Error: %s\n\n", err.c_str());
		printUsage(argv[0]);
		return err.empty() ? 0 : 1;
	}

	// ---- load (excluded from timing)
	Image A, B;
	if (!loadImage(opt.leftPath, A) || !loadImage(opt.rightPath, B))
			return 1;
	printf("Loaded A: %s (%dx%d)\n", opt.leftPath.c_str(), A.width, A.height);
	printf("Loaded B: %s (%dx%d)\n", opt.rightPath.c_str(), B.width, B.height);

	const Params P;
	Image pano;
	StageTimes T;
	PairInfo info;

	// ---- warm-up (not timed), then N timed runs
	if (!runPipeline(opt.version, A, B, P, pano, T, info))
		return 2;
	std::vector<StageTimes> runs;
	for (int r = 0; r < opt.runs; ++r) {
		if (!runPipeline(opt.version, A, B, P, pano, T, info))
			return 2;
		runs.push_back(T);
	}

	// ---- report
	printf("\nVersion %d | device: %s | image %dx%d | %d timed runs (+1 warm-up)\n",
				 opt.version, deviceName(opt.version).c_str(), A.width, A.height,
				 opt.runs);
	printf("%-14s %11s %11s %11s %11s\n", "Stage", "avg (ms)", "min (ms)",
				 "max (ms)", "std (ms)");
	std::vector<double> v(runs.size());
	for (int s = 0; s < ST_COUNT; ++s) {
		for (size_t r = 0; r < runs.size(); ++r)
			v[r] = runs[r].ms[s];
		reportRow(kStageNames[s], v);
	}
	for (size_t r = 0; r < runs.size(); ++r)
		v[r] = runs[r].stageSum();
	reportRow("stage sum", v);
	for (size_t r = 0; r < runs.size(); ++r)
		v[r] = runs[r].endToEndMs;
	reportRow("end-to-end", v);
	printf("host-device transfer: n/a (V0 runs entirely on the CPU)\n");

	printf("\nKeypoints A/B = %d/%d, matches = %d, inliers = %d\n",
				 info.numKeypointsA, info.numKeypointsB, info.numMatches, info.numInliers);
	printf("H (B->A) = [% .6f % .6f % .6f; % .6f % .6f % .6f; % .6f % .6f "
				 "% .6f]\n",
				 info.H.h[0], info.H.h[1], info.H.h[2], 
				 info.H.h[3], info.H.h[4], info.H.h[5], 
				 info.H.h[6], info.H.h[7], info.H.h[8]);

	if (!saveImage(opt.outputPath, pano))
		return 3;
	printf("\nPanorama %dx%d written to %s\n", pano.width, pano.height, opt.outputPath.c_str());
	return 0;
}
