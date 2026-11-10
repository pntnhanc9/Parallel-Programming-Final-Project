// main.cu -- entry point: parse CLI, load images, run the selected pipeline version
//            (1 warm-up + N timed runs), print per-stage timing, save the panorama.
// Authors: <TODO: names / student IDs>
// External sources: none
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>
#include "stages.h"
#include "ultis.h"

// Stitches A (left/reference) and B (right). Per-stage times are ADDED into T.
static bool stitchPair(int version, const Image& A, const Image& B, const Params& P,
                       Image& out, StageTimes& T, PairInfo& info) {
    if (version != 0) { fprintf(stderr, "Version %d is not implemented yet\n", version); return false; }
    CpuTimer t;

    GrayImage gA, gB;
    t.start(); v0::preprocess(A, P, gA); v0::preprocess(B, P, gB); T.ms[ST_PREPROCESS] += t.stopMs();

    std::vector<Keypoint> kA, kB;
    t.start(); v0::detectKeypoints(gA, P, kA); v0::detectKeypoints(gB, P, kB); T.ms[ST_DETECT] += t.stopMs();

    std::vector<Descriptor> dA, dB;
    t.start(); v0::computeDescriptors(gA, P, kA, dA); v0::computeDescriptors(gB, P, kB, dB); T.ms[ST_DESCRIBE] += t.stopMs();
    info.numKeypointsA = (int)kA.size(); info.numKeypointsB = (int)kB.size();

    std::vector<Match> matches;
    t.start(); v0::matchDescriptors(dA, dB, P, matches); T.ms[ST_MATCH] += t.stopMs();
    info.numMatches = (int)matches.size();

    std::vector<uint8_t> inliers;
    bool ok;
    t.start(); ok = v0::estimateHomography(kA, kB, matches, P, info.H, inliers, info.numInliers); T.ms[ST_RANSAC] += t.stopMs();
    if (!ok) { fprintf(stderr, "Homography estimation failed (%d matches, %d inliers)\n", info.numMatches, info.numInliers); return false; }

    t.start(); ok = v0::warpAndBlend(A, B, info.H, P, out); T.ms[ST_WARP] += t.stopMs();
    return ok;
}

// N images: fold left to right, panorama_k = stitch(panorama_{k-1}, image_k).
static bool runPipeline(int version, const std::vector<Image>& imgs, const Params& P,
                        Image& pano, StageTimes& T, std::vector<PairInfo>& infos) {
    T = StageTimes(); infos.clear();
    CpuTimer total; total.start();
    Image cur = imgs[0];
    for (size_t i = 1; i < imgs.size(); ++i) {
        Image next; PairInfo info;
        if (!stitchPair(version, cur, imgs[i], P, next, T, info)) return false;
        infos.push_back(info);
        cur = std::move(next);
    }
    pano = std::move(cur);
    T.endToEndMs = total.stopMs();
    return true;
}

int main(int argc, char** argv) {
    CliOptions opt; std::string err;
    if (!parseArgs(argc, argv, opt, err)) {
        if (!err.empty()) fprintf(stderr, "Error: %s\n\n", err.c_str());
        printUsage(argv[0]);
        return err.empty() ? 0 : 1;
    }

    // ---- load (excluded from timing)
    std::vector<Image> imgs(opt.inputs.size());
    for (size_t i = 0; i < imgs.size(); ++i) {
        if (!loadImage(opt.inputs[i], imgs[i])) return 1;
        printf("Loaded [%zu] %s (%dx%d)\n", i, opt.inputs[i].c_str(), imgs[i].width, imgs[i].height);
    }

    const Params P;
    Image pano; StageTimes T; std::vector<PairInfo> infos;

    // ---- warm-up, then timed runs
    if (!runPipeline(opt.version, imgs, P, pano, T, infos)) return 2;
    std::vector<StageTimes> runs;
    for (int r = 0; r < opt.runs; ++r) {
        if (!runPipeline(opt.version, imgs, P, pano, T, infos)) return 2;
        runs.push_back(T);
    }

    // ---- report
    printf("\nVersion %d | device: CPU (single thread) | %zu images | %d timed runs (+1 warm-up)\n",
           opt.version, imgs.size(), opt.runs);
    printf("%-14s %12s %12s\n", "Stage", "avg (ms)", "min (ms)");
    auto report = [&](const char* name, auto get) {
        double sum = 0, mn = 1e300;
        for (const auto& s : runs) { double v = get(s); sum += v; mn = std::min(mn, v); }
        printf("%-14s %12.3f %12.3f\n", name, sum / runs.size(), mn);
    };
    for (int s = 0; s < ST_COUNT; ++s)
        report(kStageNames[s], [s](const StageTimes& t) { return t.ms[s]; });
    report("end-to-end", [](const StageTimes& t) { return t.endToEndMs; });

    for (size_t i = 0; i < infos.size(); ++i) {
        const PairInfo& in = infos[i];
        printf("\nPair %zu: keypoints A/B = %d/%d, matches = %d, inliers = %d\n", i,
               in.numKeypointsA, in.numKeypointsB, in.numMatches, in.numInliers);
        printf("H (B->A) = [% .6f % .6f % .6f; % .6f % .6f % .6f; % .6f % .6f % .6f]\n",
               in.H.h[0], in.H.h[1], in.H.h[2], in.H.h[3], in.H.h[4], in.H.h[5], in.H.h[6], in.H.h[7], in.H.h[8]);
    }

    if (!saveImage(opt.outputPath, pano)) return 3;
    printf("\nPanorama %dx%d written to %s\n", pano.width, pano.height, opt.outputPath.c_str());
    return 0;
}
