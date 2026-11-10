// ultis.h -- CLI parsing, image I/O, CPU timer, small evaluation helpers.
// Authors: <TODO: names / student IDs>
// External sources: stb_image.h / stb_image_write.h (public domain, Sean Barrett,
//                   https://github.com/nothings/stb) for decoding/encoding images.
#pragma once
#include <chrono>
#include <string>
#include <vector>
#include "types.h"

// ---------------------------------------------------------------- command line
// Supported forms:
//   ./stitch -F <input_dir>                 -o <output_image> [--version N] [--runs N]
//   ./stitch -f <img1> <img2> [<img3> ...]  -o <output_image> [--version N] [--runs N]
//   ./stitch <left> <right> <output_image>  [--version N] [--runs N]     (course-spec form)
// Images are stitched left to right, in the order given (or in natural filename
// order for -F).
struct CliOptions {
    std::vector<std::string> inputs;     // ordered left -> right, >= 2 entries
    std::string outputPath;
    int version = 0;
    int runs    = 5;                     // timed runs (a warm-up run is always done first)
};

// Returns false on error (message in `err`; empty err + false means "print usage only").
bool parseArgs(int argc, char** argv, CliOptions& opt, std::string& err);
void printUsage(const char* exe);

// ------------------------------------------------------------------- image I/O
bool loadImage(const std::string& path, Image& img);        // any format stb supports -> RGB8
bool saveImage(const std::string& path, const Image& img);  // by extension: .png/.jpg/.jpeg/.bmp (default png)
// All .jpg/.jpeg/.png/.bmp files of `dir`, natural-sorted (img2 < img10).
bool listImagesInDir(const std::string& dir, std::vector<std::string>& paths);

// ---------------------------------------------------------------------- timing
// V0 timer: std::chrono::high_resolution_clock, as required by the project spec.
struct CpuTimer {
    std::chrono::high_resolution_clock::time_point t0;
    void   start() { t0 = std::chrono::high_resolution_clock::now(); }
    double stopMs() const {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::high_resolution_clock::now() - t0).count();
    }
};

// ------------------------------------------------------- evaluation (midterm 3.5)
void applyHomography(const Homography& H, double x, double y, double& ox, double& oy);
// Mean distance (pixels) between the 4 image corners of a w x h image mapped by
// `est` and by `gt`. Used for the synthetic-pair correctness check.
double homographyCornerError(const Homography& est, const Homography& gt, int w, int h);