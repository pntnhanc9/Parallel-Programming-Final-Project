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

struct DevCliOptions: public CliOptions {
    int verbose = 0;
    int step = 0;                        // 0 = all, 1 = stage 1, 2 = stage 2, ...
}

// Returns false on error (message in `err`; empty err + false means "print usage only").
bool parseArgs(int argc, char** argv, CliOptions& opt, std::string& err);
void printUsage(const char* exe);

// ------------------------------------------------------------------- image I/O
bool loadImage(const std::string& path, Image& img);        // any format stb supports -> RGB8
bool saveImage(const std::string& path, const Image& img);  // by extension: .png/.jpg/.jpeg/.bmp (default png)

// ---------------------------------------------------------------------- ???
#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", error,\
				cudaGetErrorString(error));\
		exit(EXIT_FAILURE);\
	}\
}

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

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);
		cudaEventSynchronize(start);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

// ------------------------------------------------------- evaluation (midterm 3.5)
void applyHomography(const Homography& H, double x, double y, double& ox, double& oy);
// Mean distance (pixels) between the 4 image corners of a w x h image mapped by
// `est` and by `gt`. Used for the synthetic-pair correctness check.
double homographyCornerError(const Homography& est, const Homography& gt, int w, int h);