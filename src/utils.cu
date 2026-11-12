// utils.cu -- CLI parsing, image I/O, evaluation helpers (pure C++, no CUDA
// calls).
// Authors: <TODO: names / student IDs>
// External sources: stb_image.h / stb_image_write.h (public domain, Sean
//                   Barrett, https://github.com/nothings/stb).
#include "utils.cuh"
#include "stb_image.h" // declarations only; implementation is in stb_impl.cpp
#include "stb_image_write.h"
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;

// ----------------------------------------------------------------- CLI parsing
void printUsage(const char *exe) {
  fprintf(stderr,
          "Usage:\n"
          "  %s <left_image> <right_image> <output_image> --version <N> "
          "[--runs N]\n"
          "\n"
          "  <left_image>    image A, the reference frame\n"
          "  <right_image>   image B, warped onto A's frame\n"
          "  <output_image>  panorama (.png / .jpg / .bmp, default png)\n"
          "  --version N     pipeline version, 0..%d (default 0 = sequential "
          "CPU)\n"
          "  --runs N        timed runs after one warm-up run (default 5)\n",
          exe, kMaxVersion);
}

// Strict integer parse: the whole string must be a valid int.
static bool parseInt(const char *s, int &out) {
  char *end = nullptr;
  errno = 0;
  const long v = std::strtol(s, &end, 10);
  if (end == s || *end != '\0' || errno == ERANGE || v < INT_MIN || v > INT_MAX)
    return false;
  out = (int)v;
  return true;
}

// An option looks like "-x" / "--xyz"; a lone "-" is treated as a positional.
static bool isFlag(const char *s) { return s[0] == '-' && s[1] != '\0'; }

bool parseArgs(int argc, char **argv, CliOptions &opt, std::string &err) {
  std::vector<std::string> positional;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      err.clear();
      return false;
    } else if (a == "--version") {
      if (i + 1 >= argc || !parseInt(argv[i + 1], opt.version)) {
        err = "--version needs an integer value";
        return false;
      }
      ++i;
    } else if (a == "--runs") {
      if (i + 1 >= argc || !parseInt(argv[i + 1], opt.runs)) {
        err = "--runs needs an integer value";
        return false;
      }
      ++i;
    } else if (isFlag(argv[i])) {
      err = "unknown option '" + a + "'";
      return false;
    } else {
      positional.push_back(a);
    }
  }

  if (positional.size() != 3) {
    err = "expected 3 positional arguments <left_image> <right_image> "
          "<output_image>, got " +
          std::to_string(positional.size());
    return false;
  }
  opt.leftPath = positional[0];
  opt.rightPath = positional[1];
  opt.outputPath = positional[2];

  if (opt.version < 0 || opt.version > kMaxVersion) {
    err = "--version " + std::to_string(opt.version) +
          " is not available (implemented: 0.." + std::to_string(kMaxVersion) +
          ")";
    return false;
  }
  if (opt.runs < 1) {
    err = "--runs must be >= 1";
    return false;
  }
  return true;
}

// ------------------------------------------------------------------- image I/O
static std::string lowerExt(const std::string &path) {
  std::string e = fs::path(path).extension().string();
  std::transform(e.begin(), e.end(), e.begin(),
                 [](unsigned char c) { return (char)std::tolower(c); });
  return e;
}

bool loadImage(const std::string &path, Image &img) {
  int w = 0, h = 0, c = 0;
  unsigned char *px = stbi_load(path.c_str(), &w, &h, &c, 3); // force RGB
  if (!px) {
    fprintf(stderr, "Cannot load '%s': %s\n", path.c_str(),
            stbi_failure_reason());
    return false;
  }
  img.width = w;
  img.height = h;
  img.data.assign(px, px + (size_t)w * h * 3);
  stbi_image_free(px);
  return true;
}

bool saveImage(const std::string &path, const Image &img) {
  if (img.empty()) {
    fprintf(stderr, "saveImage: empty image\n");
    return false;
  }
  const fs::path parent = fs::path(path).parent_path();
  if (!parent.empty()) {
    std::error_code ec;
    fs::create_directories(parent, ec);
  }

  const std::string ext = lowerExt(path);
  int ok = 0;
  if (ext == ".jpg" || ext == ".jpeg")
    ok = stbi_write_jpg(path.c_str(), img.width, img.height, 3, img.data.data(),
                        95);
  else if (ext == ".bmp")
    ok = stbi_write_bmp(path.c_str(), img.width, img.height, 3, img.data.data());
  else
    ok = stbi_write_png(path.c_str(), img.width, img.height, 3, img.data.data(),
                        img.width * 3);
  if (!ok)
    fprintf(stderr, "Cannot write '%s'\n", path.c_str());
  return ok != 0;
}

// ------------------------------------------------------------------ evaluation
void applyHomography(const Homography &H, double x, double y, double &ox,
                     double &oy) {
  const double *h = H.h;
  const double w = h[6] * x + h[7] * y + h[8];
  ox = (h[0] * x + h[1] * y + h[2]) / w;
  oy = (h[3] * x + h[4] * y + h[5]) / w;
}

double homographyCornerError(const Homography &est, const Homography &gt, int w,
                             int h) {
  const double cx[4] = {0, (double)w, (double)w, 0};
  const double cy[4] = {0, 0, (double)h, (double)h};
  double sum = 0;
  for (int i = 0; i < 4; ++i) {
    double ex, ey, gx, gy;
    applyHomography(est, cx[i], cy[i], ex, ey);
    applyHomography(gt, cx[i], cy[i], gx, gy);
    sum += std::hypot(ex - gx, ey - gy);
  }
  return sum / 4.0;
}

// --------------------------------------------------------------------- helpers
int clampInt(int val, int lo, int hi) {
  if (val < lo)
    return lo;
  if (val > hi)
    return hi;
  return val;
}
