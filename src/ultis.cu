// ultis.cu -- CLI parsing, image I/O, evaluation helpers (pure C++, no CUDA
// calls). Authors: <TODO: names / student IDs> External sources: stb_image.h /
// stb_image_write.h (public domain, Sean Barrett).
#include "stb_image.h" // declarations only; implementation is in stb_impl.cpp
#include "stb_image_write.h"
#include "ultis.cuh"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

// ------------------------------------------------------------------ CLI
// parsing
void printUsage(const char *exe) {
  fprintf(stderr,
          "Usage:\n"
          "  %s -F <input_dir>                 -o <output_image> [--version N] "
          "[--runs N]\n"
          "  %s -f <img1> <img2> [<img3> ...]  -o <output_image> [--version N] "
          "[--runs N]\n"
          "  %s <left> <right> <output_image>  [--version N] [--runs N]\n"
          "Images are stitched left to right (-F: natural filename order).\n"
          "  --version N   pipeline version, default 0 (sequential)\n"
          "  --runs N      timed runs after one warm-up run, default 5\n",
          exe, exe, exe);
}

static bool isFlag(const char *s) { return s[0] == '-' && s[1] != '\0'; }

bool parseArgs(int argc, char **argv, CliOptions &opt, std::string &err) {
  std::string dir;
  bool haveF = false, havef = false;
  std::vector<std::string> files, positional;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      err.clear();
      return false;
    } else if (a == "-F") {
      if (i + 1 >= argc) {
        err = "-F needs a directory";
        return false;
      }
      dir = argv[++i];
      haveF = true;
    } else if (a == "-f") {
      havef = true;
      while (i + 1 < argc && !isFlag(argv[i + 1]))
        files.push_back(argv[++i]);
      if (files.empty()) {
        err = "-f needs at least one image path";
        return false;
      }
    } else if (a == "-o") {
      if (i + 1 >= argc) {
        err = "-o needs an output path";
        return false;
      }
      opt.outputPath = argv[++i];
    } else if (a == "--version") {
      if (i + 1 >= argc) {
        err = "--version needs a number";
        return false;
      }
      opt.version = std::atoi(argv[++i]);
    } else if (a == "--runs") {
      if (i + 1 >= argc) {
        err = "--runs needs a number";
        return false;
      }
      opt.runs = std::atoi(argv[++i]);
    } else if (isFlag(argv[i])) {
      err = "unknown option '" + a + "'";
      return false;
    } else
      positional.push_back(a);
  }

  if (haveF && havef) {
    err = "use either -F or -f, not both";
    return false;
  }
  if (haveF) {
    if (!listImagesInDir(dir, opt.inputs)) {
      err = "cannot read image directory '" + dir + "'";
      return false;
    }
  } else if (havef) {
    opt.inputs = files;
  } else if (positional.size() == 3 &&
             opt.outputPath.empty()) { // course-spec form
    opt.inputs = {positional[0], positional[1]};
    opt.outputPath = positional[2];
  } else {
    err = "no input given (use -F <dir>, -f <img1> <img2> ..., or <left> "
          "<right> <out>)";
    return false;
  }
  if (opt.inputs.size() < 2) {
    err = "need at least 2 input images (found " +
          std::to_string(opt.inputs.size()) + ")";
    return false;
  }
  if (opt.outputPath.empty()) {
    err = "missing -o <output_image>";
    return false;
  }
  if (opt.runs < 1) {
    err = "--runs must be >= 1";
    return false;
  }
  if (opt.version < 0) {
    err = "--version must be >= 0";
    return false;
  }
  return true;
}

// -------------------------------------------------------------------- image
// I/O
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
  fs::path parent = fs::path(path).parent_path();
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
    ok =
        stbi_write_bmp(path.c_str(), img.width, img.height, 3, img.data.data());
  else
    ok = stbi_write_png(path.c_str(), img.width, img.height, 3, img.data.data(),
                        img.width * 3);
  if (!ok)
    fprintf(stderr, "Cannot write '%s'\n", path.c_str());
  return ok != 0;
}

// Natural order: digit runs compare by numeric value, everything else
// case-insensitively.
static bool naturalLess(const std::string &a, const std::string &b) {
  size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    if (std::isdigit((unsigned char)a[i]) &&
        std::isdigit((unsigned char)b[j])) {
      size_t si = i, sj = j;
      while (i < a.size() && std::isdigit((unsigned char)a[i]))
        ++i;
      while (j < b.size() && std::isdigit((unsigned char)b[j]))
        ++j;
      while (si + 1 < i && a[si] == '0')
        ++si; // strip leading zeros
      while (sj + 1 < j && b[sj] == '0')
        ++sj;
      size_t la = i - si, lb = j - sj;
      if (la != lb)
        return la < lb;
      int c = a.compare(si, la, b, sj, lb);
      if (c)
        return c < 0;
    } else {
      int ca = std::tolower((unsigned char)a[i]),
          cb = std::tolower((unsigned char)b[j]);
      if (ca != cb)
        return ca < cb;
      ++i;
      ++j;
    }
  }
  return (a.size() - i) < (b.size() - j);
}

bool listImagesInDir(const std::string &dir, std::vector<std::string> &paths) {
  std::error_code ec;
  if (!fs::is_directory(dir, ec))
    return false;
  paths.clear();
  for (const auto &e : fs::directory_iterator(dir, ec)) {
    if (!e.is_regular_file())
      continue;
    const std::string ext = lowerExt(e.path().string());
    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
      paths.push_back(e.path().string());
  }
  std::sort(paths.begin(), paths.end(), naturalLess);
  return true;
}

// -------------------------------------------------------------------
// evaluation
void applyHomography(const Homography &H, double x, double y, double &ox,
                     double &oy) {
  const double *h = H.h;
  double w = h[6] * x + h[7] * y + h[8];
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
int clampInt(int val, int lo, int hi) {
  if (val < lo)
    return lo;
  if (val > hi)
    return hi;
  return val;
}