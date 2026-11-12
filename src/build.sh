#!/usr/bin/env bash
# build.sh -- build the `stitch` program from every .cu / .cpp file in src/.
#
#   bash build.sh           build ./stitch (next to this script)
#   bash build.sh clean     remove the binary
#
# Works from any working directory (local machine and Colab cells alike).
# Source files are discovered automatically, so adding v1_*.cu etc. needs no
# edit here.
#
# Optional environment variables
#   OUT=<path>          output binary          (default: <script dir>/stitch)
#   NVCC=<nvcc>         CUDA compiler          (default: nvcc, if found)
#   CXX=<g++>           host compiler, used by the CPU-only fallback
#   CUDA_ARCH=<arch>    e.g. sm_75 (T4), sm_86, native; passed to nvcc as
#                       -arch=. Unset = nvcc default. Needed once V1+ has
#                       kernels; not needed for V0.
#   OPT=<flags>         optimisation flags     (default: -O2)
#
# V0 fairness: the same flags are used for every version; no -ffast-math and
# no -march=native, so the CPU baseline does not silently depend on the host.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
OUT="${OUT:-$PWD/stitch}"

if [[ "${1:-}" == "clean" ]]; then
  rm -f "$OUT"
  echo "[build] removed $OUT"
  exit 0
fi

# ---- stb headers: fetch any that is missing (needs internet once; commit both
# files to avoid this). Pinned to a commit on purpose: stb `master` from Nov 2026
# has a regression in stbi_write_bmp / stbi_write_jpg (fclose on a bad handle ->
# "glibc detected an invalid stdio handle"). b5ca528 = stb_image_write v1.16.
STB_COMMIT="b5ca5281428127f3b956ee123f7413c6f4b208c1"
STB_URL="https://raw.githubusercontent.com/nothings/stb/$STB_COMMIT"
for h in stb_image.h stb_image_write.h; do
  if [[ ! -f "$h" ]]; then
    echo "[build] $h not found, downloading from $STB_URL"
    if command -v curl >/dev/null 2>&1; then
      curl -fsSL -o "$h" "$STB_URL/$h"
    else
      wget -q -O "$h" "$STB_URL/$h"
    fi
  fi
done

SRC_FILES=(*.cu *.cpp)
OPT="${OPT:--O2}"
NVCC="${NVCC:-nvcc}"
CXX="${CXX:-g++}"

if command -v "$NVCC" >/dev/null 2>&1; then
  ARCH_FLAG=()
  [[ -n "${CUDA_ARCH:-}" ]] && ARCH_FLAG=(-arch="$CUDA_ARCH")
  echo "[build] nvcc: $("$NVCC" --version | grep -o 'release [0-9.]*')"
  echo "[build] sources: ${SRC_FILES[*]}"
  set -x
  "$NVCC" -std=c++17 $OPT ${ARCH_FLAG[@]+"${ARCH_FLAG[@]}"} \
    -Xcompiler -Wall,-Wextra \
    "${SRC_FILES[@]}" -o "$OUT"
  set +x
else
  # CPU-only fallback so V0 can be developed on a machine without CUDA.
  # .cu files are plain C++ in V0, so they are compiled as C++. This stops
  # working as soon as a file contains real CUDA code (V1+): use nvcc then.
  echo "[build] nvcc not found -> CPU-only build with $CXX (V0 only)"
  echo "[build] sources: ${SRC_FILES[*]}"
  set -x
  "$CXX" -std=c++17 $OPT -Wall -Wextra -x c++ "${SRC_FILES[@]}" -o "$OUT"
  set +x
fi
echo "[build] done -> $OUT"