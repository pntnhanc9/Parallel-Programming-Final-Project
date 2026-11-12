#!/usr/bin/env bash
# build.sh -- build the `stitch` program from every .cu / .cpp file in src/.
#
#   bash build.sh           build ./stitch (next to this script)
#   bash build.sh clean     remove the binary
#
# Optional environment variables
#   OUT=<path>          output binary          (default: <script dir>/stitch)
#   CUDA_ARCH=<arch>    e.g. sm_75 (T4), sm_86, native; passed to nvcc as
#                       -arch=. Unset = nvcc default.
#   OPT=<flags>         optimisation flags     (default: -O2)
#
# The same flags are used for every version; no -ffast-math nor -march=native,
# so the CPU baseline does not silently depend on the host.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
OUT="${OUT:-$PWD/stitch}"

if [[ "${1:-}" == "clean" ]]; then
  rm -f "$OUT"
  echo "[build] removed $OUT"
  exit 0
fi

SRC_FILES=(*.cu *.cpp)
OPT="${OPT:--O2}"

ARCH_FLAG=()
[[ -n "${CUDA_ARCH:-}" ]] && ARCH_FLAG=(-arch="$CUDA_ARCH")
echo "[build] nvcc: $(nvcc --version | grep -o 'release [0-9.]*')"
echo "[build] sources: ${SRC_FILES[*]}"
set -x
nvcc -std=c++17 $OPT ${ARCH_FLAG[@]+"${ARCH_FLAG[@]}"} -Xcompiler -Wall,-Wextra "${SRC_FILES[@]}" -o "$OUT"
{ set +x; } 2>/dev/null