#!/usr/bin/env bash
# Compile a single .cu file. Works unchanged in three places:
#   - Google Colab / Kaggle (real GPU present -> arch detected from it)
#   - the podman nvidia/cuda container (no GPU -> falls back to sm_75)
#   - any machine with a local CUDA toolkit
#
# Set CUDA_COMPILE_ONLY=1 to syntax/type-check without linking an executable.
# Set CUDA_ARCH=sm_XX to override architecture detection.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <path/to/file.cu>" >&2
    exit 2
fi

src="$1"
[[ -f "$src" ]] || { echo "no such file: $src" >&2; exit 2; }

# Target the architecture of the GPU we are actually on. Compiling for the real
# compute capability emits SASS for that device, which avoids the Colab failure
# mode where the toolkit is newer than the driver and the driver refuses to JIT
# the PTX ("the provided PTX was compiled with an unsupported toolchain").
arch="${CUDA_ARCH:-}"
if [[ -z "$arch" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
    [[ -n "$cc" ]] && arch="sm_${cc}"
fi
arch="${arch:-sm_75}"   # Colab free tier hands out a T4 = compute capability 7.5

# -lineinfo costs nothing and lets compute-sanitizer and Nsight map results back
# to source lines, which matters as soon as you start profiling.
flags=(-arch="$arch" -std=c++17 -O2 -lineinfo -I"$(dirname "$0")/../src/common")

if [[ "${CUDA_COMPILE_ONLY:-0}" == "1" ]]; then
    echo "  [check ${arch}] ${src}"
    nvcc "${flags[@]}" -c "$src" -o /dev/null
else
    mkdir -p build
    out="build/$(basename "${src%.cu}")"
    echo "  [build ${arch}] ${src} -> ${out}"
    nvcc "${flags[@]}" "$src" -o "$out"
fi
