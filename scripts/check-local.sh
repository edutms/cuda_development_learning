#!/usr/bin/env bash
# Compile-check every .cu file locally, inside a CUDA container.
#
# This machine has no NVIDIA GPU, but nvcc does not need one to *compile* --
# only to run. So the whole edit loop (typos, type errors, bad kernel launches)
# is caught here in seconds, and Colab's metered GPU hours are spent only on
# actually executing kernels.
set -euo pipefail

# Bump to match whatever `nvcc --version` reports in Colab if you ever hit a
# version-specific compile difference.
IMAGE="${CUDA_IMAGE:-docker.io/nvidia/cuda:12.4.1-devel-ubuntu22.04}"

cd "$(dirname "$0")/.."

mapfile -t sources < <(find src -name '*.cu' | sort)
if [[ ${#sources[@]} -eq 0 ]]; then
    echo "no .cu files under src/"
    exit 0
fi

echo "Compile-checking ${#sources[@]} file(s) in ${IMAGE}"
echo "(first run pulls ~7GB, then it is cached)"
echo

# :Z relabels the bind mount for SELinux -- required on Fedora, without it the
# container cannot read the sources.
podman run --rm -v "$PWD":/work:Z -w /work --entrypoint bash "$IMAGE" \
    -c 'set -e; for f in "$@"; do CUDA_COMPILE_ONLY=1 ./scripts/build.sh "$f"; done' \
    _ "${sources[@]}"

echo
echo "All files compiled cleanly."
