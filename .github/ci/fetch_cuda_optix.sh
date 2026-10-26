#!/bin/bash
# Fetch what the OptiX backend needs to build, into $DEPS (default ../deps):
#   - nvcc, the CUDA runtime and cccl from NVIDIA's redistributable archives of
#     CUDA 11.7 (about 40 MB instead of the full installer); 11.7 keeps the PTX
#     loadable by drivers R515 and newer, as OptiX 7.5 itself
#   - the OptiX 7.5 headers from https://github.com/NVIDIA/optix-dev
# Prints the CUDA and OptiX directories as CUDA_DIR=... and OPTIX_DIR=... lines.
set -e

DEPS=${DEPS:-$(pwd)/../deps}
CUDAVER=${CUDAVER:-11.7.99}
CCCLVER=${CCCLVER:-11.7.91}
OPTIXVER=${OPTIXVER:-v7.5.0}
REDIST=https://developer.download.nvidia.com/compute/cuda/redist
CUDAHOME=$DEPS/cuda-$CUDAVER
OPTIXHOME=$DEPS/optix-$OPTIXVER

mkdir -p "$DEPS"

if [ ! -x "$CUDAHOME/bin/nvcc" ]; then
    mkdir -p "$CUDAHOME" "$DEPS/dl"
    for pkg in "cuda_nvcc:$CUDAVER" "cuda_cudart:$CUDAVER" "cuda_cccl:$CCCLVER"; do
        name=${pkg%%:*}
        ver=${pkg##*:}
        curl -fsSL -o "$DEPS/dl/$name.tar.xz" "$REDIST/$name/linux-x86_64/$name-linux-x86_64-$ver-archive.tar.xz"
        tar -xJf "$DEPS/dl/$name.tar.xz" -C "$CUDAHOME" --strip-components=1
    done
    rm -rf "$DEPS/dl"
fi

if [ ! -f "$OPTIXHOME/include/optix.h" ]; then
    git -c advice.detachedHead=false clone -q --depth 1 --branch "$OPTIXVER" https://github.com/NVIDIA/optix-dev.git "$OPTIXHOME"
fi

echo "CUDA_DIR=$CUDAHOME"
echo "OPTIX_DIR=$OPTIXHOME"
