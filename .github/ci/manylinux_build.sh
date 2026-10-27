#!/bin/bash
# Build the release binaries in the manylinux2014 image (CentOS 7, glibc 2.17)
# so that they run on any x86-64 Linux distribution from 2014 on:
#
#   make -C src spvheaders              # on the host: SPIR-V needs a recent glslang
#   docker run --rm -e HOST_UID=$(id -u) -v $PWD:/src -w /src \
#       quay.io/pypa/manylinux2014_x86_64 bash .github/ci/manylinux_build.sh
#
# Inside the container this script
#   - builds the Vulkan headers and loader 1.3.204 (CentOS 7's are too old for
#     ray query) into $DEPS/vulkan; they are only linked against: at run time
#     the system's libvulkan.so.1 is used
#   - builds GLFW 3.4 (X11 only) as a static library for the --show viewer;
#     it loads libX11 and its extensions at run time, so vkmmc does not need
#     them to run simulations on headless machines
#   - fetches CUDA 11.7 and the OptiX 7.5 headers (fetch_cuda_optix.sh)
#   - builds bin/vkmmc (with --show) and bin/vkmmc_optix with libstdc++,
#     libgcc and the CUDA runtime linked statically
#   - checks that they need glibc 2.17 at most and only system libraries
#   - runs the smoke tests and packages dist/vkmmc-<version>-linux-x86_64.tar.gz
set -e

cd "${SRCDIR:-/src}"
export PATH=/opt/python/cp311-cp311/bin:$PATH
DEPS=${DEPS:-$(pwd)/.deps}
VKVER=${VKVER:-v1.3.204}
VULKAN=$DEPS/vulkan-$VKVER
GLFWVER=${GLFWVER:-3.4}
GLFW=$DEPS/glfw-$GLFWVER

git config --global --add safe.directory '*' 2>/dev/null || true
command -v xxd > /dev/null || yum install -y -q vim-common xz > /dev/null

for h in vkmmc_core vkmmc_show_vert vkmmc_show_frag vkmmc_show_vol_vert vkmmc_show_vol_frag; do
    [ -f "src/${h}_spv.h" ] || { echo "run 'make -C src spvheaders' on the host first" >&2; exit 1; }
done

# --- Vulkan headers and loader, for linking only ---
if [ ! -f "$VULKAN/include/vulkan/vulkan.h" ] || [ ! -e "$VULKAN/lib64/libvulkan.so" ]; then
    mkdir -p "$DEPS"
    rm -rf "$DEPS/vkbuild"
    mkdir -p "$DEPS/vkbuild"
    for r in Vulkan-Headers Vulkan-Loader; do
        git -c advice.detachedHead=false clone -q --depth 1 -b "$VKVER" "https://github.com/KhronosGroup/$r.git" "$DEPS/vkbuild/$r"
    done
    cmake -S "$DEPS/vkbuild/Vulkan-Headers" -B "$DEPS/vkbuild/h" -DCMAKE_INSTALL_PREFIX="$VULKAN" -Wno-dev > /dev/null
    cmake --install "$DEPS/vkbuild/h" > /dev/null
    cmake -S "$DEPS/vkbuild/Vulkan-Loader" -B "$DEPS/vkbuild/l" -DCMAKE_INSTALL_PREFIX="$VULKAN" -DCMAKE_INSTALL_LIBDIR=lib64 \
          -DVULKAN_HEADERS_INSTALL_DIR="$VULKAN" -DBUILD_WSI_XCB_SUPPORT=OFF -DBUILD_WSI_XLIB_SUPPORT=OFF \
          -DBUILD_WSI_WAYLAND_SUPPORT=OFF -DBUILD_WSI_DIRECTFB_SUPPORT=OFF -DBUILD_TESTS=OFF \
          -DCMAKE_BUILD_TYPE=Release -Wno-dev > /dev/null
    cmake --build "$DEPS/vkbuild/l" -j"$(nproc)" > /dev/null
    cmake --install "$DEPS/vkbuild/l" > /dev/null
    rm -rf "$DEPS/vkbuild"
fi

# --- GLFW, static and X11 only, for the --show viewer ---
if [ ! -f "$GLFW/lib/libglfw3.a" ]; then
    yum install -y -q libX11-devel libXrandr-devel libXinerama-devel libXcursor-devel libXi-devel > /dev/null
    rm -rf "$DEPS/glfwbuild"
    git -c advice.detachedHead=false clone -q --depth 1 -b "$GLFWVER" https://github.com/glfw/glfw.git "$DEPS/glfwbuild/glfw"
    cmake -S "$DEPS/glfwbuild/glfw" -B "$DEPS/glfwbuild/b" -DCMAKE_INSTALL_PREFIX="$GLFW" -DCMAKE_INSTALL_LIBDIR=lib \
          -DBUILD_SHARED_LIBS=OFF -DGLFW_BUILD_X11=ON -DGLFW_BUILD_WAYLAND=OFF -DGLFW_BUILD_EXAMPLES=OFF \
          -DGLFW_BUILD_TESTS=OFF -DGLFW_BUILD_DOCS=OFF -DCMAKE_C_FLAGS=-D_GNU_SOURCE -DCMAKE_BUILD_TYPE=Release \
          -Wno-dev > /dev/null
    cmake --build "$DEPS/glfwbuild/b" -j"$(nproc)" > /dev/null
    cmake --install "$DEPS/glfwbuild/b" > /dev/null
    rm -rf "$DEPS/glfwbuild"
fi

# --- CUDA 11.7 and the OptiX headers ---
eval "$(DEPS=$DEPS bash .github/ci/fetch_cuda_optix.sh)"

# --- build ---
rm -f bin/vkmmc bin/vkmmc_optix
CXXFLAGS="-I$VULKAN/include" LDFLAGS="-L$VULKAN/lib64" make -C src -j"$(nproc)" STATIC=1 SHOW=1 \
    GLFW_INC="-I$GLFW/include" GLFW_LIB="$GLFW/lib/libglfw3.a -ldl -lrt -lm"
make -C optix clean > /dev/null
make -C optix -j"$(nproc)" STATIC=1 CUDA_DIR="$CUDA_DIR" OPTIX_DIR="$OPTIX_DIR"

# --- checks ---
SYSLIBS="libm.so libc.so libdl.so libpthread.so librt.so ld-linux"
MAXGLIBC=2.17 bash .github/ci/check_deps.sh bin/vkmmc libvulkan.so.1 $SYSLIBS
MAXGLIBC=2.17 bash .github/ci/check_deps.sh bin/vkmmc_optix $SYSLIBS
LD_LIBRARY_PATH=$VULKAN/lib64 bash .github/ci/smoke.sh bin/vkmmc
LD_LIBRARY_PATH=$VULKAN/lib64 bin/vkmmc -h | grep -q -- '--show' || { echo "ERROR: bin/vkmmc has no --show" >&2; exit 1; }
bash .github/ci/smoke.sh bin/vkmmc_optix

# --- package ---
VER=$(sed -n 's/^#define VKMMC_VERSION *"\(.*\)"/\1/p' src/vkmmc_host.h)
PKG=vkmmc-$VER-linux-x86_64
rm -rf "dist/$PKG"
mkdir -p "dist/$PKG/bin"
strip -o "dist/$PKG/bin/vkmmc" bin/vkmmc
strip -o "dist/$PKG/bin/vkmmc_optix" bin/vkmmc_optix
cp README.md LICENSE.txt AUTHORS.txt ChangeLog.txt "dist/$PKG/"
tar -C dist -czf "dist/$PKG.tar.gz" "$PKG"
ls -l "dist/$PKG.tar.gz"

[ -n "$HOST_UID" ] && chown -R "$HOST_UID" bin dist src optix "$DEPS" 2>/dev/null || true
