#!/bin/bash
# package.sh tag binary... : make packages/vkmmc-<tag>.zip, as the mcx/mmc
# packages: a vkmmc/ folder with bin/ (the binaries, stripped) and the docs.
# tag is e.g. linux-x64-github-latest (os-arch-github-latest, lower case).
set -e
tag=$1
shift
pkg=packages/vkmmc
rm -rf "$pkg"
mkdir -p "$pkg/bin"

for b in "$@"; do
    cp "$b" "$pkg/bin/"
done

case "$tag" in
    linux*) strip --strip-unneeded "$pkg"/bin/* ;;
    macos*) strip -x "$pkg"/bin/* ;;
esac

cp README.md LICENSE.txt AUTHORS.txt ChangeLog.txt "$pkg/"

case "$tag" in
    macos*)
        cat > "$pkg/README_MACOS.txt" << 'EOT'
vkmmc needs a Vulkan driver with ray query (VK_KHR_ray_query). No macOS
driver provides it at the moment: MoltenVK, the Vulkan driver for Apple GPUs,
does not implement ray tracing. On macOS this build can parse and check
inputs and write the generated meshes (--dumpjson, --dumpmesh, --debugcurv),
but it cannot run simulations; use Linux or Windows with an NVIDIA RTX,
AMD RX 6000+ or Intel Arc GPU for that.

The binary loads the Vulkan loader (libvulkan.1.dylib) from the LunarG
Vulkan SDK or Homebrew (brew install vulkan-loader).
EOT
        ;;
    windows*)
        cat > "$pkg/README_WINDOWS.txt" << 'EOT'
vkmmc.exe needs a GPU driver with Vulkan ray query (VK_KHR_ray_query): NVIDIA
RTX, AMD RX 6000 or newer, or Intel Arc. The Vulkan loader (vulkan-1.dll) is
installed with the GPU driver. Run "vkmmc -L" to list the usable devices.
EOT
        ;;
esac

out=packages/vkmmc-$tag.zip
rm -f "$out"

if command -v 7z > /dev/null && [[ "$tag" == windows* ]]; then
    (cd packages && 7z a -tzip "vkmmc-$tag.zip" vkmmc > /dev/null)
else
    (cd packages && zip -qr "vkmmc-$tag.zip" vkmmc)
fi

rm -rf "$pkg"
ls -l "$out"
