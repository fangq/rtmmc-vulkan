Ray-Tracing Mesh-based Monte Carlo (RT-MMC) - Vulkan and OptiX
===============================================================

-   Author: Qianqian Fang (q.fang at neu.edu)
-   License: GNU General Public License version 3 (GPL v3), see LICENSE.txt
-   Version: 0.5.0 (Kylin, beta)
-   URL: <https://github.com/fangq/rtmmc-vulkan>

![Build and test](https://github.com/fangq/rtmmc-vulkan/actions/workflows/build.yml/badge.svg)

Table of Content:

  * [What's New](#whats-new)
  * [Introduction](#introduction)
  * [Requirements](#requirements)
  * [Download and Compile](#download-and-compile)
  * [Running Simulations](#running-simulations)
    + [Command line options](#command-line-options)
    + [Input file format](#input-file-format)
    + [CSG shapes](#csg-shapes)
    + [Mesh input](#mesh-input)
    + [Units](#units)
    + [Output files](#output-files)
    + [Interactive viewer](#interactive-viewer)
  * [Accuracy and reproducibility](#accuracy-and-reproducibility)
  * [Testing](#testing)
  * [Known issues and limitations](#known-issues-and-limitations)
  * [Acknowledgement](#acknowledgement)
  * [Reference](#reference)


What's New
----------

vkmmc 0.5.0 (Kylin) is the first, beta release of the ray-tracing (RT) accelerated mesh-based
Monte Carlo (MMC) photon transport simulator. It brings the RT-MMC algorithm,
previously available in MMC through NVIDIA OptiX, to any GPU that supports the
Vulkan ray-query extension (NVIDIA RTX, AMD RDNA2 and newer, Intel Arc), and
adds shape-based (CSG) domains with curvature-corrected surface normals [Yen2026].

Key features of this release

* two backends sharing all host code: `vkmmc` (Vulkan ray query) and `vkmmc_optix`
  (OptiX 7); every kernel variant is embedded in the binary and selected at run
  time from the input and the GPU features
* all 18 MCX/MMC source types, shared patterns (photon sharing) and multiple
  sources
* two domain types: tetrahedral or triangular meshes (MMC/iso2mesh JSON input with
  JData-encoded arrays), and MCX-style CSG shape lists (Grid, Subgrid, Box, Sphere,
  Cylinder, X/Y/Z-Slabs and -Layers, Lens, Origin) triangulated on the fly
* curvature-corrected normals for curved shapes (sphere, cylinder, lens caps)
  that make refraction and Fresnel reflection on coarse meshes close to those
  of the exact surfaces
* exact energy deposition: paths are traversed voxel by voxel and split at
  time-gate boundaries, and sums are accumulated as 64-bit fixed-point integers,
  so results do not lose precision with the photon count and runs are
  bit-reproducible for a given seed and GPU, whatever the thread count and
  batch size
* dynamic load balancing: GPU threads claim photons from a shared counter, and
  kernel launches are sized automatically
* time-resolved energy, fluence and flux output, saved as JData (`.jdat`) files
  compressed with multi-threaded zlib (zmat)
* photon trajectories (`-D M`, Vulkan backend) and an interactive Vulkan viewer
  (`--show`) for the geometry, trajectories and fluence volumes
* strict validation of every input field, with error messages that name the
  offending field; host-side unit tests, a GPU functional test suite and
  libFuzzer/mutation fuzzers are included


Introduction
------------

Mesh-based Monte Carlo (MMC) [Fang2010] models light transport in complex media
whose boundaries are described by a triangle surface mesh. Instead of walking
the photon from tetrahedron to tetrahedron, RT-MMC [Yan2026] uses the GPU's
ray-tracing hardware to find the next boundary a photon crosses along its path:
the surface triangles are stored in a bounding volume hierarchy (BVH), each
propagation step is a single ray query, and the media on the two sides of each
triangle tell the photon which medium it enters.

Photons are launched from the source, travel exponentially distributed
scattering lengths, are scattered with the Henyey-Greenstein phase function,
lose weight by absorption along their paths, are reflected or refracted at
refractive index mismatches by Fresnel's equations, and are terminated when
they leave the domain, reach the end of the time window or lose the Russian
roulette. The energy lost along each path is accumulated on a regular voxel
grid that covers the domain.

In CSG mode a photon carries a small state (up to 8 nested shapes) listing the
shapes that enclose it; the shape with the highest Tag defines the medium, so
later/higher-tagged shapes override the ones they overlap, as in MCX.


Requirements
------------

Building:

* a C++11 compiler, and GNU make or CMake 3.17 or newer
* Vulkan backend (`vkmmc`): the Vulkan headers and loader (`libvulkan-dev`) and
  `glslangValidator` (`glslang-tools`); `xxd` for the Makefile build
* OptiX backend (`vkmmc_optix`): the CUDA toolkit (11.7 or newer; only `nvcc`,
  the runtime and cccl are used) and the OptiX 7 headers only, e.g.
  `git clone -b v7.5.0 https://github.com/NVIDIA/optix-dev`; no OptiX SDK
  installation is needed
* optional: GLFW 3 for the `--show` viewer; zlib (otherwise the bundled miniz is
  used); python3 for the tests; clang for the fuzzers

Running:

* `vkmmc`: `libvulkan.so.1` (the Vulkan loader) and a driver with
  `VK_KHR_ray_query` and `VK_KHR_acceleration_structure`: NVIDIA RTX 20xx or
  newer, AMD RX 6000 or newer with RADV, Intel Arc, or Mesa's CPU driver
  lavapipe (tested with Mesa 25.0 and 26.0; about 500 times slower than an
  RTX 4070 Ti SUPER on 16 CPU cores, but with the same results, which makes it
  useful for testing); 64-bit shader atomics are used when available and
  emulated otherwise
* `vkmmc_optix`: an NVIDIA GPU and driver; OptiX comes with the driver and the
  CUDA runtime is linked statically. A binary built with the OptiX 7.5 headers
  needs driver R515 or newer; newer headers need newer drivers
* both: glibc and, unless built with `STATIC=1`, libstdc++; the kernels are
  embedded, so no other files are needed

`vkmmc -L` lists the Vulkan devices and whether each can run vkmmc.


Download and Compile
--------------------

Prebuilt Linux binaries (`vkmmc-<version>-linux-x86_64.tar.gz`, both
backends) are attached to the GitHub releases and to every CI run. They are
built in the manylinux2014 image (CentOS 7), link libstdc++, libgcc and the
CUDA runtime statically, and run on any x86-64 Linux with glibc 2.17 or newer
(tested on CentOS 7, Ubuntu 16.04 and 22.04); `vkmmc` also needs the
distribution's Vulkan loader (`libvulkan1` or `vulkan-loader`). To build them
locally:

```
make -C src vkmmc_core_spv.h      # SPIR-V needs a recent glslang, on the host
docker run --rm -e HOST_UID=$(id -u) -v $PWD:/src -w /src \
    quay.io/pypa/manylinux2014_x86_64 bash .github/ci/manylinux_build.sh
```

To build from source with make:

```
git clone https://github.com/fangq/rtmmc-vulkan.git
cd rtmmc-vulkan
make -C src                                  # builds bin/vkmmc
make -C optix OPTIX_DIR=/path/to/optix-dev   # builds bin/vkmmc_optix
```

Build options (for `make -C src`, and `STATIC`/`HAVE_ZLIB`/`DEBUG` also for `optix`):

| option          | effect                                                       |
|-----------------|--------------------------------------------------------------|
| `SHOW=1`        | add the interactive `--show` viewer (needs GLFW 3)           |
| `STATIC=1`      | link libstdc++ and libgcc statically (for releases)          |
| `HAVE_ZLIB=yes` | link the system zlib instead of the bundled miniz            |
| `DEBUG=1`       | debug build                                                  |
| `CUDA_DIR=`, `OPTIX_DIR=` | CUDA toolkit and OptiX headers (`optix` only)      |
| `make asan`     | AddressSanitizer/UBSan build `bin/vkmmc_asan`                |
| `make pretty`   | format the sources with astyle                               |

With CMake (both backends; the OptiX one when CUDA and the OptiX headers are
found):

```
cmake -S . -B build -DOPTIX_ROOT=/path/to/optix-dev
cmake --build build -j
ctest --test-dir build -LE gpu     # tests that need no GPU; drop -LE gpu for all
```

CMake options: `VKMMC_BUILD_VULKAN` (ON), `VKMMC_BUILD_OPTIX` (AUTO/ON/OFF),
`OPTIX_ROOT`, `CUDAToolkit_ROOT`, `VKMMC_SHOW`, `VKMMC_USE_SYSTEM_ZLIB`,
`VKMMC_STATIC_RUNTIME` and `VKMMC_BUILD_TESTS` (ON).

No build option selects the simulation mode: the mesh, CSG and
CSG+curvature kernels are all compiled and embedded, and the one matching the
input is used. The source settings (type, focus, patterns, multiple sources)
are compile-time constants of the kernel, set when it is loaded (Vulkan
specialization constants, OptiX bound values), as are `DoSpecular` and a
known initial medium: the GPU driver compiles only the source code a run
uses, so the many source types cost a pencil beam nothing.


Running Simulations
-------------------

```
bin/vkmmc input.json                     # Vulkan backend
bin/vkmmc_optix input.json -n 1e7        # OptiX backend
```

A minimal input, a pencil beam entering an absorbing and scattering sphere:

```json
{
    "Session": {"ID": "sphere", "Photons": 1000000, "RNGSeed": 1234, "DoMismatch": true},
    "Forward": {"T0": 0, "T1": 5e-9, "Dt": 5e-9},
    "Domain":  {"LengthUnit": 1, "Media": [
                   {"mua": 0,    "mus": 0, "g": 1,   "n": 1},
                   {"mua": 0.01, "mus": 10, "g": 0.9, "n": 1.37}]},
    "Optode":  {"Source": {"Type": "pencil", "Pos": [30, 30, 19], "Dir": [0, 0, 1]}},
    "Shapes":  [{"Sphere": {"Tag": 1, "O": [30, 30, 30], "R": 10}}]
}
```

The run prints the absorbed fraction and the speed, and saves the volume to
`sphere.jdat`.

### Command line options

| option | description | default |
|--------|-------------|---------|
| `-f/--input file` | JSON input file (or the first positional argument) | |
| `-j/--json str` | JSON string merged into the input (JSON merge patch) | |
| `-n/--photon num` | photon number, accepts `1e7` | `Session.Photons` |
| `-s/--session name` | session name, the output file name prefix | `Session.ID` |
| `-u/--unitinmm mm` | length of one coordinate unit in mm | `Domain.LengthUnit` |
| `-E/--seed int` | RNG seed; 0 seeds from the clock | `Session.RNGSeed` |
| `-O/--outputtype c` | `x` energy, `f` flux, `l` fluence | `Session.OutputType` |
| `-b/--reflect 0\|1` | Fresnel reflection at refractive index mismatches | `Session.DoMismatch` |
| `-V/--specular 0\|1` | Fresnel reflection where photons from outside enter the domain | `Session.DoSpecular` |
| `-U/--normalize 0\|1` | normalize the output | `Session.DoNormalize` |
| `-t/--thread num` | total GPU threads, rounded up to a multiple of 64 | 131072 |
| `-B/--batch num` | photons per kernel launch (0: all at once) | auto |
| `-G/--gpuid id` | Vulkan: 1-based index among usable devices (0: auto, prefers a discrete GPU); OptiX: CUDA device | 0 |
| `-m/--meshres num` | tessellation of curved shapes, 4-2048 | `Domain.MeshRes`, 24 |
| `-c/--curv -1\|0\|1` | curvature-corrected normals (-1: on for curved shapes) | -1 |
| `-D/--debug flags` | `M` (or 8192): save photon trajectories | |
| `--maxjumpdebug num` | maximum trajectory events | 10000000 |
| `--dumpjson` | print the parsed input and mesh summary as JSON, exit | |
| `--dumpmesh` | save the surface mesh as a mesh-mode input `<session>_mesh.json`, exit | |
| `--debugcurv` | save the curvature normal error map `<session>_curv.json`, exit | |
| `--show [file.jdat]` | interactive viewer (built with `SHOW=1`) | |
| `-L/--listgpu` | list GPUs | |
| `-v/--version`, `-h/--help` | version, help | |

The threads claim the photons of a launch one by one from a shared counter,
so threads whose photons end early take more (dynamic load balancing). By
default the first launch runs 4 photons per thread and each later launch is
sized from the measured speed to take about 0.5 s (at most 2^24 photons),
within the watchdog limit of GPUs that drive a display; the output is read back
and cleared between launches. Each photon seeds its random number generator
from the run seed and its photon index, so `-t` and `-B` do not change the
result.

A trailing `.spv` (vkmmc) or `.ptx` (vkmmc_optix) argument replaces the
embedded kernel, for kernel development. The environment variable
`VKMMC_NO_ATOMIC64=1` forces the portable 32-bit atomic Vulkan kernel.

### Input file format

The input is an MMC/MCX-style JSON file:

* `Session`: `ID`, `Photons`, `RNGSeed`, `DoMismatch`, `DoSpecular`,
  `DoNormalize`, `OutputType` (`x`/`energy`, `f`/`flux`, `l`/`fluence`),
  `RussianRoulette` (weight threshold, default 1e-6), `RussianRouletteSize`
  (10), `DebugFlag` (a number or MCX's letters, e.g. `"M"`), `MaxJumpDebug`
* `Forward`: `T0`, `T1`, `Dt` in seconds; the output has `round((T1-T0)/Dt)`
  time gates; energy deposited outside `[T0, T1)` is not recorded
* `Domain`: `Media` (a list of `{mua, mus, g, n}` objects or `[mua, mus, g, n]`
  rows, `mua`/`mus` in 1/mm; up to 256 media, medium 0 is the background and its
  `n` is used outside the domain), `LengthUnit`, `Step` (output voxel size in
  length units, default 1), `MeshRes`
* `Optode.Source`: `Type`, `Pos`, `Dir` (normalized internally; `Dir[3]` is
  the focal length), `Param1`, `Param2`, `Pattern`, `SrcNum`, `ID`, see
  [Sources](#sources)
* `Shapes`: a list of CSG shapes, or a mesh object (see below)

A source outside the domain launches photons toward it, as in MCX: they
enter at the first surface they hit, after the time of flight through the
background; with `DoSpecular` they are reflected there by Fresnel's law
(the default, as MCX and MMC, is no reflection or refraction at the entry;
in a mesh with MMC's negative wide-field labels, where the photons cross a
real surface, `DoMismatch` applies there). Photons that miss the domain are launched again, so
the launched energy counts the photons that enter. A source on a surface is
inside the tissue there: a photon pointing out of it meets the boundary
(Fresnel reflection with `DoMismatch`).

### Sources

All 18 MCX/MMC source types are implemented and sampled as in MMC, with the
same `Param1`/`Param2` conventions (see the MCX documentation for details):

| `Type` | launch | `Param1` / `Param2` |
|--------|--------|---------------------|
| `pencil` | from `Pos` along `Dir` | |
| `isotropic` | uniform over 4pi | |
| `cone` | uniform within a cone around `Dir` | `[half angle (rad), 0: uniform in solid angle, >0: in angle]` |
| `arcsine` | zenith angle uniform in [0, pi] around `Dir` | |
| `zgaussian` | Gaussian zenith angle around `Dir` | `[sigma (rad)]` |
| `gaussian` | Gaussian beam profile | `[waist radius]`; with a focal length `[waist, wavelength]` |
| `disk`, `ring` | uniform disk / annulus perpendicular to `Dir` | `[outer radius, inner radius, phi0, phi1]` (ring: optional arc) |
| `hyperboloid` | hyperboloid Gaussian beam | `[waist radius, distance to the waist, Rayleigh range]` |
| `planar` | uniform over the parallelogram `Pos + u Param1 + v Param2` | edge vectors |
| `pattern` | `planar` weighted by an image | edge vectors; `Param1[3]` x `Param2[3]` pixels |
| `pattern3d` | box `Pos + Param1[0..2]` weighted by a volume | `[nx, ny, nz]`: the size and the pattern dimensions |
| `fourier` | `planar` weighted by (1+cos(2pi (kx u + ky v) + phi))/2 | `Param1[3] = kx + phi/2pi`, `Param2[3] = ky + M` |
| `fourierx`, `fourierx2d` | as `fourier`, the 2nd edge is `Dir x Param1` | `[edge1, |edge2|]` / `[kx, ky, phi0, M]` or `[kx, ky, phix, phiy]` |
| `pencilarray` | `Param1[3]` x `Param2[3]` pencil beams over the parallelogram | edge vectors and beam counts |
| `line` | a segment from `Pos` along `Param1`, emitting perpendicular to `Dir` | segment vector |
| `slit` | the same segment, emitting along `Dir` | segment vector |

* Focal length `Dir[3]`: wide-field beams (planar, pattern, fourier, disk,
  slit, ...) converge to (>0) or diverge from (<0) the point at that distance
  from the source center; `"_NaN_"` launches each photon isotropically and
  `"-_Inf_"` with a Lambertian (cosine) distribution.
* `Pattern`: a JData array `[Nx, Ny]` (`[Nx, Ny, Nz]` for `pattern3d`), read
  with the first index fastest as MMC does, or `{"Nx", "Ny", "Nz", "Data"}`
  with `Data` a flat array or a float32 file name. Values are in [0, 1000].
* Shared patterns (photon sharing): a `[SrcNum, Nx, Ny]` pattern, or
  `SrcNum` > 1, simulates SrcNum patterns with the same photons; the output has
  one field per pattern.
* Multiple sources: `Pos`, `Dir`, `Param1`, `Param2` given as lists of rows
  (a single row applies to all), each row a source of `Type` with its own
  parameters as in MCX; `Pos[3]` is the source weight (default 1), and for
  `pencil` rows `Param1[0]` is an optional disk radius (as mmc-optix). `ID` 0
  (or -1) launches each photon from a random source and saves one field per
  source, `ID` k simulates source k only. An isotropic (`"_NaN_"`) or
  Lambertian (`"-_Inf_"`) `Dir[3]` must be the same for all rows.
* The output is normalized by the launched energy (the sum of the launch
  weights; per pattern for shared patterns), and the printed absorbed
  fraction is relative to it.
* `InitElem` gives the initial medium of a single source with a fixed launch
  direction; otherwise the medium is found per photon.

Differences to mmc's OptiX backend: the `cone`, `arcsine` and `zgaussian`
directions are rotated to `Dir` (as in mmc's CPU and MCX; mmc-optix leaves
them around +z), and `cone` is sampled directly instead of by rejection.

### CSG shapes

`Shapes` is a list of single-key objects, processed in order. Coordinates are
in length units; `Tag` is the medium index (1-64).

| shape | parameters |
|-------|------------|
| `Grid` | `Size: [sx, sy, sz]`, `Tag` - a box from the origin; sets the extent used by slabs/layers |
| `Box` | `O`, `Size`, `Tag` |
| `Subgrid` | `O` (1-based voxel corner, as MCX), `Size`, `Tag` |
| `Sphere` | `O`, `R`, `Tag` |
| `Cylinder` | `C0`, `C1` (axis end points), `R`, `Tag` |
| `XSlabs`/`YSlabs`/`ZSlabs` | `Bound: [[lo, hi], ...]`, `Tag` - slabs spanning the Grid |
| `XLayers`/`YLayers`/`ZLayers` | `[[start, end, tag], ...]` - 1-based voxel layers, as MCX |
| `Lens` | `O`, `Dir`, `R` (aperture radius), `Tag`, `Front: {D, R}`, `Back: {D, R}` - two spherical caps at apex distances `D` from `O`, signed radii (negative: concave) |
| `Origin` | `[x, y, z]` subtracted from the positions of the following shapes |
| `Name` | a label, ignored |

The enclosing shape of a point is the one with the highest Tag; coincident
or touching flat faces of different shapes (for example adjacent layers) are
handled exactly. `--dumpmesh` writes the triangulation as a mesh input; for
nested shapes this mesh assigns medium 0 outside each shape and therefore
does not reproduce the CSG overlap rules.

### Mesh input

`Shapes` can instead be an object with

* `MeshNode`: N x 3 node coordinates
* `MeshElem`: M x 5 tetrahedra (4 1-based node indices + medium label); the
  surfaces between different labels are extracted, or
* `MeshSurf`: K x 4 triangles (3 1-based node indices + label). A triangle
  listed once is an exterior face; a triangle between two media is listed twice,
  once with each label. Each row is wound so that its normal (right-hand rule)
  points out of the region of its label.
* `InitElem`: optional 1-based element that contains the source (`MeshElem` only)

Negative labels (MMC's wide-field source and detector domains, -1/-2) are the
background (medium 0); the surfaces of the tissue are extracted from the rest.
The implicit MMC (`MeshROI`) is not supported.

Arrays may be inline JSON arrays or JData objects with any `_ArrayType_`
(`single`, `double`, `int8`-`uint64`), compressed by `zlib`, `gzip` or stored as
`base64` or plain `_ArrayData_`, in row- or column-major (`_ArrayOrder_: "c"`)
order, i.e. the files written by iso2mesh, jdata/JSONLab and pyjdata.

### Units

All coordinates are in length units of `LengthUnit` mm (`-u`). Optical
properties are in 1/mm, times in seconds. The output voxel is `Step` length
units (default 1). The grid covers the bounding box of the mesh with half a
voxel of margin; `Domain.Origin` in the output gives the corner of voxel
(0,0,0).

### Output files

`<session>.jdat` is a JSON file with `Session`, `Forward`, `Domain` (`Dim`,
`VoxelSize`, `Origin`, `LengthUnit`) and `Fluence`: a JData array of
`[nx, ny, nz(, nt)]` single-precision values stored x-fastest
(`_ArrayOrder_: "c"`), `[nx, ny, nz, nt, nfield]` for several fields (shared
patterns, multiple sources). It loads directly with `jd.load()` (pyjdata) or
`loadjd()` (JSONLab). With `DoNormalize`, the values are per unit of launched
energy (per launched photon for unit-weight sources):

| OutputType | quantity | unit |
|------------|----------|------|
| `x` | energy deposited per voxel volume | 1/mm^3 |
| `l` | fluence integrated over each time gate | 1/mm^2 |
| `f` | flux, the fluence divided by the gate width | 1/(mm^2 s) |

`-D M` also saves `<session>_traj.jdat` (MCX trajectory layout: `photonid`,
`p` (position), `w0` (weight) and `e0` (event: 0 launch, 1 scattering,
2 termination, 3 boundary crossing, 4 reflection)).

### Interactive viewer

Built with `SHOW=1`, `vkmmc input.json --show` opens a window showing the
refractive interfaces and the source. Drag to rotate, Ctrl+drag to pan, scroll
to zoom, `Space` or the toolbar to start the simulation, `M` to toggle the
maximum intensity projection of the fluence, `Esc` to quit. `vkmmc input.json
--show result.jdat` displays a saved volume and/or trajectory file.


Accuracy and reproducibility
----------------------------

* energy is conserved exactly: the deposited energy equals the weight lost by
  the photons (checked against the trajectory weights in the test suite)
* the output is accumulated as 64-bit fixed-point integers (2^-32 per unit
  weight), so the result does not depend on the order of the atomic additions,
  and a photon's random numbers depend only on the seed and its index: the same
  seed on the same GPU gives a bit-identical output for any thread count and
  batch size, and the native and portable atomic kernels give identical results
* deposits are summed in registers while a photon stays in one voxel and time
  gate and written when it leaves, which saves most atomic additions in
  scattering media with voxels larger than the scattering length
* float atomics were tried and dropped: a plain float sum stops absorbing
  small deposits once it is large (the source voxel of a pencil beam lost over
  half its value in a 1e8-photon launch), and a compensated float pair, as
  accurate as needed, was slower than the 64-bit integer sums on every GPU
  tested (RTX 2080 SUPER, 4090, 5090)
* straight-path absorption, Fresnel transmission and time of flight agree
  with the analytic values to within the float precision of the geometry
* the Vulkan and OptiX backends use the same algorithms and agree within the
  statistical error (their random number streams differ)


Testing
-------

```
make -C tests/unit run             # host unit tests (no GPU), also: asan, coverage
cd tests && ./test_vkmmc.sh ../bin/vkmmc          # GPU functional tests
XEXEC=../bin/vkmmc_optix ./test_vkmmc.sh ../bin/vkmmc   # + cross-backend checks
make -C tests/fuzz run T=600       # libFuzzer harnesses (needs clang)
make -C tests/fuzz regress         # replay past fuzzer findings
python3 tests/fuzz/fuzz_vkmmc.py bin/vkmmc_asan -n 5000         # mutation fuzzing
python3 tests/fuzz/fuzz_vkmmc.py bin/vkmmc --gpu -n 1000        # ... on the GPU
```

* `tests/unit`: 58 tests (966 checks) of the JSON/JData parser, input
  validation, mesh extraction, shape triangulation, curvature, command line,
  normalization and file output; 97-100% line coverage of the host headers
* `tests/test_vkmmc.sh`: 39 GPU tests (86 checks) against analytic solutions
  (Beer-Lambert, Fresnel, time of flight), conservation identities and
  equivalent inputs (mesh vs. CSG, every JData encoding, coplanar CSG faces);
  they cover 98.9% of the lines of the Vulkan front-end `vkmmc.cpp`
* `tests/fuzz`: libFuzzer harnesses for the input pipeline, the JData decoder
  and the command line, and a JSON-aware mutation fuzzer for the binaries

The GitHub Actions workflow (`.github/workflows/build.yml`) builds both
backends with make and CMake on Ubuntu 22.04 and 24.04, runs the unit tests
(also under ASan/UBSan), runs the complete functional suite on the CPU with
Mesa's lavapipe Vulkan driver, builds the OptiX backend against the CUDA 11.7
redistributables and the optix-dev headers, replays and runs the fuzzers, and
builds the release package in manylinux2014.


Known issues and limitations
----------------------------

* CSG: Tags 1-64 and at most 8 nested shapes enclosing one point; coincident
  flat faces (adjacent layers, boxes flush with the Grid) are handled exactly,
  coincident curved surfaces (e.g. two identical spheres) are not
* coordinates are limited to +-1e6 length units, times to 1e-15-1 s
* trajectories (`-D M`) are saved by the Vulkan backend only
* detected photons/detectors, replay, the adjoint outputs and implicit MMC
  (`MeshROI`) are not implemented
* the voxel grid starts half a voxel below the mesh bounding box, so voxel
  centers lie on the box minimum plus multiples of the voxel size (for a
  `Grid` from 0, at integer coordinates); MCX voxel centers are offset by half
  a voxel
* AddressSanitizer reports a heap-use-after-free inside the Ubuntu 22.04 Vulkan
  loader (`vkCreateInstance`); `tests/fuzz/asan_suppressions.txt` suppresses it


Acknowledgement
---------------

* JSON for Modern C++ by Niels Lohmann (MIT License), `src/nlohmann`
* ZMat by Qianqian Fang (GPL v3) and miniz by Rich Geldreich et al. (MIT
  License), `src/zmat`


Reference
---------

* [Fang2026] Qianqian Fang, Fan-Yu Yen, "How 'vibe-coding' helps build a
  new-generation Monte Carlo simulator over a weekend," Finnish-French
  Workshop on Biomedical Optical Tomography, Mar. 2026, Kuopio, Finland
* [Yen2026] Fan-Yu Yen and Qianqian Fang, "Principal-curvature-informed
  mesh-based Monte Carlo for accurate photon transport modeling across curved
  boundaries," Optical Tomography and Spectroscopy (OT&S), OS3D.7 (2026),
  https://doi.org/10.1364/OTS.2026.OS3D.7
* [Yan2026] Shijie Yan, Douglas Dwyer, David R. Kaeli and Qianqian Fang,
  "Accelerating mesh-based Monte Carlo simulations using contemporary graphics
  ray-tracing hardware," J. Biomed. Opt. 31(3), 035002 (2026),
  https://doi.org/10.1117/1.JBO.31.3.035002
* [Yan2020] Shijie Yan and Qianqian Fang, "Hybrid mesh and voxel based Monte
  Carlo algorithm for accurate and efficient photon transport modeling in
  complex bio-tissues," Biomed. Opt. Express 11(11), 6262-6270 (2020),
  https://opg.optica.org/boe/abstract.cfm?uri=boe-11-11-6262
* [Fang2010] Qianqian Fang, "Mesh-based Monte Carlo method using fast
  ray-tracing in Plücker coordinates," Biomed. Opt. Express 1(1), 165-175 (2010),
  https://opg.optica.org/boe/abstract.cfm?uri=boe-1-1-165

If you use vkmmc in your research, please cite [Fang2026], [Yen2026],
[Yan2026], [Yan2020] and [Fang2010].
