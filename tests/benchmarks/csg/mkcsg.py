#!/usr/bin/env python3
"""Write vkmmc CSG inputs equivalent to the mmc examples whose meshes are built
from boxes, spheres, cylinders and layers with iso2mesh."""

import json, os, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "."


def med(rows):
    return [{"mua": r[0], "mus": r[1], "g": r[2], "n": r[3]} for r in [[0, 0, 1, 1]] + rows]


def case(name, photons, seed, t, media, src, shapes, mismatch=False, note=""):
    cfg = {
        "Session": {"ID": name, "Photons": photons, "RNGSeed": seed, "DoMismatch": bool(mismatch)},
        "Forward": {"T0": t[0], "T1": t[1], "Dt": t[2]},
        "Domain": {"LengthUnit": 1, "Media": med(media)},
        "Optode": {"Source": src},
        "Shapes": shapes,
    }
    if note:
        cfg = {"Comment": note, **cfg}
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(cfg, f, indent=4)
        f.write("\n")


box60 = {"Grid": {"Tag": 1, "Size": [60, 60, 60]}}

# examples/validation, mmclab demo_example_validation, demo_mmcl_b1_b1d (B1)
case("cube60", 1000000, 27182818, [0, 5e-9, 1e-10], [[0.005, 1.0, 0.01, 1.0]],
     {"Type": "pencil", "Pos": [30.1, 30.2, 0], "Dir": [0, 0, 1]}, [box60],
     note="mmc examples/validation (cube.inp, meshgrid6/tessmesh box)")

# examples/meshtest, mmclab demo_example_meshtest, demo_dualmesh_output
case("meshtest", 1000000, 27182820, [0, 5e-9, 1e-10],
     [[0.002, 1.0, 0.01, 1.37], [0.050, 5.0, 0.9, 1.37]],
     {"Type": "pencil", "Pos": [30, 30, 0], "Dir": [0, 0, 1]},
     [box60, {"Sphere": {"Tag": 2, "O": [30, 30, 30], "R": 10}}],
     note="mmc examples/meshtest (meshasphere R10 + box, surf2mesh)")

# examples/sphshells, mmclab demo_dmmc_sphshells, demo_mmcl_b2_b2d (B2)
case("sphshells", 1000000, 1648335518, [0, 5e-9, 5e-10],
     [[0.02, 7.0, 0.89, 1.37], [0.004, 0.009, 0.89, 1.37], [0.02, 9.0, 0.89, 1.37], [0.05, 0.0, 1.0, 1.37]],
     {"Type": "pencil", "Pos": [30, 30.1, 0], "Dir": [0, 0, 1]},
     [box60, {"Sphere": {"Tag": 2, "O": [30, 30, 30], "R": 25}},
      {"Sphere": {"Tag": 3, "O": [30, 30, 30], "R": 23}},
      {"Sphere": {"Tag": 4, "O": [30, 30, 30], "R": 10}}], mismatch=True,
     note="mmc examples/sphshells (3 meshasphere + box, mergemesh/surf2mesh)")

# examples/optix, mmclab demo_optix_mmc
case("optixsph", 1000000, 1648335518, [0, 5e-9, 5e-10],
     [[0.005, 1, 0, 1.37], [0.01, 10, 0, 1.0]],
     {"Type": "pencil", "Pos": [30, 30, 0.01], "Dir": [0, 0, 1]},
     [box60, {"Sphere": {"Tag": 2, "O": [30, 30, 30], "R": 25}}], mismatch=True,
     note="mmc examples/optix (meshasphere R25 + box, mergemesh/surf2mesh)")

# mmclab demo_compare_mmc_mcx (inclusion=1), pencil and planar
for typ in ["pencil", "planar"]:
    src = {"Type": typ, "Pos": [30.1, 30.2, 0], "Dir": [0, 0, 1]}
    if typ == "planar":
        src.update({"Pos": [10, 10, -10], "Param1": [40, 0, 0, 0], "Param2": [0, 40, 0, 0]})
    case("boxinbox_" + typ, 1000000, 1648335518, [0, 5e-9, 1e-10],
         [[0.005, 1.0, 0.01, 1.0], [0.01, 0.2, 0.01, 1.37]], src,
         [box60, {"Box": {"Tag": 2, "O": [20, 20, 20], "Size": [20, 20, 20]}}], mismatch=True,
         note="mmclab demo_compare_mmc_mcx (two meshabox, mergemesh/surf2mesh)")

# examples/sfdi2layer, mmclab demo_sfdi_2layer (fourier source above the slab)
case("sfdi2layer", 1000000, 1648335518, [0, 5e-9, 1e-10],
     [[0.02, 9.0, 0.89, 1.37], [0.019, 7.8, 0.89, 1.37]],
     {"Type": "fourier", "Pos": [10, 10, 35], "Dir": [0, 0, -1],
      "Param1": [40, 0, 0, 3 + 1 / 6], "Param2": [0, 40, 0, 0]},
     [{"Box": {"Tag": 1, "O": [0, 0, 0], "Size": [60, 60, 25]}},
      {"Box": {"Tag": 2, "O": [0, 0, 25], "Size": [60, 60, 5]}}],
     note="mmc examples/sfdi2layer (latticegrid z=[0 25 30], mmcaddsrc)")

# examples/widedet, mmclab demo_wide_det (the planar detector is not modeled)
case("widedet", 1000000, 12345678, [0, 2e-9, 4e-11],
     [[0.01, 1.0, 0.01, 1.37], [0.05, 10.0, 0.9, 1.37]],
     {"Type": "pencil", "Pos": [25, 35, 10], "Dir": [0, 0, -1]},
     [{"Box": {"Tag": 1, "O": [0, 0, 0], "Size": [60, 60, 5]}},
      {"Box": {"Tag": 2, "O": [0, 0, 5], "Size": [60, 60, 5]}}], mismatch=True,
     note="mmc examples/widedet (latticegrid z=[0 5 10]); the wide-field detector is not modeled")

# examples/skinvessel, mmclab demo_mcxyz_skinvessel, demo_mmcl_b4 (B4), in mm;
# the media are renumbered so that the vessel has the highest Tag
case("skinvessel", 1000000, 1648335518, [0, 5e-8, 5e-8],
     [[1.6572, 37.594, 0.9, 1.37], [0.0458, 35.6541, 0.9, 1.37], [23.0543, 9.3985, 0.9, 1.37]],
     {"Type": "disk", "Pos": [0.5, 0.5, -0.005], "Dir": [0, 0, 1], "Param1": [0.3, 0, 0, 0]},
     [{"Box": {"Tag": 1, "O": [0, 0, 0.1], "Size": [1, 1, 0.06]}},
      {"Box": {"Tag": 2, "O": [0, 0, 0.16], "Size": [1, 1, 0.84]}},
      {"Cylinder": {"Tag": 3, "C0": [0, 0.4975, 0.4975], "C1": [1, 0.4975, 0.4975], "R": 0.1}}],
     note="mmc examples/skinvessel (latticegrid + meshacylinder, surfboolean/s2m); media 1/2/3 = mmc labels 4/3/2")

# examples/sphere (meshasphere R24 at the origin, planar source above it)
case("sphere24", 1000000, 1648335518, [0, 5e-9, 1e-10], [[0.002, 1.0, 0.01, 1.37]],
     {"Type": "planar", "Pos": [-5, -5, 25], "Dir": [0, 0, -1], "Param1": [10, 0, 0, 0], "Param2": [0, 10, 0, 0]},
     [{"Sphere": {"Tag": 1, "O": [0, 0, 0], "R": 24}}],
     note="mmc examples/sphere (meshasphere R24, mmcaddsrc/mmcadddet); the planar detector is not modeled")

# mmclab demo_mmclab_basic/_slit/_multisrc/_mesh_adjoint (meshabox 60x60x30)
case("slab60x30", 1000000, 1648335518, [0, 5e-9, 1e-10], [[0.005, 1.0, 0.01, 1.0]],
     {"Type": "pencil", "Pos": [30.1, 30.2, 0], "Dir": [0, 0, 1]},
     [{"Grid": {"Tag": 1, "Size": [60, 60, 30]}}],
     note="mmclab demo_mmclab_basic (meshabox [0 0 0]-[60 60 30])")
