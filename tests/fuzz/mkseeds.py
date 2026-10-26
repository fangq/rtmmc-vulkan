#!/usr/bin/env python3
"""
mkseeds.py - write the seed corpora for the vkmmc fuzzers

  mkseeds.py corpus_input corpus_jdata corpus_cmdline

corpus_input gets small but valid inputs covering every shape type, source
type and mesh encoding; corpus_jdata random JData headers; corpus_cmdline
typical command lines.
"""
import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MEDIA = [{"mua": 0, "mus": 0, "g": 1, "n": 1}, {"mua": 0.01, "mus": 1, "g": 0.9, "n": 1.37},
         {"mua": 0.05, "mus": 5, "g": 0.8, "n": 1.4}, {"mua": 0.1, "mus": 0, "g": 1, "n": 1.5}]

SHAPES = [
    [{"Grid": {"Tag": 1, "Size": [30, 30, 30]}}],
    [{"Sphere": {"Tag": 1, "O": [15, 15, 15], "R": 8}}],
    [{"Box": {"Tag": 1, "O": [5, 5, 5], "Size": [10, 10, 10]}}],
    [{"Grid": {"Tag": 1, "Size": [30, 30, 30]}}, {"Subgrid": {"Tag": 2, "O": [5, 5, 5], "Size": [10, 10, 10]}}],
    [{"Cylinder": {"Tag": 1, "C0": [15, 15, 2], "C1": [15, 15, 28], "R": 5}}],
    [{"Cylinder": {"Tag": 2, "C0": [2, 3, 4], "C1": [20, 25, 18], "R": 3}}],
    [{"Grid": {"Tag": 1, "Size": [30, 30, 30]}}, {"ZLayers": [[1, 10, 1], [11, 20, 2], [21, 30, 3]]}],
    [{"Grid": {"Tag": 1, "Size": [30, 30, 30]}}, {"XSlabs": {"Tag": 2, "Bound": [[5, 10], [15, 20]]}}],
    [{"Grid": {"Tag": 1, "Size": [30, 30, 30]}}, {"YLayers": [1, 15, 2]}],
    [{"Name": "demo"}, {"Origin": [1, 1, 1]}, {"Sphere": {"Tag": 2, "O": [15, 15, 15], "R": 4}}],
    [{"Grid": {"Tag": 1, "Size": [40, 40, 60]}},
     {"Lens": {"O": [20, 20, 20], "Dir": [0, 0, 1], "R": 8, "Tag": 3, "Front": {"D": 2, "R": 15}, "Back": {"D": 2, "R": 15}}}],
    [{"Lens": {"O": [20, 20, 20], "Dir": [1, 1, 0], "R": 6, "Tag": 2, "Front": {"D": 1, "R": -12}, "Back": {"D": 3, "R": 10}}}],
    [{"Grid": {"Tag": 1, "Size": [30, 30, 30]}}, {"Box": {"Tag": 2, "O": [10, 10, 0], "Size": [10, 10, 15]}},
     {"Sphere": {"Tag": 3, "O": [15, 15, 15], "R": 5}}],
]

SOURCES = [
    {"Type": "pencil", "Pos": [15, 15, 0], "Dir": [0, 0, 1]},
    {"Type": "disk", "Pos": [15, 15, 0], "Dir": [0, 0, 1], "Param1": [3, 0, 0, 0]},
    {"Type": "planar", "Pos": [5, 5, 0], "Dir": [0, 0, 1], "Param1": [10, 0, 0], "Param2": [0, 10, 0]},
    {"Type": "pencil", "Pos": [15, 15, 15], "Dir": [1, 1, 1, 0]},
    {"Type": "isotropic", "Pos": [15, 15, 15]},
    {"Type": "cone", "Pos": [15, 15, 0], "Dir": [0, 0, 1], "Param1": [0.5, 0, 0, 0]},
    {"Type": "gaussian", "Pos": [15, 15, 0], "Dir": [0, 0, 1, 10], "Param1": [2, 0.5, 0, 0]},
    {"Type": "pattern", "Pos": [5, 5, 0], "Param1": [10, 0, 0, 2], "Param2": [0, 10, 0, 2],
     "Pattern": {"Nx": 2, "Ny": 2, "Data": [1, 0.5, 0, 0.25]}},
    {"Type": "pattern", "Pos": [5, 5, 0], "Param1": [10, 0, 0, 2], "Param2": [0, 10, 0, 1], "SrcNum": 2,
     "Pattern": {"_ArrayType_": "single", "_ArraySize_": [2, 2, 1], "_ArrayData_": [1, 0, 0.5, 1]}},
    {"Type": "pattern3d", "Pos": [5, 5, 5], "Param1": [2, 1, 2, 0], "Pattern": {"Nx": 2, "Ny": 1, "Nz": 2, "Data": [1, 2, 3, 4]}},
    {"Type": "fourier", "Pos": [5, 5, 0], "Param1": [10, 0, 0, 2.5], "Param2": [0, 10, 0, 1.5]},
    {"Type": "fourierx", "Pos": [5, 5, 0], "Param1": [10, 0, 0, 10], "Param2": [2, 1, 0.25, 0.5]},
    {"Type": "fourierx2d", "Pos": [5, 5, 0], "Param1": [10, 0, 0, 10], "Param2": [2, 1, 0.25, 0.1]},
    {"Type": "arcsine", "Pos": [15, 15, 15]},
    {"Type": "zgaussian", "Pos": [15, 15, 0], "Param1": [0.3, 0, 0, 0]},
    {"Type": "line", "Pos": [5, 15, 15], "Param1": [10, 0, 0, 0], "Param2": [1, 0, 0, 0]},
    {"Type": "slit", "Pos": [5, 15, 0], "Param1": [10, 0, 0, 0], "Param2": [0.1, 0.2, 0, 0]},
    {"Type": "pencilarray", "Pos": [5, 5, 0], "Param1": [10, 0, 0, 3], "Param2": [0, 10, 0, 2]},
    {"Type": "hyperboloid", "Pos": [15, 15, 0], "Param1": [1, 5, 2, 0]},
    {"Type": "ring", "Pos": [15, 15, 0], "Param1": [5, 2, 0, 3], "Dir": [0, 0, 1, "_NaN_"]},
    {"Type": "planar", "Pos": [5, 5, 0], "Dir": [0, 0, 1, "-_Inf_"], "Param1": [10, 0, 0], "Param2": [0, 10, 0]},
    {"Type": "disk", "Pos": [[10, 10, 0, 1], [20, 20, 0, 0.5]], "Dir": [[0, 0, 1, 0], [0, 0.6, 0.8, -5]],
     "Param1": [[2, 0, 0, 0], [0, 0, 0, 0]], "ID": 0},
    {"Type": "pencil", "Pos": [[10, 10, 0], [20, 20, 0]], "Dir": [0, 0, 1], "ID": 2},
]


def base(i):
    return {"Session": {"ID": "seed%d" % i, "Photons": 1000, "RNGSeed": i + 1, "DoMismatch": bool(i % 2),
                        "OutputType": "xfl"[i % 3]},
            "Forward": {"T0": 0, "T1": 5e-9, "Dt": 5e-9 / (1 + i % 3)},
            "Domain": {"LengthUnit": [1, 0.5, 2][i % 3], "MeshRes": [8, 12, 24][i % 3], "Media": MEDIA},
            "Optode": {"Source": SOURCES[i % len(SOURCES)]}}


def main():
    cin, cjd, ccl = sys.argv[1:4]
    for d in (cin, cjd, ccl):
        os.makedirs(d, exist_ok=True)
    n = 0
    for i, sh in enumerate(SHAPES):
        cfg = base(i)
        cfg["Shapes"] = sh
        if i == 3:
            cfg["Domain"]["Step"] = [2, 2, 2]
        json.dump(cfg, open(os.path.join(cin, "shape%02d.json" % i), "w"))
        n += 1
    # mesh inputs in every encoding
    k = 0
    for enc in ("zlib", "gzip", "inline", "data", "base64"):
        for extra in ([], ["--surf"], ["--dtype", "double"], ["--order", "c", "--dtype", "double"]):
            f = os.path.join(cin, "mesh%02d.json" % k)
            subprocess.check_call([sys.executable, os.path.join(HERE, "..", "mkmesh.py"), f, "--size", "20", "20", "20",
                                   "--h", "10", "--enc", enc, "--json", '{"Session":{"Photons":1000}}'] + extra)
            k += 1
    f = os.path.join(cin, "mesh_init.json")
    subprocess.check_call([sys.executable, os.path.join(HERE, "..", "mkmesh.py"), f, "--size", "20", "20", "20", "--h", "10",
                           "--sphere", "10", "10", "10", "6", "--json", '{"Shapes":{"InitElem":3}}'])

    rnd = random.Random(1)
    for i in range(64):
        hdr = bytes([rnd.randrange(256) for _ in range(8)])
        body = bytes([rnd.randrange(256) for _ in range(rnd.randrange(4, 200))])
        open(os.path.join(cjd, "jd%02d" % i), "wb").write(hdr + body)

    cmds = ["in.json", "in.json\n-n\n1e5\n-E\n7", "-f\nin.json\n-b\n0\n-U\n1\n-O\nl", "in.json\n-m\n12\n-c\n1\n-D\nM",
            "in.json\n--maxjumpdebug\n1000\n-t\n4096\n-B\n0", "in.json\n-u\n0.5\n-s\nabc\n-G\n1", "-L", "-v", "-h",
            "in.json\n-j\n{\"Session\":{\"Photons\":10}}", "in.json\n--dumpjson\n--dumpmesh", "in.json\nk.spv",
            "in.json\n-D\n8192\n-O\nflux", "in.json\n-E\n-5\n-n\n9e18"]
    for i, c in enumerate(cmds):
        open(os.path.join(ccl, "cl%02d" % i), "wb").write(c.encode())
    print("seeds: %d inputs, 64 jdata, %d command lines" % (n + k + 1, len(cmds)))


if __name__ == "__main__":
    main()
