#!/usr/bin/env python3
"""
mkmesh.py - generate mesh-mode vkmmc input files for the test suite

A box [0,X]x[0,Y]x[0,Z] is split into cubes of size h, each split into 6
tetrahedra (Kuhn subdivision, conforming across cubes). Optionally the cubes
whose centers fall inside a sphere get a second label.

The mesh arrays can be written in every encoding vkmmc accepts, so that the
JData decoder can be checked against a reference:
  --enc zlib   (float32/int32, zlib, default)   --enc gzip
  --enc inline (plain JSON arrays)              --enc data (uncompressed _ArrayData_)
  --enc base64 (_ArrayZipType_ base64)
  --dtype double      store MeshNode as float64 and MeshElem as uint32
  --order c           store column-major data with _ArrayOrder_ "c"
  --surf              store MeshSurf (triangles) instead of MeshElem

Usage: mkmesh.py out.json [--size X Y Z] [--h h] [--sphere cx cy cz r]
                 [--enc ...] [--dtype ...] [--order ...] [--surf] [--json '{...}']
Only the Python standard library is needed.
"""
import argparse
import base64
import gzip
import json
import struct
import zlib

# 6 tets per cube along the main diagonal (corners indexed by bits x|y<<1|z<<2)
KUHN = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]


def build(size, h, sphere):
    nx, ny, nz = (int(round(s / h)) for s in size)
    node = lambda i, j, k: i + j * (nx + 1) + k * (nx + 1) * (ny + 1)
    nodes = [(i * h, j * h, k * h) for k in range(nz + 1) for j in range(ny + 1) for i in range(nx + 1)]
    elems = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                c = [node(i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1)) for b in range(8)]
                label = 1
                if sphere:
                    cx, cy, cz = (i + 0.5) * h, (j + 0.5) * h, (k + 0.5) * h
                    if (cx - sphere[0]) ** 2 + (cy - sphere[1]) ** 2 + (cz - sphere[2]) ** 2 <= sphere[3] ** 2:
                        label = 2
                for t in KUHN:
                    elems.append([c[t[0]] + 1, c[t[1]] + 1, c[t[2]] + 1, c[t[3]] + 1, label])
    return nodes, elems


def surface(nodes, elems):
    """triangles between different labels, listed once per side (MeshSurf rows);
    each row is wound so that its normal points out of the row's label region"""
    faces = {}
    for e in elems:
        v, lab = e[:4], e[4]
        for f, opp in (((0, 1, 2), 3), ((0, 1, 3), 2), ((0, 2, 3), 1), ((1, 2, 3), 0)):
            tri = [v[i] for i in f]
            p = [nodes[i - 1] for i in tri]
            q = nodes[v[opp] - 1]
            e1 = [p[1][k] - p[0][k] for k in range(3)]
            e2 = [p[2][k] - p[0][k] for k in range(3)]
            n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]]
            if sum(n[k] * (q[k] - p[0][k]) for k in range(3)) > 0:  # normal points into the tet: flip
                tri = [tri[0], tri[2], tri[1]]
            faces.setdefault(tuple(sorted(tri)), []).append(tri + [lab])
    rows = []
    for key, ent in faces.items():
        if len(ent) == 1 or ent[0][3] != ent[1][3]:
            rows.extend(ent)
    return rows


def encode(rows, kind, enc, order):
    """encode a 2-D table as a JData object (or an inline list)"""
    if enc == 'inline':
        return rows
    nr, nc = len(rows), len(rows[0])
    flat = [x for r in rows for x in r] if order != 'c' else [rows[i][j] for j in range(nc) for i in range(nr)]
    fmt = {'single': 'f', 'double': 'd', 'int32': 'i', 'uint32': 'I'}[kind]
    raw = struct.pack('<%d%s' % (len(flat), fmt), *flat)
    obj = {'_ArrayType_': kind, '_ArraySize_': [nr, nc]}
    if order == 'c':
        obj['_ArrayOrder_'] = 'c'
    if enc == 'data':
        obj['_ArrayData_'] = flat
        return obj
    if enc == 'zlib':
        payload, ztype = zlib.compress(raw), 'zlib'
    elif enc == 'gzip':
        payload, ztype = gzip.compress(raw), 'gzip'
    else:
        payload, ztype = raw, 'base64'
    obj.update({'_ArrayZipType_': ztype, '_ArrayZipSize_': [nr, nc],
                '_ArrayZipData_': base64.b64encode(payload).decode()})
    return obj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out')
    ap.add_argument('--size', nargs=3, type=float, default=[60, 60, 60])
    ap.add_argument('--h', type=float, default=10)
    ap.add_argument('--sphere', nargs=4, type=float)
    ap.add_argument('--enc', default='zlib', choices=['zlib', 'gzip', 'inline', 'data', 'base64'])
    ap.add_argument('--dtype', default='single', choices=['single', 'double'])
    ap.add_argument('--order', default='r', choices=['r', 'c'])
    ap.add_argument('--surf', action='store_true')
    ap.add_argument('--json', default='{}', help='JSON merged into the generated input')
    a = ap.parse_args()

    nodes, elems = build(a.size, a.h, a.sphere)
    ikind = 'uint32' if a.dtype == 'double' else 'int32'
    shapes = {'MeshNode': encode([list(n) for n in nodes], a.dtype, a.enc, a.order)}
    if a.surf:
        shapes['MeshSurf'] = encode(surface(nodes, elems), ikind, a.enc, a.order)
    else:
        shapes['MeshElem'] = encode(elems, ikind, a.enc, a.order)

    cfg = {
        'Session': {'ID': a.out.rsplit('.', 1)[0], 'Photons': 100000, 'DoMismatch': False},
        'Forward': {'T0': 0, 'T1': 5e-9, 'Dt': 5e-9},
        'Domain': {'LengthUnit': 1, 'Media': [{'mua': 0, 'mus': 0, 'g': 1, 'n': 1},
                                              {'mua': 0.01, 'mus': 1, 'g': 0.9, 'n': 1},
                                              {'mua': 0.05, 'mus': 5, 'g': 0.9, 'n': 1}]},
        'Optode': {'Source': {'Type': 'pencil', 'Pos': [a.size[0] / 2, a.size[1] / 2, 0.5], 'Dir': [0, 0, 1]}},
        'Shapes': shapes,
    }

    def merge(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                merge(dst[k], v)
            else:
                dst[k] = v

    merge(cfg, json.loads(a.json))
    with open(a.out, 'w') as f:
        json.dump(cfg, f)


if __name__ == '__main__':
    main()
