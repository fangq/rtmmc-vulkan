#!/usr/bin/env python3
"""
jdat.py - inspect vkmmc output files for the test suite (standard library only)

  jdat.py sum   out.jdat            sum of the Fluence array
  jdat.py dims  out.jdat            Fluence dimensions, e.g. 21x21x21
  jdat.py gates out.jdat            per-time-gate sums, space separated
  jdat.py zprof out.jdat ix iy      Fluence along z at (ix, iy), summed over gates
  jdat.py argmax out.jdat           ix iy iz of the maximum (summed over gates)
  jdat.py gatez out.jdat            per-gate energy-weighted mean iz (-1 for an empty gate)
  jdat.py check out.jdat            validate the JData annotations, print OK
  jdat.py traj  out_traj.jdat       trajectory summary: events photons launches
  jdat.py json  file.json key.path  print a value from a JSON file
"""
import base64
import gzip
import json
import struct
import sys
import zlib

TYPES = {'single': 'f', 'float32': 'f', 'double': 'd', 'float64': 'd', 'int8': 'b', 'uint8': 'B',
         'int16': 'h', 'uint16': 'H', 'int32': 'i', 'uint32': 'I', 'int64': 'q', 'uint64': 'Q'}


def decode(obj):
    """decode a JData array; returns (dims, flat list in row-major order)"""
    dims = obj['_ArraySize_']
    dims = dims if isinstance(dims, list) else [dims]
    n = 1
    for d in dims:
        n *= d
    fmt = TYPES[obj['_ArrayType_']]
    raw = base64.b64decode(obj['_ArrayZipData_'])
    z = obj.get('_ArrayZipType_', 'zlib')
    raw = zlib.decompress(raw) if z == 'zlib' else (gzip.decompress(raw) if z == 'gzip' else raw)
    if len(raw) != n * struct.calcsize(fmt):
        raise ValueError('decoded %d bytes, expected %d' % (len(raw), n * struct.calcsize(fmt)))
    flat = list(struct.unpack('<%d%s' % (n, fmt), raw))
    if obj.get('_ArrayOrder_', 'r')[0] in 'cCfF' and len(dims) > 1:
        # column-major -> row-major
        out = [0] * n
        rstride = [1] * len(dims)
        for d in range(len(dims) - 2, -1, -1):
            rstride[d] = rstride[d + 1] * dims[d + 1]
        idx = [0] * len(dims)
        for v in flat:
            out[sum(i * s for i, s in zip(idx, rstride))] = v
            for d in range(len(dims)):
                idx[d] += 1
                if idx[d] < dims[d]:
                    break
                idx[d] = 0
        flat = out
    return dims, flat


def volume(fname):
    """Fluence as a function v(ix, iy, iz, ig) plus dims (nx, ny, nz, ng)"""
    with open(fname) as f:
        root = json.load(f)
    dims, flat = decode(root['Fluence'])
    nx, ny, nz = dims[:3]
    ng = dims[3] if len(dims) > 3 else 1
    return (lambda ix, iy, iz, ig=0: flat[((ix * ny + iy) * nz + iz) * ng + ig]), (nx, ny, nz, ng), flat, root


def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == 'json':
        with open(args[0]) as f:
            v = json.load(f)
        for k in args[1].split('.'):
            v = v[int(k)] if isinstance(v, list) else v[k]
        print(json.dumps(v) if isinstance(v, (dict, list)) else v)
        return
    if cmd == 'traj':
        with open(args[0]) as f:
            tr = json.load(f)['MCXData']['Trajectory']
        _, ids = decode(tr['photonid'])
        _, ev = decode(tr['e0'])
        _, pos = decode(tr['p'])
        assert len(pos) == 3 * len(ids) and len(ev) == len(ids)
        print(len(ids), len(set(ids)), sum(1 for e in ev if e == 0))
        return
    v, (nx, ny, nz, ng), flat, root = volume(args[0])
    if cmd == 'sum':
        print('%.9g' % sum(flat))
    elif cmd == 'dims':
        full = decode(root['Fluence'])[0]   # [nx, ny, nz(, nt)] or [nx, ny, nz, nt, nfield]
        print('x'.join(str(d) for d in (full if len(full) > 4 else [nx, ny, nz] + ([ng] if ng > 1 else []))))
    elif cmd == 'gates':
        print(' '.join('%.9g' % sum(v(ix, iy, iz, g) for ix in range(nx) for iy in range(ny) for iz in range(nz))
                       for g in range(ng)))
    elif cmd == 'zprof':
        ix, iy = int(args[1]), int(args[2])
        print(' '.join('%.6g' % sum(v(ix, iy, iz, g) for g in range(ng)) for iz in range(nz)))
    elif cmd == 'argmax':
        best = max(((sum(v(ix, iy, iz, g) for g in range(ng)), ix, iy, iz)
                    for ix in range(nx) for iy in range(ny) for iz in range(nz)))
        print(best[1], best[2], best[3])
    elif cmd == 'gatez':
        out = []
        for g in range(ng):
            w = [sum(v(ix, iy, iz, g) for ix in range(nx) for iy in range(ny)) for iz in range(nz)]
            tot = sum(w)
            out.append('%.4f' % (sum(i * x for i, x in enumerate(w)) / tot) if tot > 0 else '-1')
        print(' '.join(out))
    elif cmd == 'check':
        fl = root['Fluence']
        assert fl['_ArrayType_'] == 'single', 'type'
        assert fl.get('_ArrayOrder_') == 'c', 'order'
        assert root['Domain']['Dim'] == [nx, ny, nz], 'Dim'
        assert all(x == x for x in flat), 'NaN'
        print('OK')
    else:
        sys.exit('unknown command ' + cmd)


if __name__ == '__main__':
    main()
