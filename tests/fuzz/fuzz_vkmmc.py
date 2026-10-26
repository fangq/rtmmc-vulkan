#!/usr/bin/env python3
"""
fuzz_vkmmc.py - JSON-aware mutation fuzzing of the vkmmc / vkmmc_optix binaries

Each iteration takes a seed input, applies a few structure-aware mutations
(change, delete, duplicate or retype values; swap shapes; insert known keys
with extreme values) plus random command-line options, and runs the binary.

  host mode (default): adds --dumpjson so the run stops before the GPU; use an
                       AddressSanitizer build (make -C src asan) to catch memory
                       errors in the whole front-end, incl. option handling
  gpu mode (--gpu):    runs a small simulation (-n <= 2000) on the GPU and
                       also checks the output volume (finite, absorbed <= 100%)

A run that exits with 0, or with 1 and a "ERROR:" message, is fine. A signal,
a sanitizer report, any other exit code, a timeout, a missing or non-finite
output, or an absorbed fraction outside [0, 100] is a finding; the input and
command line are saved to the findings directory.

  fuzz_vkmmc.py EXEC [--gpu] [-n ITER] [--seeds DIR] [--out DIR] [--timeout S] [--seed S]
Only the Python standard library is needed.
"""
import argparse
import copy
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import jdat  # noqa: E402

KEYS = ['Session', 'ID', 'Photons', 'RNGSeed', 'DoMismatch', 'DoSpecular', 'DoNormalize', 'OutputType', 'RussianRoulette',
        'Forward', 'T0', 'T1', 'Dt', 'Domain', 'LengthUnit', 'Media', 'mua', 'mus', 'g', 'n', 'Step', 'MeshRes',
        'Optode', 'Source', 'Type', 'Pos', 'Dir', 'Param1', 'Param2', 'Pattern', 'SrcNum', 'Nx', 'Ny', 'Nz',
        'Data', 'Shapes', 'Tag', 'O', 'R', 'Size', 'C0', 'C1',
        'Bound', 'Front', 'Back', 'D', 'MeshNode', 'MeshElem', 'MeshSurf', 'InitElem', '_ArrayType_', '_ArraySize_',
        '_ArrayZipType_', '_ArrayZipSize_', '_ArrayZipData_', '_ArrayOrder_', '_ArrayData_', '_ArrayZipOffsets_']
SHAPE_TYPES = ['Grid', 'Subgrid', 'Sphere', 'Box', 'Cylinder', 'XSlabs', 'YSlabs', 'ZSlabs', 'XLayers', 'ZLayers',
               'Lens', 'Name', 'Origin', 'Torus']
NUMBERS = [0, -0.0, 1, -1, 0.5, 2, 3, 7, 64, 65, 255, 256, 1e-300, 1e-45, 1e-7, 1e6, 1e7, 3.5e38, 1e39, 1e308,
           -1e308, 2 ** 31 - 1, 2 ** 31, 2 ** 32, 2 ** 53 + 1, 2 ** 63, 2 ** 64, -2 ** 63]
STRINGS = ['', 'pencil', 'disk', 'planar', 'cone', 'isotropic', 'pattern', 'pattern3d', 'fourierx', 'line', 'slit',
           'ring', 'hyperboloid', '_NaN_', '_Inf_', '-_Inf_', 'x', 'f', 'l', 'fluence', 'zlib', 'gzip', 'c', 'r', 'single',
           'double', 'uint8', 'bogus', 'A' * 300, '../../etc/passwd', 'é中']
OPTIONS = [['-n', '100'], ['-n', '1e18'], ['-n', '0'], ['-E', '-1'], ['-E', '0'], ['-b', '0'], ['-b', '1'], ['-V', '1'], ['-V', '2'],
           ['-U', '0'], ['-O', 'f'], ['-O', 'l'], ['-O', 'q'], ['-t', '64'], ['-t', '65'], ['-t', '16777216'],
           ['-B', '0'], ['-B', '1'], ['-m', '1'], ['-m', '4'], ['-m', '2048'], ['-m', '5000'], ['-c', '0'],
           ['-c', '1'], ['-c', '2'], ['-u', '0.001'], ['-u', '1e7'], ['-D', 'M'], ['-D', 'Q'], ['-D', '8192'],
           ['--maxjumpdebug', '10'], ['--atomic', 'float'], ['--atomic', 'x'], ['-G', '99'], ['-s', 'x' * 300], ['-j', '{"Session":{"Photons":5}}'],
           ['-j', '{bad'], ['--bogus'], ['-n'], ['--dumpmesh'], ['--debugcurv'], ['k.spv'], ['-L']]


def walk(node, path=()):
    """yield (container, key) of every value in the document"""
    if isinstance(node, dict):
        for k in list(node):
            yield node, k
            yield from walk(node[k], path + (k,))
    elif isinstance(node, list):
        for i in range(len(node)):
            yield node, i
            yield from walk(node[i], path + (i,))


def rand_value(rnd, depth=0):
    r = rnd.random()
    if r < 0.45:
        return rnd.choice(NUMBERS) * rnd.choice([1, 1, -1, 0.999])
    if r < 0.6:
        return rnd.choice(STRINGS)
    if r < 0.7 and depth < 3:
        return [rand_value(rnd, depth + 1) for _ in range(rnd.randrange(0, 6))]
    if r < 0.8 and depth < 3:
        return {rnd.choice(KEYS): rand_value(rnd, depth + 1) for _ in range(rnd.randrange(0, 4))}
    return rnd.choice([None, True, False, []])


def mutate(doc, rnd):
    doc = copy.deepcopy(doc)
    for _ in range(rnd.choice([1, 1, 1, 2, 2, 3, 4])):
        slots = list(walk(doc))
        if not slots:
            return rand_value(rnd)
        cont, key = rnd.choice(slots)
        v = cont[key]
        op = rnd.random()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and op < 0.5:
            cont[key] = rnd.choice([v * 10 ** rnd.randrange(-6, 7), -v, v + rnd.choice([-1, 1]), rnd.choice(NUMBERS),
                                    int(v) if math.isfinite(v) else 0, str(v)])
        elif op < 0.6:
            del cont[key]
        elif op < 0.7 and isinstance(cont, list):
            cont.insert(key, copy.deepcopy(v))
        elif op < 0.75 and isinstance(cont, list) and len(cont) > 1:
            rnd.shuffle(cont)
        elif op < 0.82 and isinstance(cont, dict):
            cont[rnd.choice(KEYS)] = rand_value(rnd)
        elif op < 0.87 and isinstance(cont, list) and isinstance(v, dict) and len(v) == 1:
            # change the shape type, keep its parameters
            cont[key] = {rnd.choice(SHAPE_TYPES): list(v.values())[0]}
        elif op < 0.92 and isinstance(v, str) and len(v) > 8:
            # corrupt a string (e.g. base64 payloads)
            i = rnd.randrange(len(v))
            cont[key] = v[:i] + rnd.choice(['', '=', 'A', '\n', '*', v[i:i + 40]]) + v[i + 1:]
        else:
            cont[key] = rand_value(rnd)
    return doc


def check_output(tmp, sid):
    """the output volume must exist, decode and be finite"""
    f = os.path.join(tmp, sid + '.jdat')
    if not os.path.exists(f):
        return 'no output file'
    try:
        with open(f) as fp:
            root = json.load(fp)
        _, flat = jdat.decode(root['Fluence'])
    except Exception as e:  # noqa: BLE001
        return 'unreadable output: %s' % e
    if not all(math.isfinite(x) for x in flat):
        return 'non-finite output'
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('exe')
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('-n', type=int, default=1000)
    ap.add_argument('--seeds', default=os.path.join(HERE, 'corpus_input'))
    ap.add_argument('--out', default=os.path.join(HERE, 'findings_vkmmc'))
    ap.add_argument('--timeout', type=float, default=60)
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    a.exe = os.path.abspath(a.exe)

    rnd = random.Random(a.seed)
    seeds = []
    for f in sorted(os.listdir(a.seeds)):
        try:
            with open(os.path.join(a.seeds, f)) as fp:
                seeds.append(json.load(fp))
        except Exception:  # noqa: BLE001
            pass
    if not seeds:
        sys.exit('no JSON seeds in ' + a.seeds)
    os.makedirs(a.out, exist_ok=True)
    supp = os.path.join(HERE, 'asan_suppressions.txt')
    env = dict(os.environ, ASAN_OPTIONS='detect_leaks=0:abort_on_error=0:exitcode=99:suppressions=' + supp,
               UBSAN_OPTIONS='print_stacktrace=1:halt_on_error=1:exitcode=99')
    stats = {'ok': 0, 'rejected': 0, 'finding': 0}
    tmp = tempfile.mkdtemp(prefix='vkmmc_fuzz_')
    try:
        for it in range(a.n):
            doc = mutate(rnd.choice(seeds), rnd)
            sid = 'f%d' % it
            if isinstance(doc, dict) and isinstance(doc.get('Session'), dict):
                doc['Session']['ID'] = sid
            inp = os.path.join(tmp, 'in.json')
            with open(inp, 'w') as fp:
                json.dump(doc, fp)
            # -L only enumerates the GPUs (slow under ASan): GPU mode only
            pool = OPTIONS if a.gpu else [o for o in OPTIONS if o != ['-L']]
            opts = [x for _ in range(rnd.choice([0, 0, 1, 2, 3])) for x in rnd.choice(pool)]
            cmd = [a.exe, inp, '-s', sid] + opts
            if a.gpu:
                cmd += ['-n', str(rnd.choice([1, 63, 64, 1000, 2000]))]
            else:
                cmd += ['--dumpjson']
            try:
                p = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, timeout=a.timeout)
                out = p.stdout.decode(errors='replace') + p.stderr.decode(errors='replace')
                rc = p.returncode
            except subprocess.TimeoutExpired:
                out, rc = 'TIMEOUT', None
            problem = None
            if rc is None:
                problem = 'timeout'
            elif rc < 0 or rc not in (0, 1) or 'Sanitizer' in out or 'runtime error' in out:
                problem = 'exit %s' % rc
            elif rc == 1 and 'ERROR' not in out:
                problem = 'exit 1 without an error message'
            elif rc == 0 and a.gpu and '--dumpjson' not in opts and '--dumpmesh' not in opts and \
                    '--debugcurv' not in opts and '-L' not in opts:
                problem = check_output(tmp, sid)
                m = re.search(r'absorbed:\s*([0-9.eE+-]+)%', out)
                if not problem and m and not (0 <= float(m.group(1)) <= 100.0001):
                    problem = 'absorbed %s%%' % m.group(1)
            if problem:
                stats['finding'] += 1
                base = os.path.join(a.out, '%s-%04d' % (re.sub(r'\W+', '_', problem)[:24], it))
                shutil.copy(inp, base + '.json')
                with open(base + '.txt', 'w') as fp:
                    fp.write(' '.join(cmd[2:]) + '\n\n' + out[-20000:])
                print('[%d] FINDING %s: %s' % (it, problem, ' '.join(opts)), flush=True)
            else:
                stats['ok' if rc == 0 else 'rejected'] += 1
            for f in os.listdir(tmp):
                if f != 'in.json':
                    os.remove(os.path.join(tmp, f))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('runs: %d  ok: %d  rejected: %d  findings: %d' % (a.n, stats['ok'], stats['rejected'], stats['finding']))
    sys.exit(1 if stats['finding'] else 0)


if __name__ == '__main__':
    main()
