#!/bin/bash
# smoke.sh binary: front-end checks that need no GPU (version, input parsing,
# mesh generation and the error path), for CI runners without a GPU
set -e
bin=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
inp=tests/fuzz/regressions/input-flat-lens-cap-timeout-1436.json
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

"$bin" -v
"$bin" "$inp" --dumpjson -m 48 > "$tmp/dump.txt"
grep -q '"Faces"' "$tmp/dump.txt"
(cd "$tmp" && "$bin" "$OLDPWD/$inp" --dumpmesh -s mesh > /dev/null && test -s mesh_mesh.json)
"$bin" "$tmp/mesh_mesh.json" --dumpjson > /dev/null

if "$bin" "$tmp/missing.json" > "$tmp/err.txt" 2>&1; then
    echo "ERROR: a missing input did not fail"
    exit 1
fi

grep -q "ERROR" "$tmp/err.txt"
echo "smoke test of $bin passed"
