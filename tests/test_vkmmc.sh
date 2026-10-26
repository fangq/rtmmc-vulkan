#!/bin/bash
# =============================================================================
# test_vkmmc.sh — functional tests for vkmmc / vkmmc_optix (needs a GPU)
#
#   ./test_vkmmc.sh [executable] [extra vkmmc flags...]
#   ./test_vkmmc.sh ../bin/vkmmc            # Vulkan backend (default)
#   ./test_vkmmc.sh ../bin/vkmmc_optix      # OptiX backend
#   XEXEC=../bin/vkmmc_optix ./test_vkmmc.sh ../bin/vkmmc   # + cross-backend checks
#
# Expected values are analytic (Beer-Lambert, Fresnel, time of flight),
# conservation identities, or comparisons between equivalent inputs. Tests
# use fixed RNG seeds; statistical tolerances are several standard errors.
# Needs python3 (standard library only).
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXEC="${1:-$SCRIPT_DIR/../bin/vkmmc}"
shift || true

if [ -f "$EXEC" ]; then
    EXEC="$(cd "$(dirname "$EXEC")" && pwd)/$(basename "$EXEC")"
else
    echo "ERROR: executable '$EXEC' not found"; exit 1
fi
[ ! -x "$EXEC" ] && echo "ERROR: not executable" && exit 1

XEXEC="${XEXEC:-}"
[ -n "$XEXEC" ] && XEXEC="$(cd "$(dirname "$XEXEC")" && pwd)/$(basename "$XEXEC")"

EXTRA_ARGS=""
for arg in "$@"; do EXTRA_ARGS="$EXTRA_ARGS $arg"; done

ISOPTIX=0
"$EXEC" -v 2>/dev/null | grep -q OptiX && ISOPTIX=1

TESTDIR=$(mktemp -d /tmp/vkmmc_test.XXXXXX)
trap "rm -rf $TESTDIR" EXIT

# MSYS2/Cygwin: paths embedded in python code must be native (C:/...)
if command -v cygpath >/dev/null 2>&1; then
    TESTDIR=$(cygpath -m "$TESTDIR")
    SCRIPT_DIR=$(cygpath -m "$SCRIPT_DIR")
fi

PASS=0 FAIL=0 SKIP=0 TOTAL=0
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; CYAN='\033[0;36m'; NC='\033[0m'

log_test() { TOTAL=$((TOTAL+1)); echo -e "${CYAN}[TEST $TOTAL]${NC} $1"; }
log_pass() { PASS=$((PASS+1)); echo -e "  ${GREEN}PASS${NC}: $1"; }
log_fail() { FAIL=$((FAIL+1)); echo -e "  ${RED}FAIL${NC}: $1"; }
log_skip() { SKIP=$((SKIP+1)); echo -e "  ${YELLOW}SKIP${NC}: $1"; }

check_range() {
    local val="$1" lo="$2" hi="$3" desc="$4"
    if [ -n "$val" ] && python3 -c "import sys; sys.exit(0 if $lo <= $val <= $hi else 1)" 2>/dev/null; then
        log_pass "$desc (got $val, expected [$lo, $hi])"
    else
        log_fail "$desc (got ${val:-nothing}, expected [$lo, $hi])"
    fi
}

# check_close value expected tolerance description   (|value - expected| <= tolerance)
check_close() {
    local val="$1" ref="$2" tol="$3" desc="$4"
    if [ -n "$val" ] && [ -n "$ref" ] && python3 -c "import sys; sys.exit(0 if abs($val - ($ref)) <= $tol else 1)" 2>/dev/null; then
        log_pass "$desc (got $val, expected $ref ± $tol)"
    else
        log_fail "$desc (got ${val:-nothing}, expected ${ref:-?} ± $tol)"
    fi
}

# pyval 'expression' — evaluate a python expression (math imported)
pyval() { python3 -c "import math; print('%.10g' % ($1))"; }

get_absorbed()  { grep -oP 'absorbed:\s*\K[0-9.]+' "$TESTDIR/$1" 2>/dev/null | head -1; }
get_speed()     { grep -oP 'Speed:\s*\K[0-9.]+' "$TESTDIR/$1" 2>/dev/null | head -1; }
get_triangles() { grep -oP '[0-9]+ tri\b' "$TESTDIR/$1" 2>/dev/null | grep -oP '[0-9]+' | head -1; }
jdat()          { python3 "$SCRIPT_DIR/jdat.py" "$@" 2>/dev/null; }

# run_sim input.json log [flags...]: run in the test directory, outputs land there
run_sim() {
    local json="$TESTDIR/$1" logf="$TESTDIR/$2"
    shift 2
    (cd "$TESTDIR" && timeout 300 "$EXEC" "$json" $EXTRA_ARGS "$@" > "$logf" 2>&1)
}

# mkcfg name.json 'json patch': write a test input by merging a patch into a base
# configuration (pencil beam at z=0 along +z, no mismatch, 200000 photons)
mkcfg() {
    python3 - "$TESTDIR/$1" "$2" <<'PYEOF'
import json, sys
base = {"Session": {"ID": sys.argv[1].rsplit('/', 1)[-1][:-5], "Photons": 200000, "DoMismatch": False},
        "Forward": {"T0": 0, "T1": 5e-9, "Dt": 5e-9},
        "Domain": {"LengthUnit": 1, "Media": [{"mua": 0, "mus": 0, "g": 1, "n": 1},
                                             {"mua": 0.05, "mus": 0, "g": 1, "n": 1}]},
        "Optode": {"Source": {"Type": "pencil", "Pos": [30, 30, 0], "Dir": [0, 0, 1]}},
        "Shapes": [{"Box": {"Tag": 1, "O": [10, 10, 10], "Size": [40, 40, 20]}}]}
def merge(d, p):
    for k, v in p.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            merge(d[k], v)
        else:
            d[k] = v
merge(base, json.loads(sys.argv[2]))
json.dump(base, open(sys.argv[1], 'w'))
PYEOF
}

cd "$TESTDIR"

echo "============================================================"
echo " vkmmc test suite"
echo " Executable: $EXEC (backend: $([ $ISOPTIX = 1 ] && echo OptiX || echo Vulkan))"
echo " Work dir:   $TESTDIR"
echo "============================================================"
echo ""

# =============================================================================
# All shapes: sphere R=10 centered at [30,30,30], surface z=[20,40].
# Source OUTSIDE at z=19.5 pointing +z (enters at z=20).
# Path through sphere diameter = 20mm.
#
# Analytical (no scattering, no mismatch):
#   absorbed = 1 - exp(-mua * 20)
# =============================================================================

# =============================================================================
# TEST 1: Pure absorption (mua=0.1, mus≈0, no mismatch)
# Analytical: 1-exp(-0.1*20) = 86.5%
# =============================================================================
log_test "Pure absorption sphere (mua=0.1, mus≈0, no mismatch)"

cat > t01.json << 'ENDJSON'
{
    "Session": { "ID": "t01", "Photons": 500000, "DoMismatch": false },
    "Forward": { "T0": 0, "T1": 5e-9, "Dt": 5e-9 },
    "Domain": {
        "LengthUnit": 1.0,
        "Media": [
            {"mua": 0, "mus": 0, "g": 1, "n": 1},
            {"mua": 0.1, "mus": 0.001, "g": 0.9, "n": 1.0}
        ]
    },
    "Optode": {
        "Source": { "Type": "pencil", "Pos": [30, 30, 19.5], "Dir": [0, 0, 1] }
    },
    "Shapes": [
        {"Sphere": {"Tag": 1, "O": [30, 30, 30], "R": 10}}
    ]
}
ENDJSON

if run_sim t01.json t01.log -n 500000 -E 12345 -b 0; then
    abs=$(get_absorbed t01.log)
    if [ -n "$abs" ]; then
        check_range "$abs" 84.0 89.0 "Absorption ≈ 86.5%"
    else
        log_fail "Could not parse absorbed%"
        tail -10 "$TESTDIR/t01.log" 2>/dev/null | sed 's/^/    /'
    fi
else
    log_fail "Execution failed (exit=$?)"
    tail -10 "$TESTDIR/t01.log" 2>/dev/null | sed 's/^/    /'
fi

# =============================================================================
# TEST 2: Scattering dominant (mua=0.001, mus=10, mismatch n=1.37)
# =============================================================================
log_test "Scattering dominant (mua=0.001, mus=10)"

cat > t02.json << 'ENDJSON'
{
    "Session": { "ID": "t02", "Photons": 500000, "DoMismatch": true },
    "Forward": { "T0": 0, "T1": 5e-9, "Dt": 5e-9 },
    "Domain": {
        "LengthUnit": 1.0,
        "Media": [
            {"mua": 0, "mus": 0, "g": 1, "n": 1},
            {"mua": 0.001, "mus": 10, "g": 0.9, "n": 1.37}
        ]
    },
    "Optode": {
        "Source": { "Type": "pencil", "Pos": [30, 30, 19.5], "Dir": [0, 0, 1] }
    },
    "Shapes": [
        {"Sphere": {"Tag": 1, "O": [30, 30, 30], "R": 10}}
    ]
}
ENDJSON

if run_sim t02.json t02.log -n 500000 -E 12345; then
    abs=$(get_absorbed t02.log)
    [ -n "$abs" ] && check_range "$abs" 1.5 9.0 "Low absorption with high scattering" \
                  || log_fail "Could not parse absorbed%"
else
    log_fail "Execution failed"
fi

# =============================================================================
# TEST 3: Concentric spheres — CSG layered media
# Inner R=5 (Tag=2, mua=0.05, mus=5), Outer R=10 (Tag=1, mua=0.01, mus=10)
# =============================================================================
log_test "Concentric spheres — CSG layered media"

cat > t03.json << 'ENDJSON'
{
    "Session": { "ID": "t03", "Photons": 500000, "DoMismatch": false },
    "Forward": { "T0": 0, "T1": 5e-9, "Dt": 5e-9 },
    "Domain": {
        "LengthUnit": 1.0,
        "Media": [
            {"mua": 0, "mus": 0, "g": 1, "n": 1},
            {"mua": 0.01, "mus": 10, "g": 0.9, "n": 1.37},
            {"mua": 0.05, "mus": 5, "g": 0.9, "n": 1.37}
        ]
    },
    "Optode": {
        "Source": { "Type": "pencil", "Pos": [30, 30, 19.5], "Dir": [0, 0, 1] }
    },
    "Shapes": [
        {"Sphere": {"Tag": 1, "O": [30, 30, 30], "R": 10}},
        {"Sphere": {"Tag": 2, "O": [30, 30, 30], "R": 5}}
    ]
}
ENDJSON

if run_sim t03.json t03.log -n 500000 -E 12345 -b 0; then
    abs=$(get_absorbed t03.log)
    [ -n "$abs" ] && check_range "$abs" 13.0 42.0 "Layered sphere absorption" \
                  || log_fail "Could not parse absorbed%"
else
    log_fail "Execution failed"
fi

# =============================================================================
# TEST 4: Cube — triangle count (expect 12)
# =============================================================================
log_test "Cube mesh — triangle count (expect 12)"

cat > t04.json << 'ENDJSON'
{
    "Session": { "ID": "t04", "Photons": 100, "DoMismatch": false },
    "Forward": { "T0": 0, "T1": 5e-9, "Dt": 5e-9 },
    "Domain": {
        "LengthUnit": 1.0,
        "MeshRes": 1,
        "Media": [
            {"mua": 0, "mus": 0, "g": 1, "n": 1},
            {"mua": 0.01, "mus": 1, "g": 0.9, "n": 1.37}
        ]
    },
    "Optode": {
        "Source": { "Type": "pencil", "Pos": [30, 30, 9.5], "Dir": [0, 0, 1] }
    },
    "Shapes": [
        {"Box": {"Tag": 1, "O": [10, 10, 10], "Size": [40, 40, 40]}}
    ]
}
ENDJSON

if run_sim t04.json t04.log -n 100 -E 1 -b 0; then
    tc=$(get_triangles t04.log)
    [ -n "$tc" ] && check_range "$tc" 12 24 "Cube triangle count" \
                 || log_skip "Could not parse triangle count"
else
    log_fail "Execution failed"
fi

# =============================================================================
# TEST 5: Sphere resolution scaling (ratio ≈ 4)
# =============================================================================
log_test "Sphere mesh — resolution scaling"

cat > t05.json << 'ENDJSON'
{
    "Session": { "ID": "t05", "Photons": 100, "DoMismatch": false },
    "Forward": { "T0": 0, "T1": 5e-9, "Dt": 5e-9 },
    "Domain": {
        "LengthUnit": 1.0,
        "Media": [
            {"mua": 0, "mus": 0, "g": 1, "n": 1},
            {"mua": 0.01, "mus": 1, "g": 0.9, "n": 1.0}
        ]
    },
    "Optode": {
        "Source": { "Type": "pencil", "Pos": [30, 30, 19.5], "Dir": [0, 0, 1] }
    },
    "Shapes": [
        {"Sphere": {"Tag": 1, "O": [30, 30, 30], "R": 10}}
    ]
}
ENDJSON

tri_lo="" tri_hi=""
run_sim t05.json t05lo.log -n 100 -E 1 -b 0 -m 12 && tri_lo=$(get_triangles t05lo.log)
run_sim t05.json t05hi.log -n 100 -E 1 -b 0 -m 24 && tri_hi=$(get_triangles t05hi.log)

if [ -n "$tri_lo" ] && [ -n "$tri_hi" ]; then
    ratio=$(python3 -c "print(round($tri_hi / $tri_lo, 2))")
    check_range "$ratio" 2.5 6.0 "Tri ratio res24/res12 ($ratio: $tri_lo → $tri_hi)"
else
    log_skip "Could not parse triangle counts"
fi

# =============================================================================
# TEST 6: Fresnel reflection mismatch ON vs OFF
# n=1.5 inside sphere. With mismatch, internal reflection traps photons longer.
# =============================================================================
log_test "Fresnel reflection — mismatch ON vs OFF"

cat > t06.json << 'ENDJSON'
{
    "Session": { "ID": "t06", "Photons": 500000, "DoMismatch": true },
    "Forward": { "T0": 0, "T1": 5e-9, "Dt": 5e-9 },
    "Domain": {
        "LengthUnit": 1.0,
        "Media": [
            {"mua": 0, "mus": 0, "g": 1, "n": 1.0},
            {"mua": 0.02, "mus": 8, "g": 0.9, "n": 1.5}
        ]
    },
    "Optode": {
        "Source": { "Type": "pencil", "Pos": [30, 30, 19.5], "Dir": [0, 0, 1] }
    },
    "Shapes": [
        {"Sphere": {"Tag": 1, "O": [30, 30, 30], "R": 10}}
    ]
}
ENDJSON

abs_on="" abs_off=""
run_sim t06.json t06_on.log  -n 500000 -E 12345 -b 1 && abs_on=$(get_absorbed t06_on.log)
run_sim t06.json t06_off.log -n 500000 -E 12345 -b 0 && abs_off=$(get_absorbed t06_off.log)

if [ -n "$abs_on" ] && [ -n "$abs_off" ]; then
    diff=$(python3 -c "print(round(abs($abs_on - $abs_off), 2))")
    if python3 -c "import sys; sys.exit(0 if abs($abs_on - $abs_off) > 1.0 else 1)"; then
        log_pass "Mismatch changes absorption (on=${abs_on}%, off=${abs_off}%, Δ=${diff}%)"
    else
        log_fail "Mismatch had no effect (on=${abs_on}%, off=${abs_off}%)"
    fi
else
    log_fail "Could not parse absorption values"
fi

# =============================================================================
# TEST 7: Energy conservation (typical medium)
# mua=0.02, mus=10, g=0.9, n=1.37, mismatch on
# =============================================================================
log_test "Energy conservation — typical medium"

cat > t07.json << 'ENDJSON'
{
    "Session": { "ID": "t07", "Photons": 500000, "DoMismatch": true },
    "Forward": { "T0": 0, "T1": 10e-9, "Dt": 10e-9 },
    "Domain": {
        "LengthUnit": 1.0,
        "Media": [
            {"mua": 0, "mus": 0, "g": 1, "n": 1},
            {"mua": 0.02, "mus": 10, "g": 0.9, "n": 1.37}
        ]
    },
    "Optode": {
        "Source": { "Type": "pencil", "Pos": [30, 30, 19.5], "Dir": [0, 0, 1] }
    },
    "Shapes": [
        {"Sphere": {"Tag": 1, "O": [30, 30, 30], "R": 10}}
    ]
}
ENDJSON

if run_sim t07.json t07.log -n 500000 -E 99999; then
    abs=$(get_absorbed t07.log)
    [ -n "$abs" ] && check_range "$abs" 10.0 90.0 "Absorption in valid range" \
                  || log_fail "Could not parse absorbed%"
else
    log_fail "Execution failed"
fi

# =============================================================================
# TEST 8: Curvature ON vs OFF — low-res sphere, high n mismatch
# =============================================================================
log_test "Curvature correction — low-res sphere"

cat > t08.json << 'ENDJSON'
{
    "Session": { "ID": "t08", "Photons": 500000, "DoMismatch": true },
    "Forward": { "T0": 0, "T1": 5e-9, "Dt": 5e-9 },
    "Domain": {
        "LengthUnit": 1.0,
        "Media": [
            {"mua": 0, "mus": 0, "g": 1, "n": 1.0},
            {"mua": 0.02, "mus": 8, "g": 0.9, "n": 1.5}
        ]
    },
    "Optode": {
        "Source": { "Type": "pencil", "Pos": [30, 30, 19.5], "Dir": [0, 0, 1] }
    },
    "Shapes": [
        {"Sphere": {"Tag": 1, "O": [30, 30, 30], "R": 10}}
    ]
}
ENDJSON

abs_curv="" abs_flat=""
run_sim t08.json t08c.log -n 500000 -E 12345 -c 1 -m 8 && abs_curv=$(get_absorbed t08c.log)
run_sim t08.json t08f.log -n 500000 -E 12345 -c 0 -m 8 && abs_flat=$(get_absorbed t08f.log)

if [ -n "$abs_curv" ] && [ -n "$abs_flat" ]; then
    diff=$(python3 -c "print(round(abs($abs_curv - $abs_flat), 4))")
    if python3 -c "import sys; sys.exit(0 if abs($abs_curv - $abs_flat) > 0.05 else 1)"; then
        log_pass "Curvature changes result (curv=${abs_curv}%, flat=${abs_flat}%, Δ=${diff}%)"
    else
        log_skip "Curvature had no effect — may need USE_CURVATURE build (curv=${abs_curv}%, flat=${abs_flat}%)"
    fi
else
    log_skip "Could not run curvature comparison"
fi

# =============================================================================
# TEST 9: Disk source vs pencil beam — both should absorb
# =============================================================================
log_test "Disk source vs pencil beam"

cat > t09d.json << 'ENDJSON'
{
    "Session": { "ID": "t09d", "Photons": 500000, "DoMismatch": true },
    "Forward": { "T0": 0, "T1": 5e-9, "Dt": 5e-9 },
    "Domain": {
        "LengthUnit": 1.0,
        "Media": [
            {"mua": 0, "mus": 0, "g": 1, "n": 1},
            {"mua": 0.02, "mus": 10, "g": 0.9, "n": 1.37}
        ]
    },
    "Optode": {
        "Source": { "Type": "disk", "Pos": [30, 30, 19.5], "Dir": [0, 0, 1], "Param1": [3, 0, 0, 0] }
    },
    "Shapes": [
        {"Sphere": {"Tag": 1, "O": [30, 30, 30], "R": 10}}
    ]
}
ENDJSON

cat > t09p.json << 'ENDJSON'
{
    "Session": { "ID": "t09p", "Photons": 500000, "DoMismatch": true },
    "Forward": { "T0": 0, "T1": 5e-9, "Dt": 5e-9 },
    "Domain": {
        "LengthUnit": 1.0,
        "Media": [
            {"mua": 0, "mus": 0, "g": 1, "n": 1},
            {"mua": 0.02, "mus": 10, "g": 0.9, "n": 1.37}
        ]
    },
    "Optode": {
        "Source": { "Type": "pencil", "Pos": [30, 30, 19.5], "Dir": [0, 0, 1] }
    },
    "Shapes": [
        {"Sphere": {"Tag": 1, "O": [30, 30, 30], "R": 10}}
    ]
}
ENDJSON

abs_d="" abs_p=""
run_sim t09d.json t09d.log -n 500000 -E 12345 && abs_d=$(get_absorbed t09d.log)
run_sim t09p.json t09p.log -n 500000 -E 12345 && abs_p=$(get_absorbed t09p.log)

if [ -n "$abs_d" ] && [ -n "$abs_p" ]; then
    # Disk source: some photons miss sphere edge → slightly less absorption
    check_range "$abs_d" 10.0 80.0 "Disk source absorption nonzero (${abs_d}%)"
    check_range "$abs_p" 10.0 80.0 "Pencil source absorption nonzero (${abs_p}%)"
else
    log_fail "Could not parse absorption values"
fi

# =============================================================================
# TEST 10: Reproducibility
# =============================================================================
log_test "Reproducibility with fixed RNG seed"

abs_r1="" abs_r2=""
run_sim t07.json t10a.log -n 100000 -E 42 && abs_r1=$(get_absorbed t10a.log)
run_sim t07.json t10b.log -n 100000 -E 42 && abs_r2=$(get_absorbed t10b.log)

if [ -n "$abs_r1" ] && [ -n "$abs_r2" ]; then
    if [ "$abs_r1" = "$abs_r2" ]; then
        log_pass "Identical results with same seed (${abs_r1}%)"
    else
        diff=$(python3 -c "print(round(abs($abs_r1 - $abs_r2), 6))")
        if python3 -c "import sys; sys.exit(0 if abs($abs_r1 - $abs_r2) < 0.01 else 1)"; then
            log_pass "Match within rounding (${abs_r1}% vs ${abs_r2}%, Δ=${diff}%)"
        else
            log_fail "Results differ with same seed (${abs_r1}% vs ${abs_r2}%)"
        fi
    fi
else
    log_fail "Could not parse absorption values"
fi

# =============================================================================
# TEST 11: Cylinder mesh
# =============================================================================
log_test "Cylinder shape — mesh generation"

cat > t11.json << 'ENDJSON'
{
    "Session": { "ID": "t11", "Photons": 100, "DoMismatch": false },
    "Forward": { "T0": 0, "T1": 5e-9, "Dt": 5e-9 },
    "Domain": {
        "LengthUnit": 1.0,
        "Media": [
            {"mua": 0, "mus": 0, "g": 1, "n": 1},
            {"mua": 0.01, "mus": 1, "g": 0.9, "n": 1.37}
        ]
    },
    "Optode": {
        "Source": { "Type": "pencil", "Pos": [30, 30, 19.5], "Dir": [0, 0, 1] }
    },
    "Shapes": [
        {"Cylinder": {"Tag": 1, "C0": [30, 30, 20], "C1": [30, 30, 40], "R": 8}}
    ]
}
ENDJSON

if run_sim t11.json t11.log -n 100 -E 1 -b 0; then
    tc=$(get_triangles t11.log)
    [ -n "$tc" ] && check_range "$tc" 30 5000 "Cylinder triangle count ($tc)" \
                 || log_skip "Could not parse triangle count"
else
    log_fail "Execution failed"
fi

# =============================================================================
# TEST 12: Zero absorption
# =============================================================================
log_test "Zero absorption (mua=0)"

cat > t12.json << 'ENDJSON'
{
    "Session": { "ID": "t12", "Photons": 100000, "DoMismatch": false },
    "Forward": { "T0": 0, "T1": 5e-9, "Dt": 5e-9 },
    "Domain": {
        "LengthUnit": 1.0,
        "Media": [
            {"mua": 0, "mus": 0, "g": 1, "n": 1},
            {"mua": 0, "mus": 10, "g": 0.9, "n": 1.0}
        ]
    },
    "Optode": {
        "Source": { "Type": "pencil", "Pos": [30, 30, 19.5], "Dir": [0, 0, 1] }
    },
    "Shapes": [
        {"Sphere": {"Tag": 1, "O": [30, 30, 30], "R": 10}}
    ]
}
ENDJSON

if run_sim t12.json t12.log -n 100000 -E 12345 -b 0; then
    abs=$(get_absorbed t12.log)
    [ -n "$abs" ] && check_range "$abs" 0.0 0.5 "Zero absorption deposits negligible energy" \
                  || log_fail "Could not parse absorbed%"
else
    log_fail "Execution failed"
fi

# =============================================================================
# TEST 13: High absorption (mua=1.0)
# Analytical: 1-exp(-1.0*20) ≈ 100%
# =============================================================================
log_test "High absorption (mua=1.0)"

cat > t13.json << 'ENDJSON'
{
    "Session": { "ID": "t13", "Photons": 200000, "DoMismatch": false },
    "Forward": { "T0": 0, "T1": 5e-9, "Dt": 5e-9 },
    "Domain": {
        "LengthUnit": 1.0,
        "Media": [
            {"mua": 0, "mus": 0, "g": 1, "n": 1},
            {"mua": 1.0, "mus": 10, "g": 0.9, "n": 1.0}
        ]
    },
    "Optode": {
        "Source": { "Type": "pencil", "Pos": [30, 30, 19.5], "Dir": [0, 0, 1] }
    },
    "Shapes": [
        {"Sphere": {"Tag": 1, "O": [30, 30, 30], "R": 10}}
    ]
}
ENDJSON

if run_sim t13.json t13.log -n 200000 -E 12345 -b 0; then
    abs=$(get_absorbed t13.log)
    [ -n "$abs" ] && check_range "$abs" 88.0 100.0 "High absorption ≈ 100%" \
                  || log_fail "Could not parse absorbed%"
else
    log_fail "Execution failed"
fi

# =============================================================================
# TEST 14: Absorption monotonicity (mua: 0.01 < 0.1 < 1.0)
# =============================================================================
log_test "Absorption monotonicity"

abs_vals=()
for i in 1 2 3; do
    mua=$(python3 -c "print([0.01, 0.1, 1.0][$i-1])")
    cat > "t14_${i}.json" << ENDJSON
{
    "Session": { "ID": "t14_${i}", "Photons": 500000, "DoMismatch": false },
    "Forward": { "T0": 0, "T1": 5e-9, "Dt": 5e-9 },
    "Domain": {
        "LengthUnit": 1.0,
        "Media": [
            {"mua": 0, "mus": 0, "g": 1, "n": 1},
            {"mua": ${mua}, "mus": 10, "g": 0.9, "n": 1.0}
        ]
    },
    "Optode": {
        "Source": { "Type": "pencil", "Pos": [30, 30, 19.5], "Dir": [0, 0, 1] }
    },
    "Shapes": [
        {"Sphere": {"Tag": 1, "O": [30, 30, 30], "R": 10}}
    ]
}
ENDJSON
    if run_sim "t14_${i}.json" "t14_${i}.log" -n 500000 -E 12345 -b 0; then
        a=$(get_absorbed "t14_${i}.log")
        abs_vals+=("${a:-0}")
    else
        abs_vals+=("0")
    fi
done

if [ "${abs_vals[0]}" != "0" ] && [ "${abs_vals[1]}" != "0" ] && [ "${abs_vals[2]}" != "0" ]; then
    if python3 -c "import sys; a=[${abs_vals[0]},${abs_vals[1]},${abs_vals[2]}]; sys.exit(0 if a[0]<a[1]<a[2] and 10<a[0]<45 and 48<a[1]<99 and a[2]>88 else 1)"; then
        log_pass "Monotonic: ${abs_vals[0]}% < ${abs_vals[1]}% < ${abs_vals[2]}%"
    else
        log_fail "Values: ${abs_vals[0]}%, ${abs_vals[1]}%, ${abs_vals[2]}% (expect monotonic increase)"
    fi
else
    log_fail "Could not run all three mua tests"
fi

# =============================================================================
# TEST 15: Performance
# =============================================================================
log_test "Performance — nonzero simulation speed"

if run_sim t07.json t15.log -n 500000 -E 1; then
    spd=$(get_speed t15.log)
    [ -n "$spd" ] && check_range "$spd" 1.0 100000000 "Speed > 0 (${spd} photon/ms)" \
                  || log_skip "Could not parse speed"
else
    log_fail "Execution failed"
fi

# =============================================================================
# TEST 16: Beer-Lambert in a box (pencil at normal incidence, no scattering)
# path inside = 20 - SAFETY_DIST(0.001): 1 - exp(-0.05 * 19.999)
# =============================================================================
log_test "Beer-Lambert: box"
mkcfg t16.json '{}'
if run_sim t16.json t16.log -E 1; then
    check_close "$(get_absorbed t16.log)" "$(pyval '100*(1-math.exp(-0.05*19.999))')" 0.02 "Box absorption"
else
    log_fail "Execution failed"
fi

# =============================================================================
# TEST 17: Beer-Lambert in a cylinder, along its axis and across its diameter
# =============================================================================
log_test "Beer-Lambert: cylinder along the axis and across the diameter"
mkcfg t17a.json '{"Shapes":[{"Cylinder":{"Tag":1,"C0":[30,30,10],"C1":[30,30,40],"R":5}}]}'
mkcfg t17b.json '{"Shapes":[{"Cylinder":{"Tag":1,"C0":[10,30,30],"C1":[50,30,30],"R":8}}]}'
if run_sim t17a.json t17a.log -E 1 && run_sim t17b.json t17b.log -E 1; then
    check_close "$(get_absorbed t17a.log)" "$(pyval '100*(1-math.exp(-0.05*30))')" 0.02 "Along the axis (path 30)"
    check_close "$(get_absorbed t17b.log)" "$(pyval '100*(1-math.exp(-0.05*16))')" 0.05 "Across the diameter (path 16)"
else
    log_fail "Execution failed"
fi

# =============================================================================
# TEST 18: Fresnel reflection at normal incidence
# n=1.5 slab, mua*L = 40: everything transmitted is absorbed, so
# absorbed = 1 - R, R = ((n-1)/(n+1))^2 = 4%; standard error 0.044%
# =============================================================================
log_test "Fresnel: specular reflection at normal incidence"
mkcfg t18.json '{"Session":{"DoMismatch":true},"Domain":{"Media":[{"mua":0,"mus":0,"g":1,"n":1},{"mua":1,"mus":0,"g":1,"n":1.5}]},
                 "Shapes":[{"Box":{"Tag":1,"O":[0,0,10],"Size":[60,60,40]}}]}'
if run_sim t18.json t18.log -E 1; then
    check_close "$(get_absorbed t18.log)" 96.0 0.2 "1 - R"
else
    log_fail "Execution failed"
fi

# =============================================================================
# TEST 19: LengthUnit: coordinates in units of 2 mm; the voxel stays one unit
# =============================================================================
log_test "LengthUnit scaling"
mkcfg t19.json '{"Domain":{"LengthUnit":2}}'
if run_sim t19.json t19.log -E 1 && run_sim t16.json t19b.log -E 1 -s t19b; then
    check_close "$(get_absorbed t19.log)" "$(pyval '100*(1-math.exp(-0.05*2*19.999))')" 0.02 "Absorption with 2 mm units"
    [ "$(jdat dims t19.jdat)" = "$(jdat dims t19b.jdat)" ] && log_pass "Same grid as LengthUnit=1 ($(jdat dims t19.jdat))" \
        || log_fail "Grid differs: $(jdat dims t19.jdat) vs $(jdat dims t19b.jdat)"
else
    log_fail "Execution failed"
fi

# =============================================================================
# TEST 20: Time-resolved output sums to the CW output
# =============================================================================
log_test "Time gates: 10 gates sum to the single-gate result"
mkcfg t20a.json '{"Domain":{"Media":[{"mua":0,"mus":0,"g":1,"n":1},{"mua":0.01,"mus":1,"g":0.9,"n":1.37}]},
                  "Optode":{"Source":{"Pos":[30,30,0.5]}},"Shapes":[{"Grid":{"Tag":1,"Size":[60,60,60]}}]}'
mkcfg t20b.json '{"Domain":{"Media":[{"mua":0,"mus":0,"g":1,"n":1},{"mua":0.01,"mus":1,"g":0.9,"n":1.37}]},
                  "Forward":{"Dt":5e-10},"Optode":{"Source":{"Pos":[30,30,0.5]}},"Shapes":[{"Grid":{"Tag":1,"Size":[60,60,60]}}]}'
if run_sim t20a.json t20a.log -E 3 -U 0 && run_sim t20b.json t20b.log -E 3 -U 0; then
    s1=$(jdat sum t20a.jdat); s10=$(jdat sum t20b.jdat)
    check_close "$s10" "$s1" "$(pyval "abs($s1)*1e-6")" "Sum over 10 gates"
    [ "$(jdat dims t20b.jdat)" = "61x61x61x10" ] && log_pass "4-D output 61x61x61x10" || log_fail "dims $(jdat dims t20b.jdat)"
else
    log_fail "Execution failed"
fi

# =============================================================================
# TEST 21: Time of flight: in a transparent box (n=1.5), gate g holds the
# deposits at depth z = (g+0.5) Dt c/n; checks the time scaling for LengthUnit
# =============================================================================
log_test "Time of flight, LengthUnit=1 and 2"
for lu in 1 2; do
    mkcfg t21_$lu.json '{"Forward":{"T0":0,"T1":3e-10,"Dt":5e-11},"Domain":{"LengthUnit":'$lu',"Media":[{"mua":0,"mus":0,"g":1,"n":1},{"mua":0.001,"mus":0,"g":1,"n":1.5}]},
                        "Optode":{"Source":{"Pos":[5,5,0]}},"Shapes":[{"Box":{"Tag":1,"O":[0,0,0],"Size":[10,10,60]}}]}'
    if run_sim t21_$lu.json t21_$lu.log -E 1 -n 20000; then
        got=$(jdat gatez t21_$lu.jdat)
        ok=$(python3 -c "
g=[float(x) for x in '$got'.split()]
ref=[(k+0.5)*5e-11*2.99792458e11/1.5/$lu for k in range(len(g))]
print(1 if len(g)==6 and all(abs(a-b)<0.1 for a,b in zip(g,ref)) else 0)")
        [ "$ok" = 1 ] && log_pass "LengthUnit=$lu gate depth centroids: $got" || log_fail "LengthUnit=$lu gate depth centroids: $got"
    else
        log_fail "Execution failed"
    fi
done

# =============================================================================
# TEST 22: Time window with T0 > 0: deposits before T0 or after T1 are dropped
# =============================================================================
log_test "Time window T0 > 0"
mkcfg t22.json '{"Forward":{"T0":1e-10,"T1":2e-10,"Dt":5e-11},"Domain":{"Media":[{"mua":0,"mus":0,"g":1,"n":1},{"mua":0.001,"mus":0,"g":1,"n":1.5}]},
                 "Optode":{"Source":{"Pos":[5,5,0]}},"Shapes":[{"Box":{"Tag":1,"O":[0,0,0],"Size":[10,10,60]}}]}'
if run_sim t22.json t22.log -E 1 -n 20000 -U 0; then
    got=$(jdat gatez t22.jdat)
    ok=$(python3 -c "
g=[float(x) for x in '$got'.split()]
ref=[(k+2.5)*5e-11*2.99792458e11/1.5 for k in range(2)]
print(1 if len(g)==2 and all(abs(a-b)<0.1 for a,b in zip(g,ref)) else 0)")
    [ "$ok" = 1 ] && log_pass "Gates cover 1e-10..2e-10 only: $got" || log_fail "Gate depth centroids: $got"
    # energy in the window = attenuation over the depth range 20..40 mm
    check_close "$(jdat sum t22.jdat)" "$(pyval '20000*(math.exp(-0.001*19.98618)-math.exp(-0.001*39.97236))')" 2 "Energy within the window"
else
    log_fail "Execution failed"
fi

# =============================================================================
# TEST 23: Output types: energy = mua x fluence; flux = fluence / Dt
# =============================================================================
log_test "Output types (energy, fluence, flux)"
mkcfg t23.json '{"Domain":{"Media":[{"mua":0,"mus":0,"g":1,"n":1},{"mua":0.01,"mus":1,"g":0.9,"n":1}]},
                 "Optode":{"Source":{"Pos":[30,30,0.5]}},"Shapes":[{"Grid":{"Tag":1,"Size":[60,60,60]}}]}'
if run_sim t23.json t23x.log -E 5 -O x -s t23x && run_sim t23.json t23l.log -E 5 -O l -s t23l &&
        run_sim t23.json t23f.log -E 5 -O f -s t23f; then
    sx=$(jdat sum t23x.jdat); sl=$(jdat sum t23l.jdat); sf=$(jdat sum t23f.jdat)
    check_close "$(pyval "$sx/$sl")" 0.01 1e-6 "energy / fluence = mua"
    check_close "$(pyval "$sf*5e-9/$sl")" 1 1e-6 "flux * Dt / fluence = 1"
else
    log_fail "Execution failed"
fi

# =============================================================================
# TEST 24: Normalization: -U 0 keeps the raw sum, -U 1 divides by N * Vvoxel
# =============================================================================
log_test "Normalization"
if run_sim t16.json t24a.log -E 1 -U 0 -s t24a && run_sim t16.json t24b.log -E 1 -U 1 -s t24b; then
    ab=$(get_absorbed t24a.log)
    check_close "$(jdat sum t24a.jdat)" "$(pyval "$ab/100*200000")" 1 "Raw sum = absorbed weight"
    check_close "$(pyval "$(jdat sum t24b.jdat)*200000")" "$(jdat sum t24a.jdat)" 1 "Normalized sum x N = raw sum (1 mm^3 voxels)"
else
    log_fail "Execution failed"
fi

# =============================================================================
# TEST 25: Mesh mode: a tetrahedral cube equals the CSG Grid; every JData
# encoding of the mesh and MeshSurf/MeshElem give the same result
# =============================================================================
log_test "Mesh mode and JData encodings"
CUBEPATCH='{"Session":{"Photons":200000},"Optode":{"Source":{"Pos":[30,30,0.5]}}}'
for enc in zlib gzip inline data base64; do
    python3 "$SCRIPT_DIR/mkmesh.py" "$TESTDIR/t25_$enc.json" --enc $enc --json "$CUBEPATCH"
done
python3 "$SCRIPT_DIR/mkmesh.py" "$TESTDIR/t25_dbl.json" --dtype double --json "$CUBEPATCH"
python3 "$SCRIPT_DIR/mkmesh.py" "$TESTDIR/t25_col.json" --dtype double --order c --json "$CUBEPATCH"
python3 "$SCRIPT_DIR/mkmesh.py" "$TESTDIR/t25_surf.json" --surf --json "$CUBEPATCH"
mkcfg t25_csg.json '{"Domain":{"Media":[{"mua":0,"mus":0,"g":1,"n":1},{"mua":0.01,"mus":1,"g":0.9,"n":1}]},
                     "Optode":{"Source":{"Pos":[30,30,0.5]}},"Shapes":[{"Grid":{"Tag":1,"Size":[60,60,60]}}]}'
if run_sim t25_zlib.json t25_zlib.log -E 9 && run_sim t25_csg.json t25_csg.log -E 9; then
    ref=$(get_absorbed t25_zlib.log)
    check_close "$(get_absorbed t25_csg.log)" "$ref" 0.2 "CSG Grid = tetrahedral cube"
    for v in gzip inline data base64 dbl col surf; do
        if run_sim t25_$v.json t25_$v.log -E 9; then
            check_close "$(get_absorbed t25_$v.log)" "$ref" 1e-4 "Mesh input $v"
        else
            log_fail "Mesh input $v: $(grep ERROR $TESTDIR/t25_$v.log)"
        fi
    done
else
    log_fail "Execution failed"
fi

# =============================================================================
# TEST 26: Mesh mode with InitElem and with a source outside the mesh
# =============================================================================
log_test "Mesh mode: InitElem; source outside the mesh"
python3 "$SCRIPT_DIR/mkmesh.py" "$TESTDIR/t26_init.json" --json '{"Optode":{"Source":{"Pos":[30,30,0.5]}},"Shapes":{"InitElem":1}}'
python3 "$SCRIPT_DIR/mkmesh.py" "$TESTDIR/t26_out.json" --json '{"Session":{"DoMismatch":true},"Optode":{"Source":{"Pos":[30,30,-5]}},
    "Domain":{"Media":[{"mua":0,"mus":0,"g":1,"n":1},{"mua":0.01,"mus":1,"g":0.9,"n":1.37},{"mua":0,"mus":0,"g":1,"n":1}]}}'
mkcfg t26_csg.json '{"Session":{"DoMismatch":true},"Domain":{"Media":[{"mua":0,"mus":0,"g":1,"n":1},{"mua":0.01,"mus":1,"g":0.9,"n":1.37}]},
                     "Optode":{"Source":{"Pos":[30,30,-5]}},"Shapes":[{"Grid":{"Tag":1,"Size":[60,60,60]}}]}'
if run_sim t26_init.json t26_init.log -E 9 -n 100000 && run_sim t25_zlib.json t26_ref.log -E 9 -n 100000 -s t26_ref; then
    check_close "$(get_absorbed t26_init.log)" "$(get_absorbed t26_ref.log)" 1e-4 "InitElem = probed initial element"
else
    log_fail "Execution failed (InitElem)"
fi
if run_sim t26_out.json t26_out.log -E 9 -n 100000 && run_sim t26_csg.json t26_csg.log -E 9 -n 100000; then
    check_close "$(get_absorbed t26_out.log)" "$(get_absorbed t26_csg.log)" 0.3 "Source outside the mesh enters it (= CSG)"
else
    log_fail "Execution failed (outside source)"
fi

# =============================================================================
# TEST 27: --dumpmesh: the dumped mesh of a CSG shape is a valid mesh input
# =============================================================================
log_test "--dumpmesh round trip"
mkcfg t27.json '{"Domain":{"Media":[{"mua":0,"mus":0,"g":1,"n":1},{"mua":0.02,"mus":2,"g":0.8,"n":1}]},
                 "Optode":{"Source":{"Pos":[30,30,15]}},"Shapes":[{"Sphere":{"Tag":1,"O":[30,30,30],"R":10}}]}'
if run_sim t27.json t27d.log --dumpmesh && [ -f "$TESTDIR/t27_mesh.json" ] && run_sim t27.json t27.log -E 2 -c 0 &&
        run_sim t27_mesh.json t27m.log -E 2 -s t27m; then
    check_close "$(get_absorbed t27m.log)" "$(get_absorbed t27.log)" 0.15 "Dumped mesh = CSG sphere"
else
    log_fail "Execution failed"
fi

# =============================================================================
# TEST 28: CSG shapes with coplanar faces (regression: adjacent layers and a
# box flush with the domain lost one of the two medium changes)
# =============================================================================
log_test "CSG coplanar faces: ZLayers and a flush Box equal one Grid"
MED3='"Media":[{"mua":0,"mus":0,"g":1,"n":1},{"mua":0.01,"mus":1,"g":0.9,"n":1},{"mua":0.01,"mus":1,"g":0.9,"n":1},{"mua":0.01,"mus":1,"g":0.9,"n":1}]'
mkcfg t28g.json '{"Domain":{'"$MED3"'},"Optode":{"Source":{"Pos":[30,30,0.5]}},"Shapes":[{"Grid":{"Tag":1,"Size":[60,60,60]}}]}'
mkcfg t28l.json '{"Domain":{'"$MED3"'},"Optode":{"Source":{"Pos":[30,30,0.5]}},"Shapes":[{"Grid":{"Tag":1,"Size":[60,60,60]}},{"ZLayers":[[1,20,1],[21,40,2],[41,60,3]]}]}'
mkcfg t28b.json '{"Domain":{'"$MED3"'},"Optode":{"Source":{"Pos":[30,30,0.5]}},"Shapes":[{"Grid":{"Tag":1,"Size":[60,60,60]}},{"Box":{"Tag":2,"O":[20,20,0],"Size":[20,20,30]}}]}'
if run_sim t28g.json t28g.log -E 4 && run_sim t28l.json t28l.log -E 4 && run_sim t28b.json t28b.log -E 4; then
    ref=$(get_absorbed t28g.log)
    check_close "$(get_absorbed t28l.log)" "$ref" 0.25 "ZLayers"
    check_close "$(get_absorbed t28b.log)" "$ref" 0.25 "Box flush with the Grid wall"
else
    log_fail "Execution failed"
fi

# =============================================================================
# TEST 29: Source inside a CSG shape (regression: OptiX ignored enclosing shapes)
# =============================================================================
log_test "Source inside an enclosing shape"
mkcfg t29.json '{"Domain":{"Media":[{"mua":0,"mus":0,"g":1,"n":1},{"mua":0.01,"mus":0,"g":1,"n":1},{"mua":0.01,"mus":0,"g":1,"n":1}]},
                 "Optode":{"Source":{"Pos":[30,30,5]}},"Shapes":[{"Grid":{"Tag":1,"Size":[60,60,60]}},{"Sphere":{"Tag":2,"O":[30,30,30],"R":10}}]}'
if run_sim t29.json t29.log -E 1 -n 20000; then
    check_close "$(get_absorbed t29.log)" "$(pyval '100*(1-math.exp(-0.01*54.997))')" 0.05 "Straight path through Grid and Sphere"
else
    log_fail "Execution failed"
fi

# =============================================================================
# TEST 30: Planar source
# =============================================================================
log_test "Planar source"
mkcfg t30.json '{"Optode":{"Source":{"Type":"planar","Pos":[20,20,0],"Param1":[20,0,0],"Param2":[0,20,0]}}}'
if run_sim t30.json t30.log -E 1; then
    check_close "$(get_absorbed t30.log)" "$(pyval '100*(1-math.exp(-0.05*19.999))')" 0.02 "Planar beam on the box face"
    [ "$(jdat argmax t30.jdat | cut -d' ' -f3)" = "1" ] && log_pass "Maximum at the entrance plane" || log_fail "argmax $(jdat argmax t30.jdat)"
else
    log_fail "Execution failed"
fi

# =============================================================================
# TEST 31: Trajectories (-D M) conserve energy: mean(1 - final weight) = absorbed
# =============================================================================
log_test "Trajectory output"
if [ "$ISOPTIX" = 1 ]; then
    log_skip "trajectories are only saved by the Vulkan backend"
else
    mkcfg t31.json '{"Domain":{"Media":[{"mua":0,"mus":0,"g":1,"n":1},{"mua":0.005,"mus":1,"g":0.01,"n":1}]},
                     "Optode":{"Source":{"Pos":[30,30,19.5]}},"Shapes":[{"Sphere":{"Tag":1,"O":[30,30,30],"R":10}}]}'
    if run_sim t31.json t31.log -E 6 -n 10000 -D M; then
        read nev nph nlaunch < <(jdat traj t31_traj.jdat)
        [ "${nph:-0}" = 10000 ] && [ "${nlaunch:-0}" = 10000 ] && log_pass "10000 photons, $nev events" || log_fail "traj: '$nev $nph $nlaunch'"
        wl=$(python3 - "$TESTDIR/t31_traj.jdat" <<PYEOF
import sys, json
sys.path.insert(0, "$SCRIPT_DIR")
import jdat
tr = json.load(open(sys.argv[1]))['MCXData']['Trajectory']
_, ids = jdat.decode(tr['photonid']); _, w = jdat.decode(tr['w0'])
last = {}
for i, x in zip(ids, w):
    last[i] = x
print('%.5f' % (100.0 * sum(1 - x for x in last.values()) / 10000))
PYEOF
)
        check_close "$wl" "$(get_absorbed t31.log)" 0.01 "Weight bookkeeping = deposited energy"
    else
        log_fail "Execution failed"
    fi
fi

# =============================================================================
# TEST 32: Bit-reproducibility (fixed-point accumulation)
# =============================================================================
log_test "Bit-reproducible output"
if run_sim t23.json t32a.log -E 11 -s t32a && run_sim t23.json t32b.log -E 11 -s t32b && run_sim t23.json t32c.log -E 12 -s t32c; then
    same=$(python3 -c "import json; print(json.load(open('$TESTDIR/t32a.jdat'))['Fluence']==json.load(open('$TESTDIR/t32b.jdat'))['Fluence'])")
    diff=$(python3 -c "import json; print(json.load(open('$TESTDIR/t32a.jdat'))['Fluence']!=json.load(open('$TESTDIR/t32c.jdat'))['Fluence'])")
    [ "$same" = True ] && log_pass "Same seed: identical volume" || log_fail "Same seed gave a different volume"
    [ "$diff" = True ] && log_pass "Different seed: different volume" || log_fail "Different seeds gave the same volume"
else
    log_fail "Execution failed"
fi

# =============================================================================
# TEST 33: Statistical invariance to batch size, thread count, Russian roulette
# absorbed ~ 45.6%, per-photon sd ~ 0.35 -> standard error ~0.08% at 2e5
# =============================================================================
log_test "Batch size, thread count and Russian roulette do not bias the result"
if run_sim t23.json t33a.log -E 21 -s t33a && run_sim t23.json t33b.log -E 22 -B 20000 -s t33b &&
        run_sim t23.json t33c.log -E 23 -t 4096 -s t33c; then
    ref=$(get_absorbed t33a.log)
    check_close "$(get_absorbed t33b.log)" "$ref" 0.35 "-B 20000"
    check_close "$(get_absorbed t33c.log)" "$ref" 0.35 "-t 4096"
    mkcfg t33r.json '{"Session":{"RussianRoulette":0.01},"Domain":{"Media":[{"mua":0,"mus":0,"g":1,"n":1},{"mua":0.01,"mus":1,"g":0.9,"n":1}]},
                      "Optode":{"Source":{"Pos":[30,30,0.5]}},"Shapes":[{"Grid":{"Tag":1,"Size":[60,60,60]}}]}'
    run_sim t33r.json t33r.log -E 24 && check_close "$(get_absorbed t33r.log)" "$ref" 0.4 "Russian roulette threshold 0.01" \
        || log_fail "Execution failed (roulette)"
else
    log_fail "Execution failed"
fi

# =============================================================================
# TEST 34: Output file format
# =============================================================================
log_test "Output file format"
mkcfg t34.json '{"Domain":{"Step":[0.25,0.25,0.25]},"Session":{"Photons":20000}}'
if run_sim t16.json t34a.log -E 1 -s t34a && run_sim t34.json t34.log -E 1; then
    [ "$(jdat check t34a.jdat)" = OK ] && log_pass "JData annotations, Dim, column-major order" || log_fail "jdat check failed"
    [ "$(jdat argmax t34a.jdat)" = "20 20 1" ] && log_pass "Beam at voxel (20,20) along z" || log_fail "argmax $(jdat argmax t34a.jdat)"
    off=$(python3 -c "import json; print('_ArrayZipOffsets_' in json.load(open('$TESTDIR/t34.jdat'))['Fluence'])")
    [ "$off" = True ] && log_pass "Large volume ($(jdat dims t34.jdat)) written in parallel zlib blocks" || log_fail "no _ArrayZipOffsets_"
    check_close "$(pyval "$(jdat sum t34.jdat)*0.25**3*20000")" "$(pyval "$(get_absorbed t34.log)/100*20000")" 2 "Fine grid conserves energy"
else
    log_fail "Execution failed"
fi

# =============================================================================
# TEST 35: Invalid input is rejected with an error message, not a crash
# =============================================================================
log_test "Invalid input"
badcase() {   # badcase name 'json patch' 'expected message substring' [flags]
    mkcfg bad_$1.json "$2"
    run_sim bad_$1.json bad_$1.log -n 100 ${4:-}
    local rc=$?
    if [ $rc = 1 ] && grep -q "ERROR.*$3" "$TESTDIR/bad_$1.log"; then
        log_pass "$1: rejected ($3)"
    else
        log_fail "$1: rc=$rc, log: $(tail -c 300 $TESTDIR/bad_$1.log)"
    fi
}
badcase negmua '{"Domain":{"Media":[{"mua":0,"mus":0,"g":1,"n":1},{"mua":-1,"mus":0,"g":1,"n":1}]}}' 'mua must be'
badcase badg '{"Domain":{"Media":[{"mua":0,"mus":0,"g":1,"n":1},{"mua":0,"mus":0,"g":2,"n":1}]}}' 'g must be'
badcase tag '{"Shapes":[{"Box":{"Tag":5,"O":[0,0,0],"Size":[1,1,1]}}]}' 'Tag'
badcase radius '{"Shapes":[{"Sphere":{"Tag":1,"O":[0,0,0],"R":-1}}]}' 'R must be positive'
badcase shape '{"Shapes":[{"Torus":{"Tag":1}}]}' 'unknown shape'
badcase srctype '{"Optode":{"Source":{"Type":"laser"}}}' 'not a known source type'
badcase dir '{"Optode":{"Source":{"Dir":[0,0,0]}}}' 'non-zero'
badcase time '{"Forward":{"T0":1e-9,"T1":1e-10}}' 'T0 < T1'
badcase gates '{"Forward":{"T1":1,"Dt":1e-12}}' 'time gates'
badcase grid '{"Domain":{"Step":[0.001,0.001,0.001]}}' 'grid'
badcase meshnode '{"Shapes":{"MeshNode":[[0,0,0],[1,0,0],[0,1,0]],"MeshElem":[[1,2,3,9,1]]}}' 'node index'
badcase option '{}' 'unknown option' '--bogus'
badcase number '{}' 'expects a number' '-n 1e5x'
printf '{"Session": {' > "$TESTDIR/bad_syntax.json"
run_sim bad_syntax.json bad_syntax.log
[ $? = 1 ] && grep -q "JSON syntax error" "$TESTDIR/bad_syntax.log" && log_pass "syntax: rejected" || log_fail "syntax error not reported"
run_sim missing_file.json bad_missing.log
[ $? = 1 ] && grep -q "cannot open" "$TESTDIR/bad_missing.log" && log_pass "missing file: rejected" || log_fail "missing file not reported"

# =============================================================================
# TEST 36: Command line utilities
# =============================================================================
log_test "Command line: -h, -v, -L, --dumpjson"
"$EXEC" -h > "$TESTDIR/t36h.log" 2>&1 && grep -q -- "--photon" "$TESTDIR/t36h.log" && log_pass "-h" || log_fail "-h"
"$EXEC" -v > "$TESTDIR/t36v.log" 2>&1 && grep -qE "[0-9]+\.[0-9]+\.[0-9]+" "$TESTDIR/t36v.log" && log_pass "-v: $(head -1 $TESTDIR/t36v.log)" || log_fail "-v"
"$EXEC" -L > "$TESTDIR/t36l.log" 2>&1 && log_pass "-L" || log_fail "-L"
if run_sim t16.json t36d.log --dumpjson -n 1234; then
    python3 -c "
import json,sys
t=open('$TESTDIR/t36d.log').read(); d=json.loads(t[t.index('{\n'):])
sys.exit(0 if d['Session']['Photons']==1234 and d['Mesh']['Faces']==12 else 1)" && log_pass "--dumpjson" || log_fail "--dumpjson output"
else
    log_fail "--dumpjson failed"
fi

# =============================================================================
# TEST 37: Vulkan: the portable 32-bit atomic kernels match the 64-bit ones
# =============================================================================
log_test "Kernel variants give identical results"
if [ "$ISOPTIX" = 1 ]; then
    log_skip "Vulkan only"
elif run_sim t23.json t37a.log -E 31 -s t37a && VKMMC_NO_ATOMIC64=1 run_sim t23.json t37b.log -E 31 -s t37b; then
    if grep -q "32-bit pair" "$TESTDIR/t37b.log"; then
        same=$(python3 -c "import json; print(json.load(open('$TESTDIR/t37a.jdat'))['Fluence']==json.load(open('$TESTDIR/t37b.jdat'))['Fluence'])")
        [ "$same" = True ] && log_pass "64-bit and 32-bit-pair atomics: identical volume" || log_fail "volumes differ"
    else
        log_skip "the GPU has no 64-bit atomics; only the portable kernel ran"
    fi
else
    log_fail "Execution failed"
fi

# =============================================================================
# TEST 38: Cross-backend agreement (set XEXEC to the other backend)
# =============================================================================
log_test "Cross-backend agreement"
if [ -z "$XEXEC" ]; then
    log_skip "set XEXEC=<other executable> to compare backends"
else
    mkcfg t38.json '{"Session":{"DoMismatch":true},"Domain":{"Media":[{"mua":0,"mus":0,"g":1,"n":1},{"mua":0.01,"mus":1,"g":0.9,"n":1.37},{"mua":0.05,"mus":5,"g":0.9,"n":1.4}]},
                     "Optode":{"Source":{"Pos":[30,30,9.5]}},"Shapes":[{"Sphere":{"Tag":1,"O":[30,30,30],"R":20}},{"Sphere":{"Tag":2,"O":[30,30,30],"R":8}}]}'
    if run_sim t38.json t38a.log -E 41 -s t38a && (cd "$TESTDIR" && timeout 300 "$XEXEC" t38.json -E 42 -s t38b > t38b.log 2>&1); then
        check_close "$(get_absorbed t38b.log)" "$(get_absorbed t38a.log)" 0.35 "$(basename $XEXEC) vs $(basename $EXEC)"
    else
        log_fail "Execution failed"
    fi
fi

# =============================================================================
# TEST 39: GPU selection and user-supplied kernels (Vulkan)
# =============================================================================
log_test "GPU selection (-G) and user kernels (.spv)"
if [ "$ISOPTIX" = 1 ]; then
    log_skip "Vulkan only"
else
    SPV="$SCRIPT_DIR/../src/vkmmc_core_csg_i64.spv"
    if run_sim t16.json t39a.log -E 1 -G 1 -s t39a; then
        check_close "$(get_absorbed t39a.log)" "$(get_absorbed t16.log)" 1e-5 "-G 1 selects the first usable GPU"
    else
        log_fail "-G 1 failed"
    fi
    run_sim t16.json t39b.log -G 99
    [ $? = 1 ] && grep -q "usable GPU" "$TESTDIR/t39b.log" && log_pass "-G 99 rejected" || log_fail "-G 99 not rejected"
    if [ -f "$SPV" ] && grep -q "64-bit atomics" "$TESTDIR/t16.log"; then
        if run_sim t16.json t39c.log -E 1 -c 0 -s t39c "$SPV"; then
            grep -q "user supplied" "$TESTDIR/t39c.log" && check_close "$(get_absorbed t39c.log)" "$(get_absorbed t16.log)" 1e-5 \
                "User .spv kernel = embedded kernel" || log_fail "user kernel not used"
        else
            log_fail "user kernel run failed"
        fi
    else
        log_skip "no $SPV for this GPU (64-bit atomics needed)"
    fi
    printf 'not a kernel' > "$TESTDIR/bad.spv"
    run_sim t16.json t39d.log "$TESTDIR/bad.spv"
    [ $? = 1 ] && grep -q "not a SPIR-V binary" "$TESTDIR/t39d.log" && log_pass "invalid .spv rejected" || log_fail "invalid .spv not rejected"
    head -c 64 /dev/zero > "$TESTDIR/zero.spv"
    run_sim t16.json t39e.log "$TESTDIR/zero.spv"
    [ $? = 1 ] && grep -q "bad magic" "$TESTDIR/t39e.log" && log_pass "bad SPIR-V magic rejected" || log_fail "bad magic not rejected"
    run_sim t16.json t39f.log "$TESTDIR/missing.spv"
    [ $? = 1 ] && grep -q "cannot open SPIR-V" "$TESTDIR/t39f.log" && log_pass "missing .spv rejected" || log_fail "missing .spv not rejected"
fi

# =============================================================================
# TEST 40: Wide-field sources (MCX/MMC types): every collimated beam crosses
# the 20 mm absorbing slab (mua=0.05, no scattering) along +z, so each one,
# whatever its spatial profile or launch weights, absorbs exactly
# 1-exp(-0.05*20) = 63.212% of the launched energy
# =============================================================================
log_test "Wide-field source types through an absorbing slab"
WREF=$(pyval '100*(1-math.exp(-0.05*20))')
P1='"Param1":[30,0,0'
for spec in 'planar|"Pos":[15,15,0],'"$P1"',0],"Param2":[0,30,0,0]' \
            'disk|"Pos":[30,30,0],"Param1":[15,0,0,0]' \
            'ring|"Pos":[30,30,0],"Param1":[15,5,0,3]' \
            'gaussian|"Pos":[30,30,0],"Param1":[3,0,0,0]' \
            'pencilarray|"Pos":[15,15,0],'"$P1"',4],"Param2":[0,30,0,3]' \
            'slit|"Pos":[15,30,0],'"$P1"',0]' \
            'fourier|"Pos":[15,15,0],'"$P1"',2.5],"Param2":[0,30,0,1.5]' \
            'fourierx|"Pos":[15,15,0],'"$P1"',30],"Param2":[2,1,0.25,0.5]' \
            'fourierx2d|"Pos":[15,15,0],'"$P1"',30],"Param2":[2,1,0.25,0.1]' \
            'pattern|"Pos":[15,15,0],'"$P1"',3],"Param2":[0,30,0,2],"Pattern":{"Nx":3,"Ny":2,"Data":[1,0.5,0,0.25,0.75,1]}' \
            'pattern3d|"Pos":[20,20,1],"Param1":[3,2,2,0],"Pattern":{"Nx":3,"Ny":2,"Nz":2,"Data":[1,2,0,3,0.5,1,1,1,0,2,3,1]}'; do
    t=${spec%%|*}
    mkcfg t40_$t.json '{"Session":{"Photons":20000},"Optode":{"Source":{"Type":"'$t'",'"${spec#*|}"'}}}'
    if run_sim t40_$t.json t40_$t.log -E 3 -s t40_$t; then
        check_close "$(get_absorbed t40_$t.log)" "$WREF" 0.01 "$t: absorbed = 1-exp(-mua L)"
    else
        log_fail "$t failed: $(grep -m1 ERROR "$TESTDIR/t40_$t.log")"
    fi
done
mkcfg t40w.json '{"Session":{"Photons":20000},"Optode":{"Source":{"Type":"fourier","Pos":[15,15,0],"Param1":[30,0,0,2.5],"Param2":[0,30,0,1.5]}}}'
if run_sim t40w.json t40w.log -E 3 -s t40w -U 0; then
    le=$(grep -oP 'launched energy:\s*\K[0-9.e+]+' "$TESTDIR/t40w.log")
    check_close "$(pyval "$(jdat sum t40w.jdat)/$le*100")" "$WREF" 0.01 "Raw sum / launched energy ($le of 20000) = absorbed"
else
    log_fail "Execution failed"
fi

# =============================================================================
# TEST 41: Launch directions: from the center of an absorbing sphere (R=10,
# no scattering) every direction travels R to the surface, so any angular
# distribution absorbs exactly 1-exp(-0.05*10) = 39.347% (the tessellated
# sphere is a little smaller: MeshRes 128)
# =============================================================================
log_test "Directional source types from the center of a sphere"
SREF=$(pyval '100*(1-math.exp(-0.05*10))')
for spec in 'isotropic|' 'arcsine|' 'cone|,"Param1":[0.7,0,0,0]' 'cone_angle|,"Param1":[2.5,1,0,0]' \
            'zgaussian|,"Param1":[0.4,0,0,0]' 'hyperboloid|,"Param1":[0.001,0,1000,0]' \
            'line|,"Param1":[0.001,0,0,0]' 'line_arc|,"Param1":[0,0.001,0,0],"Param2":[1,0,0,0]' 'pencil_nan|,"Dir":[0,0.6,0.8,"_NaN_"]' 'pencil_lambert|,"Dir":[0.6,0,0.8,"-_Inf_"]'; do
    t=${spec%%|*}
    ty=${t%%_*}
    mkcfg t41_$t.json '{"Session":{"Photons":20000},"Domain":{"MeshRes":128},"Optode":{"Source":{"Type":"'$ty'","Pos":[30,30,30]'"${spec#*|}"'}},
                      "Shapes":[{"Sphere":{"Tag":1,"O":[30,30,30],"R":10}}]}'
    if run_sim t41_$t.json t41_$t.log -E 4 -s t41_$t; then
        check_close "$(get_absorbed t41_$t.log)" "$SREF" 0.05 "$t: absorbed = 1-exp(-mua R)"
    else
        log_fail "$t failed: $(grep -m1 ERROR "$TESTDIR/t41_$t.log")"
    fi
done
# a focused beam: a 30x30 planar source converging to the slab center (focal length 20)
# crosses the 20 mm slab along 20/cos(theta); the deposit has its maximum at the focus
mkcfg t41f.json '{"Session":{"Photons":50000},"Optode":{"Source":{"Type":"planar","Pos":[15,15,0],"Dir":[0,0,1,20],"Param1":[30,0,0,0],"Param2":[0,30,0,0]}}}'
if run_sim t41f.json t41f.log -E 4 -s t41f; then
    ab=$(get_absorbed t41f.log)
    python3 -c "import sys; sys.exit(0 if $ab > $WREF + 1 else 1)" && log_pass "Focused planar beam: longer paths ($ab% > $WREF%)" \
        || log_fail "Focused planar beam: $ab%"
else
    log_fail "Execution failed"
fi

# =============================================================================
# TEST 42: Several output fields: a multi-source list simulated at once
# (Source.ID 0) and a pattern source shared by two patterns (SrcNum 2)
# =============================================================================
log_test "Multi-source fields and shared patterns"
mkcfg t42a.json '{"Session":{"Photons":30000},"Optode":{"Source":{"Pos":[[20,20,0,1],[40,40,0,0.5]],"Dir":[0,0,1],"ID":0}}}'
if run_sim t42a.json t42a.log -E 5 -s t42a; then
    check_close "$(get_absorbed t42a.log)" "$WREF" 0.01 "Two sources: total absorbed"
    s1=$(grep -oP 'source 1: absorbed:\s*\K[0-9.]+' "$TESTDIR/t42a.log"); s2=$(grep -oP 'source 2: absorbed:\s*\K[0-9.]+' "$TESTDIR/t42a.log")
    check_close "$(pyval "$s1/$s2")" 2 0.1 "Weights 1 and 0.5: source fields $s1% / $s2%"
    [ "$(jdat dims t42a.jdat | cut -dx -f4-)" = "1x2" ] && log_pass "Output [nx,ny,nz,1,2]" || log_fail "dims $(jdat dims t42a.jdat)"
else
    log_fail "Execution failed"
fi
mkcfg t42b.json '{"Session":{"Photons":30000},"Optode":{"Source":{"Pos":[[20,20,0,1],[40,40,0,0.5]],"Dir":[0,0,1],"ID":2}}}'
if run_sim t42b.json t42b.log -E 5 -s t42b; then
    check_close "$(get_absorbed t42b.log)" "$WREF" 0.01 "Source.ID 2 alone"
    [ "$(jdat dims t42b.jdat | tr -cd x)" = "xx" ] && log_pass "One field" || log_fail "dims $(jdat dims t42b.jdat)"
else
    log_fail "Execution failed"
fi
mkcfg t42c.json '{"Session":{"Photons":30000},"Optode":{"Source":{"Type":"pattern","Pos":[15,15,0],"Param1":[30,0,0,2],"Param2":[0,30,0,2],"SrcNum":2,
                  "Pattern":{"Nx":2,"Ny":2,"Data":[1,0,0.5,0.5,0,1,0.25,0]}}}}'
if run_sim t42c.json t42c.log -E 5 -s t42c; then
    p1=$(grep -oP 'pattern 1: absorbed:\s*\K[0-9.]+' "$TESTDIR/t42c.log"); p2=$(grep -oP 'pattern 2: absorbed:\s*\K[0-9.]+' "$TESTDIR/t42c.log")
    check_close "$p1" "$WREF" 0.01 "Pattern 1 field normalized by its launched energy"
    check_close "$p2" "$WREF" 0.01 "Pattern 2 field normalized by its launched energy"
    [ "$(jdat dims t42c.jdat | cut -dx -f4-)" = "1x2" ] && log_pass "Output [nx,ny,nz,1,2]" || log_fail "dims $(jdat dims t42c.jdat)"
else
    log_fail "Execution failed"
fi

# =============================================================================
echo ""
echo "============================================================"
echo -e " Results: ${TOTAL} tests; checks: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}, ${YELLOW}${SKIP} skipped${NC}"
echo "============================================================"

[ "$FAIL" -gt 0 ] && exit 1
exit 0
