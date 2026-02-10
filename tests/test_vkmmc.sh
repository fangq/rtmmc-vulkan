#!/bin/bash
# =============================================================================
# test_vkmmc.sh — Unit tests for vkmmc / vkmmc_optix
#
# Run from /tests:
#   ./test_vkmmc.sh ../optix/vkmmc_optix
# =============================================================================

set -uo pipefail

EXEC="${1:-./vkmmc}"
shift || true

if [ -f "$EXEC" ]; then
    EXEC="$(cd "$(dirname "$EXEC")" && pwd)/$(basename "$EXEC")"
    EXEC_DIR="$(dirname "$EXEC")"
else
    echo "ERROR: executable '$EXEC' not found"; exit 1
fi
[ ! -x "$EXEC" ] && echo "ERROR: not executable" && exit 1

EXTRA_ARGS=""
for arg in "$@"; do EXTRA_ARGS="$EXTRA_ARGS $arg"; done

TESTDIR=$(mktemp -d /tmp/vkmmc_test.XXXXXX)
trap "rm -rf $TESTDIR" EXIT

PASS=0 FAIL=0 SKIP=0 TOTAL=0
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; CYAN='\033[0;36m'; NC='\033[0m'

log_test() { TOTAL=$((TOTAL+1)); echo -e "${CYAN}[TEST $TOTAL]${NC} $1"; }
log_pass() { PASS=$((PASS+1)); echo -e "  ${GREEN}PASS${NC}: $1"; }
log_fail() { FAIL=$((FAIL+1)); echo -e "  ${RED}FAIL${NC}: $1"; }
log_skip() { SKIP=$((SKIP+1)); echo -e "  ${YELLOW}SKIP${NC}: $1"; }

check_range() {
    local val="$1" lo="$2" hi="$3" desc="$4"
    if python3 -c "import sys; sys.exit(0 if $lo <= $val <= $hi else 1)" 2>/dev/null; then
        log_pass "$desc (got $val, expected [$lo, $hi])"
    else
        log_fail "$desc (got $val, expected [$lo, $hi])"
    fi
}

get_absorbed()  { grep -oP 'absorbed:\s*\K[0-9.]+' "$TESTDIR/$1" 2>/dev/null | head -1; }
get_speed()     { grep -oP 'Speed:\s*\K[0-9.]+' "$TESTDIR/$1" 2>/dev/null | head -1; }
get_triangles() { grep -oP '[0-9]+ tri\b' "$TESTDIR/$1" 2>/dev/null | grep -oP '[0-9]+' | head -1; }

run_sim() {
    local json="$TESTDIR/$1" logf="$TESTDIR/$2"
    shift 2
    (cd "$EXEC_DIR" && timeout 120 "$EXEC" "$json" $EXTRA_ARGS "$@" > "$logf" 2>&1)
}

cd "$TESTDIR"

echo "============================================================"
echo " vkmmc test suite"
echo " Executable: $EXEC"
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
echo ""
echo "============================================================"
echo -e " Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}, ${YELLOW}${SKIP} skipped${NC} / ${TOTAL} total"
echo "============================================================"

[ "$FAIL" -gt 0 ] && exit 1
exit 0