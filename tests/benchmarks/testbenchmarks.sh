#!/bin/bash
#
# testbenchmarks.sh - run the benchmark inputs under tests/benchmarks and compare their absorbed
#                     fractions with the reference values in golden.txt
#
# usage: ./testbenchmarks.sh [path/to/vkmmc_or_vkmmc_optix] [extra vkmmc parameters]
#
# golden.txt lists one input per line: set/name, the photon number of the reference, the
# reference absorbed fraction(s) in %, the solver that produced it and, optionally, the reason
# the input is skipped. Each input runs with its reference's photon number and a fixed seed; a
# fraction passes when it lies within 5 binomial standard errors (of the run and the reference)
# plus RELTOL of the reference. The runs save the energy deposit (-O x), so that vkmmc reports
# the absorbed fraction. With several sources, the total and each source are compared when the
# reference lists the same number of values; otherwise only the total.
#
# environment variables:
#    BIN:          simulator binary (default ../../bin/vkmmc); the first argument overrides it
#    MMC_BUILTIN:  folder of MMC's test/builtin inputs, for the mmc/ inputs too large to keep here
#    NPHOTON:      photon number of every run (default: that of each reference)
#    SEED:         RNG seed (default 1648335518)
#    RELTOL:       relative tolerance of the absorbed fraction (default 0.02)
#    TIMEOUT:      seconds allowed for each run (default 600)
#    ONLY:         run only the inputs whose set/name matches this extended regular expression

TESTDIR=$(cd "$(dirname "$0")" && pwd)
BIN=${BIN:-$TESTDIR/../../bin/vkmmc}

if [ -n "$1" ] && [ "${1#-}" = "$1" ]; then
    BIN=$1
    shift
fi

PARAM="$@"
SEED=${SEED:-1648335518}
RELTOL=${RELTOL:-0.02}
TIMEOUT=${TIMEOUT:-600}

if [ ! -x "$BIN" ]; then
    echo "simulator binary $BIN is not found"
    exit 1
fi

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

pass=0
fail=0
skip=0

while read -r name nphoton golden ref reason; do
    case "$name" in
        "" | \#*) continue ;;
    esac

    if [ -n "$ONLY" ] && ! echo "$name" | grep -q -E "$ONLY"; then
        continue
    fi

    if [ -n "$reason" ]; then
        echo "$name: skipped, $reason"
        skip=$((skip + 1))
        continue
    fi

    json=$TESTDIR/$name.json

    if [ ! -f "$json" ] && [ -n "$MMC_BUILTIN" ] && [ "${name%%/*}" = mmc ]; then
        json=$MMC_BUILTIN/${name#mmc/}.json
    fi

    if [ ! -f "$json" ]; then
        echo "$name: skipped, input not found (set MMC_BUILTIN to MMC's test/builtin folder)"
        skip=$((skip + 1))
        continue
    fi

    n=${NPHOTON:-$nphoton}
    log=$(cd "$WORKDIR" && timeout "$TIMEOUT" "$BIN" -f "$json" -n "$n" -E "$SEED" -s run -O x $PARAM 2>&1)
    rm -f "$WORKDIR"/*
    absorbed=$(echo "$log" | grep -oP 'absorbed:\s*\K[0-9.]+' | paste -sd, -)

    if [ -z "$absorbed" ]; then
        echo "$name: FAILED, $( (echo "$log" | grep -m1 -i 'error' || echo "$log" | tail -n 1) | cut -c1-120)"
        fail=$((fail + 1))
        continue
    fi

    if [ "$golden" = none ]; then
        echo "$name: absorbed $absorbed% (no reference)"
        pass=$((pass + 1))
        continue
    fi

    verdict=$(python3 -c "
import math
a, g = [float(x) for x in '$absorbed'.split(',')], [float(x) for x in '$golden'.split(',')]
n, ng = float('$n'), float('$nphoton')
if len(a) != len(g):
    a, g = a[:1], g[:1]    # per-source values only when both list them alike; mmc lists one per run
ok = all(abs(x - y) <= 5 * math.sqrt(max(y * (100 - y), 1e-3) * (1 / n + 1 / ng)) + $RELTOL * y for x, y in zip(a, g))
print('ok' if ok else 'DIFF')")
    echo "$name: absorbed ${absorbed%%,*}% vs $ref ${golden%%,*}% $verdict"

    if [ "$verdict" = ok ]; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1))
    fi
done < "$TESTDIR/golden.txt"

echo "passed $pass, failed $fail, skipped $skip benchmark inputs"
[ $fail -eq 0 ]
