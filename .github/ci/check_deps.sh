#!/bin/bash
# check_deps.sh binary allowed-lib... : fail if the binary needs a shared
# library not in the allowed list (matched by prefix), and print the newest
# glibc symbol version it needs; with MAXGLIBC=2.17 also fail if it needs a
# newer glibc than that
set -e
bin=$1
shift
bad=0

for lib in $(objdump -p "$bin" | awk '/NEEDED/ {print $2}'); do
    ok=0
    for a in "$@"; do
        case "$lib" in
            "$a"*) ok=1 ;;
        esac
    done
    if [ $ok = 0 ]; then
        echo "ERROR: $bin needs $lib"
        bad=1
    else
        echo "  needs $lib"
    fi
done

newest=$(objdump -T "$bin" | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)
echo "  newest glibc symbol: $newest"

if [ -n "$MAXGLIBC" ] && [ "$(printf '%s\n%s\n' "GLIBC_$MAXGLIBC" "$newest" | sort -V | tail -1)" != "GLIBC_$MAXGLIBC" ]; then
    echo "ERROR: $bin needs $newest, newer than GLIBC_$MAXGLIBC"
    bad=1
fi

exit $bad
