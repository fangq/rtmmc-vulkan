#!/bin/bash
# check_deps.sh binary allowed-lib... : fail if the binary needs a shared
# library not in the allowed list (matched by prefix), and print the newest
# glibc symbol version it needs
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

echo "  newest glibc symbol: $(objdump -T "$bin" | grep -o 'GLIBC_[0-9.]*' | sort -V | tail -1)"
exit $bad
