#!/usr/bin/env bash
# Generate generalization test sets only (test split, 100 instances each).
# Safe to re-run — existing files are counted and skipped.
#
# Usage:
#   bash scripts/generate_gen_data.sh           # all families
#   bash scripts/generate_gen_data.sh kcoloring # just kcoloring
#   bash scripts/generate_gen_data.sh kclique   # just kclique

set -euo pipefail

if ! python -c "from pysat.solvers import Glucose3" 2>/dev/null; then
    echo "ERROR: python-sat not installed. Run: pip install python-sat" >&2
    exit 1
fi

FAMILY="${1:-all}"
N=100

generate_test_only() {
    local family=$1 scale=$2 n=$3
    shift 3
    local dir="data/${family}/${scale}/test"
    mkdir -p "$dir"

    local existing
    existing=$(find "$dir" -maxdepth 1 -name "*.cnf" -type f | wc -l | tr -d ' ')
    if [[ $existing -ge $n ]]; then
        echo "  ${family}/${scale}/test: ${n} instances (already complete)"
        return
    fi

    local seed_base=200000 collected=$existing attempt=0 new_count=0 tmp

    while [[ $collected -lt $n ]]; do
        local seed=$(( seed_base + attempt ))
        local out
        out=$(printf "%s/instance_%04d.cnf" "$dir" "$collected")
        tmp=$(mktemp /tmp/l2f_XXXXXX.cnf)
        cnfgen --seed "$seed" "$@" > "$tmp"
        if python scripts/check_sat.py "$tmp"; then
            mv "$tmp" "$out"
            collected=$(( collected + 1 ))
            new_count=$(( new_count + 1 ))
            printf "  [%s/%s/test] %d/%d (tried %d seeds)\r" \
                "$family" "$scale" "$collected" "$n" "$attempt"
        else
            rm "$tmp"
        fi
        attempt=$(( attempt + 1 ))
    done

    printf "\n"
    echo "  ${family}/${scale}/test: ${n} instances (${new_count} new, tried ${attempt} seeds)"
}

if [[ "$FAMILY" == "all" || "$FAMILY" == "kcoloring" ]]; then
    echo "==> Generating k-coloring generalization test sets..."
    generate_test_only kcoloring n10  "$N" kcolor 5 gnp 2  0.5
    generate_test_only kcoloring n20  "$N" kcolor 5 gnp 4  0.5
    generate_test_only kcoloring n30  "$N" kcolor 5 gnp 6  0.5
    generate_test_only kcoloring n200 "$N" kcolor 5 gnp 40 0.2
    generate_test_only kcoloring n300 "$N" kcolor 5 gnp 60 0.15
fi

if [[ "$FAMILY" == "all" || "$FAMILY" == "kclique" ]]; then
    echo "==> Generating k-clique generalization test sets..."
    generate_test_only kclique n3  "$N" kclique 3 gnp 3  0.5
    generate_test_only kclique n12 "$N" kclique 3 gnp 12 0.083
fi

if [[ "$FAMILY" == "all" || "$FAMILY" == "random_3sat" ]]; then
    echo "==> Generating random 3-SAT generalization test sets..."
    generate_test_only random_3sat n100 "$N" randkcnf 3 100 426
    generate_test_only random_3sat n200 "$N" randkcnf 3 200 852
fi

echo "==> Done."
