#!/usr/bin/env bash
# Generate SAT instance families (k-coloring and random 3-SAT).
#
# Each instance is verified satisfiable via scripts/check_sat.py (PySAT/Glucose3).
# UNSAT instances are discarded and the next seed is tried, so the script always
# produces exactly N satisfiable instances regardless of the underlying SAT rate.
#
# Seeds are offset per split so train/val/test never share a seed:
#   train: seeds 0, 1, 2, ...
#   val:   seeds 100000, 100001, ...
#   test:  seeds 200000, 200001, ...
#
# Safe to re-run: existing files are counted and skipped.
#
# Usage:
#   bash scripts/generate_data.sh           # full dataset
#   bash scripts/generate_data.sh --small   # 20 train / 10 val / 20 test per family/scale

set -euo pipefail

# Preflight: verify python-sat is available before entering any loop
if ! python -c "from pysat.solvers import Glucose3" 2>/dev/null; then
    echo "ERROR: python-sat not installed. Run: pip install python-sat" >&2
    exit 1
fi

SMALL=false
[[ "${1:-}" == "--small" ]] && SMALL=true

if $SMALL; then
    N_TRAIN=20; N_VAL=10; N_TEST=20
else
    N_TRAIN=1900; N_VAL=100; N_TEST=500
fi

# generate <family> <scale> <split> <n> <cnfgen_args...>
generate() {
    local family=$1 scale=$2 split=$3 n=$4
    shift 4
    local dir="data/${family}/${scale}/${split}"
    mkdir -p "$dir"

    # Count already-present instances; skip if complete
    # Use find (not ls) — ls exits non-zero on empty dirs and kills the script under set -e
    local existing
    existing=$(find "$dir" -maxdepth 1 -name "*.cnf" -type f | wc -l | tr -d ' ')
    if [[ $existing -ge $n ]]; then
        echo "  ${family}/${scale}/${split}: ${n} instances (0 new, already complete)"
        return
    fi

    # Seed base offset per split avoids cross-split collisions
    local seed_base
    case $split in
        train) seed_base=0 ;;
        val)   seed_base=100000 ;;
        test)  seed_base=200000 ;;
    esac

    local collected=$existing
    local attempt=0
    local new_count=0
    local tmp

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
            printf "  [%s/%s/%s] %d/%d SAT (tried %d seeds)\r" \
                "$family" "$scale" "$split" "$collected" "$n" "$attempt"
        else
            rm "$tmp"
        fi
        attempt=$(( attempt + 1 ))
    done

    printf "\n"
    echo "  ${family}/${scale}/${split}: ${n} instances (${new_count} new, tried ${attempt} seeds)"
}

echo "==> Generating k-coloring instances..."
for split in train val test; do
    case $split in
        train) N=$N_TRAIN ;;
        val)   N=$N_VAL ;;
        test)  N=$N_TEST ;;
    esac
    generate kcoloring n100 "$split" "$N" kcolor 5 gnp 20 0.5
    generate kcoloring n200 "$split" "$N" kcolor 5 gnp 40 0.3
done

echo "==> Generating random 3-SAT instances..."
for split in train val test; do
    case $split in
        train) N=$N_TRAIN ;;
        val)   N=$N_VAL ;;
        test)  N=$N_TEST ;;
    esac
    generate random_3sat n100 "$split" "$N" randkcnf 3 100 426
    generate random_3sat n200 "$split" "$N" randkcnf 3 200 852
done

echo "==> Done."
