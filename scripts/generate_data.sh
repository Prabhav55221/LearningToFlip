#!/usr/bin/env bash
# Generate all instance families (k-coloring and random 3-SAT).
# Instances are seeded by their index for reproducibility.
# Skips files that already exist, so safe to re-run after partial generation.
#
# Usage:
#   bash scripts/generate_data.sh           # full dataset (2500 per family/scale)
#   bash scripts/generate_data.sh --small   # small dataset (50 per family/scale)

set -euo pipefail

SMALL=false
[[ "${1:-}" == "--small" ]] && SMALL=true

if $SMALL; then
    N_TRAIN=20; N_VAL=10; N_TEST=20
else
    N_TRAIN=1900; N_VAL=100; N_TEST=500
fi

# generate family scale split n cnfgen_args...
generate() {
    local family=$1 scale=$2 split=$3 n=$4
    shift 4
    local dir="data/${family}/${scale}/${split}"
    mkdir -p "$dir"
    local count=0
    for i in $(seq 0 $((n - 1))); do
        local out
        out=$(printf "%s/instance_%04d.cnf" "$dir" "$i")
        if [[ ! -f "$out" ]]; then
            cnfgen --seed "$i" "$@" > "$out"
            count=$((count + 1))
        fi
    done
    echo "  ${family}/${scale}/${split}: ${n} instances (${count} new)"
}

echo "==> Generating k-coloring instances..."
for split in train val test; do
    case $split in
        train) N=$N_TRAIN ;;
        val)   N=$N_VAL ;;
        test)  N=$N_TEST ;;
    esac
    generate kcoloring n100 "$split" "$N" kcolor 5 gnp 20 0.5
    generate kcoloring n200 "$split" "$N" kcolor 5 gnp 40 0.5
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
