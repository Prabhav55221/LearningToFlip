#!/usr/bin/env bash
# Generate all SAT instance families needed for experiments.
#
# Training domains (full train/val/test splits):
#   kcoloring:   n50, n75, n100
#   random_3sat: n40, n50, n60, n70
#   kclique:     n5, n10, n15, n20      (Interian paper scales)
#   domset:      n5, n7, n9, n12        (Interian paper scales)
#
# Generalization test sets (test split only, small N):
#   kcoloring:   n200, n300
#   random_3sat: n100, n200
#
# Satisfiability filtering: every instance is verified SAT via PySAT/Glucose3.
# UNSAT instances are discarded and the next seed is tried.
#
# Seed offsets per split prevent cross-split collisions:
#   train: 0+      val: 100000+      test: 200000+
#
# Safe to re-run — existing files are counted and skipped.
#
# Usage:
#   bash scripts/generate_data.sh            # full dataset
#   bash scripts/generate_data.sh --small    # 20 train / 10 val / 20 test

set -euo pipefail

if ! python -c "from pysat.solvers import Glucose3" 2>/dev/null; then
    echo "ERROR: python-sat not installed. Run: pip install python-sat" >&2
    exit 1
fi

SMALL=false
[[ "${1:-}" == "--small" ]] && SMALL=true

if $SMALL; then
    N_TRAIN=20; N_VAL=10; N_TEST=20; N_GENTEST=10
else
    N_TRAIN=1900; N_VAL=100; N_TEST=500; N_GENTEST=100
fi

# generate <family> <scale> <split> <n> <cnfgen_args...>
generate() {
    local family=$1 scale=$2 split=$3 n=$4
    shift 4
    local dir="data/${family}/${scale}/${split}"
    mkdir -p "$dir"

    local existing
    existing=$(find "$dir" -maxdepth 1 -name "*.cnf" -type f | wc -l | tr -d ' ')
    if [[ $existing -ge $n ]]; then
        echo "  ${family}/${scale}/${split}: ${n} instances (already complete)"
        return
    fi

    local seed_base
    case $split in
        train) seed_base=0 ;;
        val)   seed_base=100000 ;;
        test)  seed_base=200000 ;;
    esac

    local collected=$existing attempt=0 new_count=0 tmp

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
            printf "  [%s/%s/%s] %d/%d (tried %d seeds)\r" \
                "$family" "$scale" "$split" "$collected" "$n" "$attempt"
        else
            rm "$tmp"
        fi
        attempt=$(( attempt + 1 ))
    done

    printf "\n"
    echo "  ${family}/${scale}/${split}: ${n} instances (${new_count} new, tried ${attempt} seeds)"
}

# generate_test_only — generalization test sets, test split only
generate_test_only() {
    local family=$1 scale=$2 n=$3
    shift 3
    generate "$family" "$scale" test "$n" "$@"
}

# ── k-coloring: n vars = n_nodes × 5 colors ────────────────────────────────
echo "==> Generating k-coloring instances (training splits)..."
for split in train val test; do
    case $split in train) N=$N_TRAIN ;; val) N=$N_VAL ;; test) N=$N_TEST ;; esac
    generate kcoloring n50  "$split" "$N" kcolor 5 gnp 10 0.5
    generate kcoloring n75  "$split" "$N" kcolor 5 gnp 15 0.5
    generate kcoloring n100 "$split" "$N" kcolor 5 gnp 20 0.5
done

echo "==> Generating k-coloring generalization test sets..."
# Lower density (0.2/0.15 vs 0.5) keeps χ < 5 for larger graphs.
# G(40,0.5) has χ≈6-8 empirically, making 5-coloring UNSAT for almost all instances.
generate_test_only kcoloring n200 "$N_GENTEST" kcolor 5 gnp 40 0.2
generate_test_only kcoloring n300 "$N_GENTEST" kcolor 5 gnp 60 0.15

# ── random 3-SAT: at phase transition (ratio ≈ 4.27) ───────────────────────
echo "==> Generating random 3-SAT instances (training splits)..."
for split in train val test; do
    case $split in train) N=$N_TRAIN ;; val) N=$N_VAL ;; test) N=$N_TEST ;; esac
    generate random_3sat n40 "$split" "$N" randkcnf 3 40  171
    generate random_3sat n50 "$split" "$N" randkcnf 3 50  213
    generate random_3sat n60 "$split" "$N" randkcnf 3 60  256
    generate random_3sat n70 "$split" "$N" randkcnf 3 70  299
done

echo "==> Generating random 3-SAT generalization test sets..."
generate_test_only random_3sat n100 "$N_GENTEST" randkcnf 3 100 426
generate_test_only random_3sat n200 "$N_GENTEST" randkcnf 3 200 852

# ── k-clique: n vars = n_nodes (find 3-clique in random graph) ─────────────
# Scales match Interian & Bernardini (KR 2023) exactly.
echo "==> Generating k-clique instances (training splits)..."
for split in train val test; do
    case $split in train) N=$N_TRAIN ;; val) N=$N_VAL ;; test) N=$N_TEST ;; esac
    generate kclique n5  "$split" "$N" kclique 3 gnp 5  0.2
    generate kclique n10 "$split" "$N" kclique 3 gnp 10 0.1
    generate kclique n15 "$split" "$N" kclique 3 gnp 15 0.066
    generate kclique n20 "$split" "$N" kclique 3 gnp 20 0.05
done

# ── dominating set: n vars = n_nodes ───────────────────────────────────────
# Scales match Interian & Bernardini (KR 2023) exactly.
echo "==> Generating dominating set instances (training splits)..."
for split in train val test; do
    case $split in train) N=$N_TRAIN ;; val) N=$N_VAL ;; test) N=$N_TEST ;; esac
    generate domset n5  "$split" "$N" domset 3 gnp 5  0.2
    generate domset n7  "$split" "$N" domset 3 gnp 7  0.2
    generate domset n9  "$split" "$N" domset 3 gnp 9  0.2
    generate domset n12 "$split" "$N" domset 4 gnp 12 0.2
done

echo "==> Done."
