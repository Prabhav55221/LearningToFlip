# Learning to Flip

Reinforcement learning for the flip heuristic in Stochastic Local Search (SLS) SAT solvers.

SLS solvers (e.g., WalkSAT) maintain a complete variable assignment and iteratively flip variables to reduce unsatisfied clauses. The central decision at each step: *given a randomly chosen unsatisfied clause, which variable in it should be flipped?* This project learns that decision from structured SAT instance families via REINFORCE, replacing hand-designed heuristics (min-break, Novelty+).

## Key Contributions

1. **MLP policy** over local per-variable features, replacing the linear scorer of Interian & Bernardini (KR 2023)
2. **k-step discounted returns** instead of full-episode returns — tighter credit assignment
3. **Online instance-specific adaptation** — gradient steps interleaved with the SLS search at test time

## Hypotheses

| | |
|---|---|
| H1 | MLP > linear on structured families (k-coloring) |
| H2 | No improvement on random 3-SAT (unstructured control) |
| H3 | No cross-family transfer of frozen learned heuristics |
| H4 | Online adaptation > frozen model on structured families |

## Setup

```bash
conda env create -f environment.yml
conda activate l2f
```

## Usage

```bash
# 1. Generate instances
python scripts/generate.py --data-dir data/

# 2. Train a policy
python scripts/train.py --config experiments/configs/base.yaml

# 3. Evaluate
python scripts/evaluate.py --config experiments/configs/base.yaml --policy mlp --checkpoint path/to/weights.pt

# Evaluate a classical baseline (no checkpoint needed)
python scripts/evaluate.py --config experiments/configs/base.yaml --policy minbreak
```

## Structure

```
src/
├── sat/        # DIMACS parser + incremental WalkSAT state (make/break/unsat_deg)
├── sls/        # Policy-agnostic SLS loop
├── policy/     # All policies — baselines (MinBreak, Novelty+) and learned (Linear, MLP)
├── train/      # REINFORCE with k-step returns; online adaptation
└── eval/       # Metrics: median flips, solve rate, CDF curves

data/           # Generated .cnf instances ({family}/{scale}/{split}/)
scripts/        # Entry points: generate, train, evaluate
experiments/    # YAML configs + result logs
CLAUDE/         # Research notes, design decisions, advisor correspondence
```

All policies implement the same `Policy` protocol (`src/policy/base.py`) and are interchangeable in the SLS loop and evaluation runner.

## Instance Families

| Family | Parameters | SAT variables | CNFgen |
|--------|-----------|---------------|--------|
| Random 3-SAT | phase transition α=4.26 | 100, 200 | `randkcnf 3 n m` |
| Graph 5-coloring | Erdős–Rényi G(N, 0.5) | 100, 200 | `kcolor 5 gnp N 0.5` |

1,900 train / 100 val / 500 test instances per family. Evaluation: max_flips=10,000 (n=100) or 50,000 (n=200), max_tries=10.

## References

- Interian & Bernardini, *Learning Interpretable Heuristics for WalkSAT*, KR 2023
- Yolcu & Póczos, *Learning Local Search Heuristics for Boolean Satisfiability*, NeurIPS 2019
- Konda & Tsitsiklis, *Actor-Critic Algorithms*, NeurIPS 1999
