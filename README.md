# Learning to Flip

`Learning to Flip` studies whether the core variable-selection step in stochastic local search (SLS) for SAT can be learned from data. In a `WalkSAT`-style solver, one first samples an unsatisfied clause and then decides which variable in that clause to flip. This repository keeps that clause-local interface fixed and learns the flip heuristic itself, using reinforcement learning over structured satisfiable SAT families. The main learned method is a compact nonlinear scorer over local search features; the repo also includes classical baselines, a reimplementation of the linear Interian–Bernardini policy, offline training code, and online test-time adaptation code.

In the final experimental story, the nonlinear policy is strongest in-domain on the structured families we study, often retains useful behavior under within-family size shift, and shows meaningful but asymmetric cross-family transfer. Online adaptation helps in the size-shift setting. At the same time, the method does **not** yield a robust overall win on random `3`-SAT: it can produce very short successful trajectories, but its failure tail is much heavier there, whereas the simpler Interian baseline remains better calibrated.

## Repository Layout

The repository is organized around a small policy-agnostic SLS core plus a set of runnable scripts:

```text
src/
├── policy/
│   ├── base.py              # common policy interface
│   ├── baselines.py         # MinBreak and Novelty+
│   ├── linear.py            # Interian-style linear scorer
│   ├── mlp.py               # our nonlinear scorer
│   └── features.py          # local feature definitions
├── sat/
│   ├── parser.py            # DIMACS parsing
│   └── state.py             # incremental SLS state and local statistics
├── sls/
│   └── solver.py            # policy-agnostic restart-based SLS loop
├── train/
│   ├── reinforce.py         # offline REINFORCE training
│   ├── interian_reinforce.py
│   └── online.py            # online KL-based adaptation
└── utils/
    └── logging.py

scripts/
├── generate_data.sh         # regenerate the full dataset
├── generate_gen_data.sh     # regenerate generalization-only test sets
├── train.py                 # train our learned policy offline
├── train_interian.py        # train the Interian baseline
├── evaluate.py              # evaluate a single policy/checkpoint on one split
├── eval_generalization.py   # frozen + online generalization evaluation
├── plot_in_domain.py        # matched-family/size evaluation + plots
└── eval_random_3sat.py      # random 3-SAT evaluation + plots

slurmrun/
├── ours/run7/               # final method training/eval jobs
├── ours/run6e/              # earlier ablation jobs
├── interian/                # Interian training/eval jobs
└── generalization/          # size-shift and cross-family jobs

data/                        # generated CNF instances
experiments/                 # models, results, plots, and summaries
```

## Data

The dataset used in the experiments is already generated in `data/`. The tree is organized as:

```text
data/{family}/{scale}/{split}/*.cnf
```

Families in the current codebase:

- `kcoloring`
- `kclique`
- `domset`
- `random_3sat`

Training/validation/test sets are SAT-filtered during generation using PySAT/Glucose3, so the stored instances are already satisfiable. If you want to regenerate everything from scratch:

```bash
bash scripts/generate_data.sh
```

For a small smoke-test version:

```bash
bash scripts/generate_data.sh --small
```

To regenerate only the generalization test sets:

```bash
bash scripts/generate_gen_data.sh
```

## How to Run the Code

### 1. Offline training: our method

The main offline training entry point is `scripts/train.py`. A typical local run for the final method (`run7`) looks like:

```bash
python scripts/train.py \
  --family domset \
  --scale n7 \
  --policy mlp \
  --feature-set full \
  --hidden-dim 32 \
  --n-layers 1 \
  --entropy-coef 0.01 \
  --kl-anchor-coef 0.01 \
  --epochs 60 \
  --warmup-epochs 5 \
  --gamma 0.5 \
  --lr 1e-3 \
  --seed 42 \
  --wandb
```

Checkpoints are written under:

```text
experiments/models/{family}/{scale}/{run_name}/
```

The exact Slurm jobs used for the paper are in:

- `slurmrun/ours/run7/`
- `slurmrun/ours/run6e/`

If you want the exact final paper jobs rather than composing commands manually, start there.

### 2. Offline training: Interian baseline

The linear learned baseline is trained with:

```bash
python scripts/train_interian.py \
  --family domset \
  --scale n7 \
  --epochs 60 \
  --warmup-epochs 5 \
  --gamma 0.5 \
  --max-lr 1e-3 \
  --seed 42 \
  --wandb
```

Its paper jobs live in:

- `slurmrun/interian/`

### 3. Evaluate one policy on one split

Use `scripts/evaluate.py` for direct evaluation of a single policy/checkpoint on a single family/scale/split.

Classical baselines:

```bash
python scripts/evaluate.py \
  --family kcoloring \
  --scale n50 \
  --split test \
  --policies minbreak noveltyplus \
  --max-tries 10
```

Interian:

```bash
python scripts/evaluate.py \
  --family kcoloring \
  --scale n50 \
  --split test \
  --policies interian \
  --model-path experiments/models/kcoloring/n50/interian/best_interian.pt \
  --max-tries 10
```

Final nonlinear method (`run7` / `LTF-KL`):

```bash
python scripts/evaluate.py \
  --family kcoloring \
  --scale n50 \
  --split test \
  --policies mlp \
  --feature-set full \
  --hidden-dim 32 \
  --n-layers 1 \
  --model-path experiments/models/kcoloring/n50/mlp_full_e_kl_sm/best_mlp_full_e_kl_sm.pt \
  --max-tries 10
```

### 4. In-domain evaluation and plots

Matched-family/matched-size evaluation for the paper tables and plots:

```bash
python scripts/plot_in_domain.py
```

This reuses cached raw results if they exist, writes a summary CSV, and produces per-family plots under `experiments/plots/in_domain_test/`.

### 5. Online adaptation and generalization

The generalization entry point is `scripts/eval_generalization.py`. It supports:

- frozen evaluation of the learned checkpoint
- `ours_online_kl`
- `ours_online_success_kl`

Example:

```bash
python scripts/eval_generalization.py \
  --family domset \
  --train-scale n7 \
  --model-path experiments/models/domset/n7/mlp_full_e_kl_sm/best_mlp_full_e_kl_sm.pt \
  --model-label run7 \
  --interian-model experiments/models/domset/n7/interian/best_interian.pt \
  --methods minbreak noveltyplus interian ours_online_kl ours_online_success_kl \
  --split test \
  --max-tries 10
```

The Slurm jobs used for the size-shift and cross-family experiments are in:

- `slurmrun/generalization/domset_from_n7.sh`
- `slurmrun/generalization/kclique_from_n5.sh`
- `slurmrun/generalization/kcoloring_from_n50.sh`
- `slurmrun/generalization/cross_*`

### 6. Random 3-SAT evaluation

The dedicated random `3`-SAT evaluation/plotting script is:

```bash
python scripts/eval_random_3sat.py
```

It evaluates:

- `MinBreak`
- `Novelty+`
- `Interian`
- `LTF-KL` (`run7`)

on the random `3`-SAT `n40` and `n50` test sets, then writes a summary CSV and plots.

## Baselines

There are three baseline families in this repository:

- `WalkSAT-MinBreak`, implemented in `src/policy/baselines.py`
- `Novelty+`, implemented in `src/policy/baselines.py`
- the learned linear Interian–Bernardini policy, implemented in `src/policy/linear.py` and trained through `scripts/train_interian.py`

For exact experiment jobs, use:

- `slurmrun/interian/` for the Interian runs
- `scripts/evaluate.py` for direct baseline evaluation

## Notes on LLM Usage

Claude Code was used during development to help write and iterate on substantial parts of the codebase, always under direct human supervision. The baselines were taken either directly from their original repositories/papers or implemented by closely following the original algorithmic descriptions in the case of classical `WalkSAT` variants. The author checked the code paths, validated the major implementation choices, and wrote the major experimental logic. Codex was used primarily for documentation work and for generating or cleaning up script files.

## Citation

If you use this repository, please cite the accompanying paper and, if appropriate, the repository itself. The core prior works this codebase builds on are also listed below.

### Repository

```bibtex
@misc{singh2026learningtofliprepo,
  author       = {Prabhav Singh},
  title        = {Learning to Flip: Code Repository},
  year         = {2026},
  howpublished = {GitHub repository}
}
```

### Interian baseline

```bibtex
@inproceedings{interian2023learning,
  author    = {Yannet Interian and Sara Bernardini},
  title     = {Learning Interpretable Heuristics for WalkSAT},
  booktitle = {Proceedings of the 20th International Conference on Principles of Knowledge Representation and Reasoning},
  year      = {2023}
}
```

### WalkSAT / Novelty

```bibtex
@inproceedings{selman1994noise,
  author    = {Bart Selman and Henry A. Kautz and Bram Cohen},
  title     = {Noise Strategies for Improving Local Search},
  booktitle = {Proceedings of the Twelfth National Conference on Artificial Intelligence},
  year      = {1994}
}
```

```bibtex
@inproceedings{selman1993empirical,
  author    = {Bart Selman and Henry A. Kautz},
  title     = {An Empirical Study of Greedy Local Search for Satisfiability Testing},
  booktitle = {Proceedings of the Eleventh National Conference on Artificial Intelligence},
  year      = {1993}
}
```

### CNFgen

```bibtex
@inproceedings{lauria2017cnfgen,
  author    = {Massimo Lauria and Jan Elffers and Jakob Nordstr{\"o}m and Marc Vinyals},
  title     = {CNFgen: A Generator of Crafted Benchmarks},
  booktitle = {Theory and Applications of Satisfiability Testing -- SAT 2017},
  year      = {2017}
}
```
