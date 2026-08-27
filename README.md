# Coherent projector-overlap QFNNs with Chebyshev responses

This repository accompanies the manuscript **“Coherent Projector-Overlap QFNNs with Chebyshev Responses: Expressivity, Approximation, and Process Lifts.”** It contains the reference implementation, theorem-level numerical sanity checks, the complete hyperparameter-search and final-evaluation records, and the Version 7 manuscript source.

The architecture treats trainable projector overlaps as quantum analogues of pre-activations. Alternating reflections transform an overlap probability \(p\) into an exact Chebyshev response

\[
T_d(2p-1),
\]

while retaining the quantum registers coherently so that layers can be composed without intermediate measurement.

## Repository contents

```text
.
├── src/
│   ├── qfnn_experiment_pipeline.py   # Optuna search and final evaluation
│   └── qfnn_theorem_sanity_checks.py # numerical checks used in the paper
├── results/qfnn_experiments/
│   ├── studies/                      # three resumable Optuna databases
│   ├── search/                       # all trial and search-seed records
│   ├── final/                        # 18 final runs and checkpoints
│   ├── reports/                      # aggregate tables, curves, and plots
│   ├── run_config.json
│   └── selected_recipes.{json,md}
└── paper/
    ├── qfnn_chebyshev_revised_v7.tex
    ├── qfnn_chebyshev_revised_v7.pdf
    └── Springer/LaTeX support files
```

The complete artifact directory is committed in unpacked form so that individual configurations, trials, checkpoints, and reports can be inspected directly. See [`results/README.md`](results/README.md) for a map of the experiment records.

## Main numerical results

The architecture was evaluated on the classes \(0,\ldots,7\) of MNIST, Fashion-MNIST, and KMNIST. Hyperparameters were selected separately for each dataset using a fixed validation split of the official training partition. The official test partition was evaluated only after the six final validation-selected checkpoints were fixed.

| Dataset | Selected depth / degree | Trainable parameters | Exact test accuracy | Finite-shot test accuracy |
|---|---:|---:|---:|---:|
| MNIST8 | 2 / 3 | 409 | 90.57 ± 0.20% | 90.51 ± 0.17% |
| FashionMNIST8 | 2 / 2 | 409 | 74.55 ± 0.51% | 73.96 ± 0.48% |
| KMNIST8 | 2 / 2 | 409 | 67.61 ± 0.65% | 67.21 ± 0.69% |

Values are means ± sample standard deviations over six independent final seeds. The finite-shot column uses 6,000 shots and five repetitions per seed. Full tables are available in [`final_results.md`](results/qfnn_experiments/reports/final_results.md), and the selected recipes are recorded in [`selected_recipes.md`](results/qfnn_experiments/selected_recipes.md).

## Installation

Python 3.12 was used for the archived runs. Create an isolated environment and install the recorded core dependencies:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The archived CPU run records Python 3.12.3, NumPy 2.5.0, PyTorch 2.13.0+cpu, torchvision 0.28.0, and Optuna 4.9.0. A compatible CPU or CUDA PyTorch build may be substituted; PyTorch installation commands can vary by platform.

## Quick verification

The pipeline has a dataset-free self-test that checks both supported depths, the Chebyshev response identities, and gradient propagation:

```bash
python src/qfnn_experiment_pipeline.py --stage self-test
```

The separate theorem checks reproduce the finite-dimensional identities and diagnostics reported in the manuscript:

```bash
python src/qfnn_theorem_sanity_checks.py
```

For a small end-to-end smoke run:

```bash
python src/qfnn_experiment_pipeline.py \
  --stage all \
  --quick \
  --device cpu \
  --outdir qfnn_experiments_smoke
```

## Reproducing the full study

The published study used the script defaults: one independent 60-trial Optuna study per dataset, two training seeds per trial, 30 epochs per search-seed run, and final retraining for 100 epochs over seeds 41–46. Depth \(L\in\{1,2\}\) and degree \(d\in\{1,2,3\}\) were included in each dataset-specific search.

The recommended staged workflow is:

```bash
python src/qfnn_experiment_pipeline.py \
  --stage search \
  --device cpu \
  --outdir qfnn_experiments_reproduction

python src/qfnn_experiment_pipeline.py \
  --stage final \
  --device cpu \
  --outdir qfnn_experiments_reproduction
```

Use `--device cuda` when a compatible CUDA installation is available. The search stage is resumable: Optuna state is stored in the SQLite databases under `studies/`, and completed trial outputs are cached. The final stage also reuses completed per-seed outputs unless `--force` is supplied.

To request both stages in one invocation:

```bash
python src/qfnn_experiment_pipeline.py \
  --stage all \
  --device cpu \
  --outdir qfnn_experiments_reproduction
```

## Reproducibility notes

- The exact generating script has SHA-256 `9facd9172f676e57ac0f0cb789fad8a2d374a9229e304e7fdf3cae2e4c8aa7bc`.
- The official test data is not used by Optuna, recipe selection, epoch selection, or seed selection.
- Search uses classes 0–7, up to 300 training examples per class, and 200 validation examples per class.
- Final training uses up to 1,000 training examples per class and evaluates the complete class-0–7 official test subset.
- Finite-shot training draws once from the full output multinomial distribution. Exact-statevector and repeated finite-shot test results are both retained.
- The raw JSON records preserve original provenance strings, including historical local output paths, rather than rewriting the completed artifacts after the run.

## Citation

GitHub can generate citation metadata from [`CITATION.cff`](CITATION.cff). Until a DOI is available, cite the manuscript title and this repository URL.

## License

The original Python source and repository documentation are released under the [MIT License](LICENSE). The manuscript and third-party Springer Nature/LaTeX support files remain subject to their applicable author, publisher, and upstream licensing terms; see [`paper/THIRD_PARTY_NOTICES.md`](paper/THIRD_PARTY_NOTICES.md).
