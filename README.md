# Outcome-Aware Recovery of Cloud Deployment Workflows Under Ambiguous API Responses

This repository contains the experiment code, synthetic data, result tables, and figures for a cost-sensitive CART recovery policy. The policy combines reconciliation evidence with a Bayes-risk action rule to choose among six responses to ambiguous cloud control-plane API calls. The modeled application is an electric-vehicle charging-network backend.

## Repository contents

- `code/`: experiment scripts, code notes, and figure sources.
- `data/`: synthetic input CSVs and their data dictionaries.
- `results/`: supplied CSV results, JPG figures, and `metrics.json`.
- `requirements.txt`: Python dependencies for installation from the repository root.

## Data

`data/A5P1_ambiguous_api_calls.csv` contains 15,000 synthetic ambiguous calls and 25 columns. The data dictionary is [`data/A5P1_README.md`](data/A5P1_README.md). These are simulated records, not operational customer logs.

## Environment and installation

Python 3.10 or newer is the source-declared baseline. The reported experiments used Python 3.13. Python dependencies are listed in `requirements.txt`. Graphviz must also be installed as a system package, with the `dot` executable available on `PATH`, because the full pipeline renders Figure 1.

Run these commands from the repository root, after selecting the appropriate Python interpreter:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

## Reproduce the experiments

From the repository root:

```sh
OMP_NUM_THREADS=1 python code/run_all.py
```

The script resolves input and output paths relative to its own `__file__`, so `python code/run_all.py` works from the repository root. The source's alternative `cd code` followed by `python run_all.py` also works. The experiments use fixed seeds and CPU execution. Rerunning the pipeline rewrites the result files in `results/` and creates PNGs in `figures/`.

Figure 1 is rendered by `code/flowchart.py`, which the full pipeline calls automatically. To regenerate only the flowchart after creating the output directory:

```sh
mkdir -p figures
python code/flowchart.py
```

## Result files

- `results/table_main_comparison.csv` and `main_comparison_per_fold.csv`: policy comparison.
- `results/table_ablation.csv` and `ablation_per_fold.csv`: component and evidence ablations.
- `results/table_significance.csv`: paired statistical comparisons.
- Sensitivity, calibration, confusion-matrix, feature-importance, and dataset-summary CSVs.
- `results/metrics.json`: structured summary.
- `results/fig*.jpg` and generated `figures/*.png`: figures.

See [`code/README.md`](code/README.md) for method and output details. Supplied runtime estimates depend on hardware and installed dependency versions.

## Scope and limitations

The results depend on the synthetic generator and assumed minutes-equivalent incident penalties. They do not establish production performance. Cost-aware logistic regression performs comparably to CS-CART, and the Bayes-risk decision rule explains most of the improvement. The code evaluates a single recovery decision per call. The dataset documentation names an upstream `gen_author5.py` generator, but that generator was not supplied in this archive; the included CSV is sufficient to run the experiments.

This release package contains code, data, and experiment outputs. Manuscripts, presentation decks, operating-system metadata, virtual environments, and transient run logs are excluded.

## Author and license

Author: Akhil Venkata Sai Madineni. This package is distributed under the [MIT license](LICENSE).

## Citation and archive

Source repository: [https://github.com/Akhilmadineni/outcome-aware-recovery](https://github.com/Akhilmadineni/outcome-aware-recovery).

Use [`CITATION.cff`](CITATION.cff) for the software citation. The matching [`.zenodo.json`](.zenodo.json) supplies the archive metadata for release `v1.0.0`. The version DOI will be added after Zenodo completes archiving.
