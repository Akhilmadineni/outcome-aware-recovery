# A5P1 code - outcome-aware recovery under ambiguous API responses

Run (CPU, single thread, ~4-8 min on a shared 2-core machine):

    cd code && OMP_NUM_THREADS=1 python3 run_all.py

Reads `../data/A5P1_ambiguous_api_calls.csv` (synthetic dataset, see `../data/A5P1_README.md`).
`call_id`, `deployment_id`, `true_outcome` and `correct_recovery_action` are never used as features.
All seeds are fixed (2026).

## Files
- `run_all.py` - full pipeline: preprocessing (winsorising, median imputation + missing flags, ratio features,
  one-hot), recovery cost model (6 actions, Eq. 1), proposed cost-sensitive CART (cost-proportionate weights
  w = r^gamma, weighted-Gini tree, Laplace leaf posterior, Bayes-risk action incl. abort), inner 3-fold
  selection of (depth, min leaf, gamma), baselines (retry-always, fail-fast, always-reconcile, logistic
  regression, LR + Bayes-risk rule), 5-fold stratified CV, ablations, sensitivity, significance, ECE.
- `flowchart.py` - Graphviz source of Fig. 1 (called by run_all.py).

## Outputs (`../results`, figures as PNG 300 dpi in `../figures` and JPG in `../results`)
- `table_main_comparison.csv`, `main_comparison_per_fold.csv` - policy comparison (mean/std over 5 folds)
- `table_ablation.csv`, `ablation_per_fold.csv` - component and evidence-group ablations
- `table_significance.csv` - paired t-test (folds), Wilcoxon (calls), bootstrap 95 % CI of cost difference
- `sensitivity_depth_leaf.csv` - inner-CV cost per (depth, min leaf) (best gamma)
- `table_sensitivity_penalty.csv`, `sensitivity_penalty_per_fold.csv` - scaling of incident penalties
- `congestion_quartile_per_fold.csv` - cost and ECE by congestion quartile
- `confusion_matrix_proposed.csv`, `feature_importance.csv`, `selected_hyperparameters.csv`,
  `table_dataset_summary.csv`, `metrics.json`
- Fig. 1 flowchart, Fig. 2 policy comparison, Fig. 3 confusion matrix, Fig. 4 depth/leaf sensitivity,
  Fig. 5 feature importance, Fig. 6 cost by congestion quartile
