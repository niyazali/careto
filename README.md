# Carotenoid bioaccessibility benchmark

Data and code for predicting in vitro carotenoid bioaccessibility across food matrices and original studies.

The primary dataset contains 414 observations from 63 study groups in nine food-matrix categories. The target is isolated-micellar bioaccessibility relative to carotenoid content in the initial food. Graph estimates are flagged. Sensitivity analyses assess graph exclusion, broader aqueous fractions, endpoint conversions, influential studies, training weights and training size. A quantitative-feature comparison uses 40 observations from seven studies.

## Installation

Use Python 3.12 or newer.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Main analysis

```sh
python run_validation.py --output results/core
python run_validation.py --sensitivity numeric_text_table_only --output results/numeric
python run_validation.py --sensitivity core_plus_supernatant --output results/aqueous
python run_validation.py --cohort COHORT_with_conversions.json --sensitivity core_plus_supernatant --output results/converted
```

`analysis_config.json` contains the features, models, tuning grids, validation designs and random seeds. Preprocessing and tuning use only training data. Study-separated and category-purged validation exclude test studies from training. Full runs can take substantial CPU time. Use a new output directory after changing inputs or code.

The primary summary and held-out predictions are supplied in `saved_results_core/`. The supplementary analyses below use these predictions. To replace them after rerunning the primary analysis, copy `results/core/RESULT_SUMMARY.json` and `results/core/oof_predictions.intermediate.csv` to `saved_results_core/RESULT_SUMMARY.json` and `saved_results_core/heldout_predictions.csv`, respectively.

## Quantitative-feature comparison

```sh
cd quantitative_feature_benchmark
python run_benchmark.py --input subset.csv --features log10_initial_carotenoid_ug_g,gastric_duration_min,intestinal_duration_min --output results/primary
python run_benchmark.py --input subset.csv --features log10_initial_carotenoid_ug_g --output results/loading_only
python test_benchmark.py
cd ..
```

The comparison uses nested leave-one-study-out validation on the same observations. `extraction_evidence.csv` provides source links, locators, units and qualifications. This analysis was developed after the main analysis and was not preregistered.

## Supplementary analyses and figures

```sh
python robustness_analysis.py
python training_weight_sensitivity.py
python training_size_sensitivity.py
python draw_endpoint_schematic.py
python plot_validation_diagnostics.py
python plot_heldout_predictions.py
```

These commands generate result directories and figures locally. Training-size analysis caches completed fits; remove its generated cache to repeat every fit.

## Data

- `core_dataset.csv` and `broader_dataset.csv`: observations, cohort membership and source identifiers.
- `COHORT.json` and `COHORT_with_conversions.json`: model inputs, including the endpoint-conversion sensitivity dataset.
- `source_bibliography.csv`: references for the primary study groups.
- `conversions.csv`: endpoint-conversion assumptions.
- `FREEZE*.json`: input hashes checked by the analysis runner.
- `SHA256SUMS`: checksums of the distributed files.

Observations within studies are correlated. Uncertainty intervals resample study losses from fixed held-out predictions and exclude refitting uncertainty. No model demonstrated a clear primary advantage over the median baseline. No prospective external validation was performed.

Original articles and downloaded full texts are not included.
