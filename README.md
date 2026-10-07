# Carotenoid bioaccessibility benchmark

Data and Python code for evaluating carotenoid bioaccessibility predictions across food matrices and original studies.

The main dataset contains 414 observations from 63 study groups in nine food-matrix categories. The target is isolated-micellar bioaccessibility expressed relative to carotenoid content in the initial food. Graph-derived values are identified in the data. Three sensitivity datasets test the inclusion of graph estimates, broader aqueous fractions and approximate endpoint conversions. A separate analysis compares categorical and quantitative predictors on 40 observations from seven studies.

## Installation

Use Python 3.12 or newer. From this directory:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Main analysis

Each command writes to a separate directory. Existing result files support comparisons with a fresh run.

```sh
python run_validation.py --output results/core
python run_validation.py --sensitivity numeric_text_table_only --output results/numeric
python run_validation.py --sensitivity core_plus_supernatant --output results/aqueous
python run_validation.py --cohort COHORT_with_conversions.json --sensitivity core_plus_supernatant --output results/converted
```

`analysis_config.json` specifies features, model settings, validation designs and seeds. Preprocessing and hyperparameter selection use only the training data within each split. The study-separated and category-purged designs exclude the test studies from training. Full runs fit several hundred models and can take substantial time on a CPU. A run resumes from a matching checkpoint; changed inputs or code require a new output directory.

Saved summaries and held-out predictions are in `saved_results_core`, `saved_results_numeric`, `saved_results_supernatant` and `saved_results_converted`. The release omits fitted-model binaries and intermediate checkpoints; fresh runs generate them.

## Quantitative predictors

```sh
cd quantitative_feature_benchmark
python run_benchmark.py --input subset.csv --features log10_initial_carotenoid_ug_g,gastric_duration_min,intestinal_duration_min --output results/primary
python run_benchmark.py --input subset.csv --features log10_initial_carotenoid_ug_g --output results/loading_only
python test_benchmark.py
cd ..
```

This analysis uses nested leave-one-study-out evaluation and the same observations for each feature comparison. `extraction_evidence.csv` records source locators and unit qualifications. The quantitative analysis was developed after the main analysis; it was not preregistered.

## Sensitivity analyses and figures

```sh
python robustness_analysis.py
python training_weight_sensitivity.py
python training_size_sensitivity.py
python draw_endpoint_schematic.py
python plot_validation_diagnostics.py
python plot_heldout_predictions.py
```

The sensitivity scripts write to their respective result directories. Preserve the supplied CSV files before rerunning them. Training-size analysis caches completed fits; remove its generated cache to repeat every fit. Figure scripts use the supplied predictions and write to `figures/`.

## Data files

- `COHORT.json`: model inputs for the main and broader aqueous cohorts.
- `COHORT_with_conversions.json`: inputs including the five approximate conversions.
- `core_dataset.csv`, `broader_dataset.csv` and `observations.csv`: tabular observations and cohort membership.
- `source_bibliography.csv`: publication identifiers for the retained main-study groups.
- `conversions.csv`: assumptions used in endpoint conversions.
- `FEATURE_ALIASES.json`: mappings used to harmonize categorical descriptors.
- `SHA256SUMS`: checksums of the release files.

Observations from one study are correlated. The number of rows is not the number of independent experiments. Reported uncertainty intervals resample held-out study losses and exclude uncertainty from refitting. No model demonstrated a clear primary advantage over the median baseline, and no prospective external validation was performed.

Original articles and downloaded full texts are not distributed in this repository. Code formatting and release paths have been cleaned; the analysis rows and model calculations are unchanged.
