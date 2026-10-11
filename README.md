# Carotenoid bioaccessibility benchmark

Code and data accompanying the study of in vitro carotenoid bioaccessibility across foods.

The main dataset has 450 observations from 64 original-study groups across nine food categories. Bioaccessibility is expressed as the carotenoid amount in the isolated micellar fraction divided by the initial amount in the food. Values estimated from graphs are marked in the data.

The analysis compares seven regression models with mean and median baselines. Additional analyses examine endpoint definitions, graph estimates, influential studies, training weights, training-set size and whole-study learning curves. A smaller comparison adds quantitative predictors for 81 existing observations from 10 studies, supported by 243 feature evidence records.

The 10 October 2026 update adds 36 quinoa observations from one independent study and quantitative covariates for 41 existing observations from three studies. Earlier outcome records are preserved. See [SOURCE_EXPANSION.md](SOURCE_EXPANSION.md) for extraction decisions and [RELEASE_NOTES.md](RELEASE_NOTES.md) for the release checks.


The 11 October 2026 revision assigns common model labels to confirmed analyte synonyms, while retaining source_analyte and all original response values. Unspecified isomers and distinct carotenoid pools remain separate. See ANALYTE_CANONICALIZATION.json, analyte_model_labels.csv and PROVENANCE_REPORTING.md. All affected main and post hoc models were refitted with unchanged cohorts, folds, candidate grids and seeds.

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

`analysis_config.json` contains the features, models, tuning grids, validation designs and random seeds. Preprocessing and tuning use only training data. Study holdouts and category holdouts with study exclusion remove test studies from training. The driver sets OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1 before NumPy import. Retain single-thread numerical execution when replaying nearest-neighbor models because equal-distance ties may be resolved differently with multiple threads. Full runs can take several hours. Use a new output directory after changing inputs or code.

Summaries and held-out predictions are supplied in `saved_results_core/` (450 observations/64 studies), `saved_results_numeric/` (382/58), `saved_results_supernatant/` (580/88) and `saved_results_converted/` (585/92). The supplementary analyses below use the primary predictions. To replace them after rerunning the primary analysis, copy `results/core/RESULTS.json`, `results/core/RESULT_SUMMARY.json` and `results/core/oof_predictions.intermediate.csv` to `saved_results_core/RESULTS.json`, `saved_results_core/RESULT_SUMMARY.json` and `saved_results_core/heldout_predictions.csv`, respectively.

## Quantitative-feature comparison

```sh
cd quantitative_feature_benchmark
python run_benchmark.py --input subset.csv --features log10_initial_carotenoid_ug_g,gastric_duration_min,intestinal_duration_min --output results/primary
python run_benchmark.py --input subset.csv --features log10_initial_carotenoid_ug_g --output results/loading_only
python test_benchmark.py
cd ..
```

The comparison uses nested leave-one-study-out validation on the same observations. `extraction_evidence.csv` records the source, table or figure location, units and extraction notes for each predictor. This analysis was developed after the main analysis and was not preregistered.

Completed results for both feature sets are supplied in `quantitative_feature_benchmark/results_primary/` and `quantitative_feature_benchmark/results_loading_only/`.

## Supplementary analyses and figures

```sh
python robustness_analysis.py
python training_weight_sensitivity.py
python training_size_sensitivity.py
python run_learning_curves.py
python draw_endpoint_schematic.py
python plot_validation_diagnostics.py
python plot_heldout_predictions.py
```

These commands generate result directories and figures locally. Completed supplementary results are supplied separately in `saved_supplementary_results/`. Training-size analysis caches completed fits; remove its generated cache to repeat every fit. Whole-study learning curves use target-independent nested subsets of training studies; their repeat variation is not a confidence interval.

## Data

- `core_dataset.csv` and `broader_dataset.csv`: observations, cohort membership and source identifiers.
- `COHORT.json` and `COHORT_with_conversions.json`: model inputs, including the endpoint-conversion sensitivity dataset.
- `source_bibliography.csv`: references for the primary study groups.
- `source_provenance.json` and `new_quinoa_extraction.json`: observation-level source evidence and the added study's extraction records.
- `screened_sources/`: reported pasta means and the endpoint review; these observations were excluded from all fitted cohorts.
- Root-level result tables: validation, diagnostics, study errors and sensitivity analyses used in the supplementary workbook.
- `conversions.csv`: endpoint-conversion assumptions.
- `FREEZE*.json`: input hashes checked by the analysis runner.
- `SHA256SUMS`: checksums of the distributed files.

## Interpretation

The primary analysis did not demonstrate a clear model advantage over the median baseline. Observations from the same study are correlated. Reported intervals resample study errors from saved predictions; they do not include model refitting, tuning uncertainty or dependence caused by overlapping training sets. They describe uncertainty conditional on these fitted models. The study has no prospective external validation.

Neither quantitative-feature comparison established a clear incremental advantage. Whole-study learning curves showed no consistent decrease in error with additional training studies under the current categorical predictors.

Original articles and downloaded full texts are not included.
