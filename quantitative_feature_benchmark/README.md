# Quantitative-feature analysis

The quantitative subset contains 81 observations from 10 source groups. Initial concentrations and incubation times were matched to the original outcomes without changing response values. Treatment and mass-basis qualifications are retained in extraction_evidence.csv. Predictions and split logs are saved for both specified comparisons.

Run with the documented analysis environment:

    python run_benchmark.py --input subset.csv --features log10_initial_carotenoid_ug_g,gastric_duration_min,intestinal_duration_min --output fresh_primary
    python run_benchmark.py --input subset.csv --features log10_initial_carotenoid_ug_g --output fresh_loading_only

Source files in the extraction ledger are archived acquisition paths. Full texts are not redistributed. The source URLs and table/method anchors permit verification. Archived acquisition paths identify source-verification files and are not required for benchmark execution. The two benchmark commands above use only subset.csv and installed dependencies.
