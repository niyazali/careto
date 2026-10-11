# Quantitative-feature analysis

The expanded subset contains 81 existing observations from 10 source groups. Initial concentrations and incubation times were matched to the original outcomes without changing response values. The original 40 observations remain included. Treatment and mass-basis qualifications are retained in extraction_evidence.csv. Predictions and split logs are saved for both specified comparisons.

Run with the documented analysis environment:

    python run_benchmark.py --input subset.csv --features log10_initial_carotenoid_ug_g,gastric_duration_min,intestinal_duration_min --output fresh_primary
    python run_benchmark.py --input subset.csv --features log10_initial_carotenoid_ug_g --output fresh_loading_only

Source files in the extraction ledger are archived acquisition paths. Full texts are not redistributed. The source URLs and table/method anchors permit verification. The audit was executed in the source workspace; audit_expanded.py and extend_existing_subset.py retain their acquisition-directory dependencies and are not standalone release commands. The two benchmark commands above use only subset.csv and installed dependencies.
