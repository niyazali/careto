# 10 October 2026 update

The primary cohort expands from 414 observations in 63 study groups to 450 observations in 64 study groups. The 36 added quinoa observations come from DOI 10.1021/acs.jafc.6c01729. The quantitative subset expands from 40 observations in seven studies to 81 observations in 10 studies already represented in the main cohort. Original outcome records are preserved.

All four cohort analyses, quantitative comparisons, deletion and weighting sensitivities, matched training-size analysis and whole-study learning curves are complete. Saved summaries, predictions and source evidence accompany the inputs. Full-text articles, fitted model binaries and working caches are excluded.

The scientific functions and classes in the existing portable analysis scripts are identical to those used for the completed expanded analysis. Differences in script hashes reflect formatting, documentation and repository-relative output paths. `FREEZE*.json` and saved quantitative configurations retain the historical fitting-script hashes; `RELEASE_CHECKS.json` records the published script hashes and the comparison. The learning-curve command adds an output-directory option and a main guard to the original calculation.

Release checks confirmed:

- All 144 cohort/scheme/model summaries have complete, unique predictions and exactly matching recomputed study-balanced MAEs.
- The quantitative subset has 81 observations, 10 study groups and 243 unique feature evidence records; its outcomes match the primary cohort.
- All 13,500 learning-curve predictions cover the expected observations; training and test studies are disjoint and repeat MAEs agree with the saved summary.
- The quantitative benchmark's synthetic tests pass deterministic reruns, study separation, training-only imputation, weighted-median agreement and rejection of insufficient study groups.
- Python files compile, frozen cohort/configuration hashes match, and Figure 1 renders with the legend fully visible.

`ANALYSIS_AUDIT.json` records the prior independent replay of 1,008 fitted models, with zero prediction difference. Release verification checks the distributed artifacts and does not repeat that full fitting audit. `SHA256SUMS` covers every distributed file except itself.

The expanded analyses do not establish a clear primary learner advantage, a clear incremental quantitative-feature advantage or consistent improvement with additional training studies under the current predictors. The manuscript and submission documents remain in the local submission package.
