# Release notes

## 11 October 2026 revision

Confirmed stereochemical synonyms and spelling variants now share conservative model labels. Original source labels, response values, endpoint definitions, cohort membership, non-analyte predictors, validation folds, candidate grids and random seeds are preserved. The core has 38 source labels and 33 model labels; 20 model-label entries changed among its 450 observations.

All four main endpoint cohorts and the post hoc weighting, matched-size, deletion and learning-curve analyses were refitted. Both quantitative-feature comparisons were rerun and retain identical predictions and metrics. The primary conclusion remains unchanged: no learner established a clear advantage over the median baseline under category holdout with contributing-study exclusion.

Source-to-model mappings, primary-source locators, retrieval-limit reporting and the author-confirmed search-date correction are provided. The correction record retains original API execution timestamps. These records do not establish dual-human screening or extraction.

The independent main audit recomputed 144 MAEs and 128 paired intervals, checked 112 outer and 2352 inner partitions, and replayed 1008 fitted models in the source workspace. Saved-model replay requires the single-thread environment used for fitting; tied nearest-neighbor choices differ under multiple threads. Fitted-model binaries and copyrighted source texts are excluded from this public release.

Post hoc results were independently checked from their saved predictions and losses; the audit receipt identifies its scope and counts. Conditional intervals exclude refitting uncertainty and remain unadjusted. This release adds no new experiments or clinical validation.

The earlier 10 October expansion added 36 quinoa observations and quantitative covariates for existing observations; SOURCE_EXPANSION.md retains those extraction decisions.
