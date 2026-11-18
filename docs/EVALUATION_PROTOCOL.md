# Evaluation protocol (fixed before any test-set result was computed)

This file was committed before the full experiment was trained or evaluated. Its git commit timestamp
predates `reports/full/`. Nothing below was changed after test results were seen.

## Data
* Examples: Kepler Q1–Q17 DR24 TCEs with `av_training_set` ∈ {PC, AFP, NTP}. PC → 1; AFP and NTP → 0; UNK excluded.
* Split: `data/splits/split_manifest.csv` (seed 20171215), grouped by host star at 70/15/15. Kepler-90
  (KIC 11442793) forms a separate `demo_holdout` split.
* Examples rejected by preprocessing are excluded with recorded reasons. They are counted in the dataset manifest and
  never imputed.

## Planned runs (`configs/full.yaml`)
1. Baseline `logreg`: logistic regression on 13 light-curve features.
2. Baseline `hgb`: histogram gradient boosting on the same features.
3. CNN global+local, seeds 0, 1, 2.
4. CNN global-only ablation, seeds 0, 1, 2 (pending if compute does not permit).

The hyperparameters are those in `configs/full.yaml`. There is no hyperparameter search.

## Selection, using validation data only
* CNN checkpoint: the epoch with the highest validation average precision (early stopping, patience 8, ≤ 40 epochs).
* Decision threshold for every model: maximise validation F1; predict PC when score ≥ threshold.
* Release model for the app: the global+local CNN run with the highest validation AP.

## Test-set reporting (single evaluation)
* For every model on identical test examples: class counts, prevalence, AP (scikit-learn definition), ROC-AUC,
  precision, recall, F1 and confusion matrix at the validation threshold, plus accuracy (secondary).
* Mean and standard deviation over CNN seeds, reported for all completed runs.
* Star-level bootstrap (1000 resamples) 95% intervals for test AP of the release CNN and both baselines, and for the
  AP difference between the release CNN and HGB.
* Subgroups: terciles of measured depth, duration and measured SNR, with edges taken from the training split; plus
  catalogue disposition (PC / AFP / NTP). AP is shown only when a subgroup has ≥ 30 examples and ≥ 5 per class.
* Error analysis: examples chosen by the fixed rules in `evaluate._select_examples`, including false positives and
  false negatives.
* Demo hold-out (Kepler-90 h) and Kepler-90 i are reported separately and are not part of the benchmark.

If a result is disappointing, it is reported as measured. Any later change to the protocol will be documented as a
new, separately labelled experiment.
