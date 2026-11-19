# Model card: Exoplanet Finder CNN

## Model details
* **Type**: dual-branch 1D convolutional neural network (global view and local view), with 1,557,633 parameters.
* **Output**: one score in [0, 1], the sigmoid of a logit. **It is not calibrated** and must not be read as the
  probability that a planet exists.
* **Decision rule**: "resembles DR24 PC" when score ≥ the stored threshold. The threshold is chosen on the validation
  split by maximising F1.
* **Files**: `models/release/cnn.pt` holds the weights, architecture configuration, preprocessing configuration,
  version and fingerprint, label mapping, threshold, seed, dataset hashes and package versions.
  `models/release/release.json` is a summary card.
* **Author**: Muhammad Abu Bakr Siddique (student portfolio project), 2026. Licence: Apache-2.0.
* **Inspiration**: Shallue & Vanderburg (2018), AstroNet. This is an independent reimplementation; see
  README "Differences from AstroNet".

## Intended use
* Education and portfolio demonstration of an end-to-end, leakage-aware scientific ML pipeline.
* Ranking **already-identified** Kepler candidate signals (with a known period, epoch and duration) by how
  closely they resemble DR24 planet-candidate dispositions, for example to prioritise human inspection.

## Out of scope
* Declaring or confirming planets. Confirmation requires further evidence, such as pixel-level centroid analysis,
  contamination checks, follow-up spectroscopy or imaging, and statistical validation.
* Blind searches: the model does not find periods.
* Data other than Kepler long-cadence `PDCSAP_FLUX` (TESS, K2, short cadence, ground-based), unless revalidated.
* Single-transit events and candidates with wrong ephemerides.

## Training data
* **Population**: Kepler Q1–Q17 DR24 TCEs with an `av_training_set` label (PC 3,600; AFP 9,596; NTP 2,541;
  15,737 TCEs on 9,865 stars). UNK (4,630) is excluded. Light curves were downloaded for all 9,865 stars;
  after preprocessing, 15,733 TCEs on 9,861 stars were usable.
* **Label meaning**: PC → 1 (DR24 *planet candidate*, not confirmed planet); AFP (astrophysical false positive,
  mostly eclipsing binaries, blends and contamination) and NTP (non-transiting phenomenon: instrumental or stellar
  artefacts) → 0. The labels came from manual vetting and other diagnostics (Catanzarite 2015) and contain known
  errors. Shallue & Vanderburg found PCs that the later DR25 KOI table lists as eclipsing binaries.
* **Inputs**: DR25 light curves (`DATA_REL 25`) currently served by MAST, paired with DR24 ephemerides and labels.
  This release mismatch is deliberate and documented.
* **Split**: grouped by host star, 70/15/15, seed 20171215, stratified by whether the star hosts a PC. Kepler-90 is
  held out for the demo.
* **Exclusions**: examples that fail preprocessing (insufficient coverage, no dip, too few points, download failure)
  are listed with reasons in `reports/full/dataset/examples.csv.gz` (committed) and summarised in the results.

## Evaluation
See [`reports/full/RESULTS.md`](reports/full/RESULTS.md) and the Results section of the README.

<!-- MODEL-CARD-RESULTS:START -->
* **Released model**: `cnn_global_local_seed2`, chosen by validation AP (0.929) among 3 global+local seeds.
  Decision threshold 0.671.
* **Test split**: 2,362 DR24 TCEs on 1,481 stars that were never used for training or selection. 545 are PC
  (prevalence 0.231).
* **Test metrics**: AP **0.913** (95% star-bootstrap interval 0.886–0.939), ROC-AUC 0.977, precision 0.830,
  recall 0.912, F1 0.869 and accuracy 0.936 (TP 497, FP 102, TN 1,715, FN 48).
* **False-positive rate by original category**: AFP 6.9% (100/1,451), NTP 0.5% (2/366). Most errors on negatives
  are astrophysical false positives whose light curves look planet-like.
* **Comparison**: the gradient-boosting feature baseline has AP 0.873; the CNN's advantage is ΔAP +0.041
  [+0.013, +0.070]. Across 3 seeds, global+local AP is 0.897 ± 0.024 and global-only AP is 0.882 ± 0.004.
* **Weakest subgroups**: measured SNR < 18.8 (AP 0.850, recall 0.820) and duration < 2.42 h (AP 0.868, recall
  0.829).
* **Usable data**: 15,733 of 15,737 labelled TCEs, on 9,861 of the 9,865 downloaded stars (4 training NTPs
  rejected by preprocessing). 0 final download failures.
* Details: [docs/RESULTS_REPORT.md](docs/RESULTS_REPORT.md).
<!-- MODEL-CARD-RESULTS:END -->

## Known failure modes and caveats
* **Label noise**: some DR24 dispositions are wrong, so test metrics measure agreement with DR24, not truth.
* **Shallow or low-SNR signals**: when the transit is near the noise level, the local-view normalisation divides by
  a depth that is partly noise, so the views become noisy.
* **Systems with several planets**: transits of other planets are not removed and can appear in the views.
  The median binning suppresses them only partly.
* **Ephemeris errors** smear the folded transit. DR24 ephemerides may drift relative to DR25 light curves for long
  baselines.
* **Long periods** with 1–2 observed transits give sparse global views.
* **Training instability**: one of three seeds (seed 1) plateaued early (test AP 0.869 vs 0.908 and 0.913), so a
  single retrained model may perform noticeably differently.
* **Light-curve-only blind spot**: AFPs identified from centroid shifts or contaminating neighbours can look like
  clean transits and receive high scores. Later-catalogue context (DR25 KOI flags, matched by star, period and
  epoch): one of the two highest-scoring test AFPs carries the DR25 centroid-offset flag. The other is a DR25
  candidate, now the confirmed planet Kepler-529 d, so that error is a DR24/DR25 label disagreement. See
  `reports/full/dr25_koi_check/RESULTS.md`.
* **Score is uncalibrated** (test Brier 0.052, for reference only). Scores near the threshold should be treated as uncertain.

## Distribution limits
The model has only seen phase-folded Kepler long-cadence PDCSAP light curves of DR24 TCEs, normalised as described
in the README. Usable training-population periods span 0.50–726.6 days and durations 0.50–57.7 hours. Inputs
outside these ranges, or from other instruments, are out of distribution even if the app accepts them.

## Ethical and scientific responsibility
Public communication about this model must not describe its outputs as discoveries. Kepler-90 i is a planet
validated by Shallue & Vanderburg (2018), with a false-positive probability of about 0.005 from `vespa`. It is
used here only as a historical demonstration.
