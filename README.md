# Exoplanet Finder

**A reproducible, AstroNet-inspired 1D CNN that scores candidate transit signals in NASA Kepler light curves.**

Give it a Kepler light curve and a candidate's **period, epoch and duration**. It detrends and phase-folds the light
curve, builds fixed-length *global* and *local* views, and returns a **model score** for how closely the signal
resembles transits labelled *planet candidate* (PC) in the Kepler DR24 Autovetter training set, as opposed to
*astrophysical false positives* (AFP) or *non-transiting phenomena* (NTP).

> **A score is not a discovery.** The model reproduces catalogue dispositions from the light curve alone.
> Confirming an exoplanet needs further evidence: centroid and contamination analysis, follow-up observations
> or statistical validation. The score is not a calibrated probability.

Inspired by Christopher Shallue and Andrew Vanderburg's 2018 work, which found **Kepler-90 i** with a
CNN ([arXiv:1712.05044](https://arxiv.org/abs/1712.05044),
[google-research/exoplanet-ml](https://github.com/google-research/exoplanet-ml)). This is an independent, smaller
reimplementation with a stricter evaluation protocol. See [Differences from AstroNet](#differences-from-astronet).

<!-- RESULTS:START -->
## Results

*Pending: the full-data experiment is running. This section is filled from `reports/full/metrics.json` after
evaluation.*
<!-- RESULTS:END -->

---

## Quick start

Requirements: Python ≥ 3.11 (developed on 3.14.7, macOS / Apple M5). CPU works; Apple MPS or CUDA are used
automatically when available. No paid services or API keys.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # exact versions used: requirements-lock.txt

# Demo with the committed release model and bundled real Kepler light curves (offline):
streamlit run app/streamlit_app.py

# Verification
ruff check src tests app scripts
pytest -q
```

### Reproduce from scratch

The catalogue snapshot (`data/catalog/`) and the fixed split (`data/splits/`) are committed, so the steps below
reproduce the same examples. `fetch-catalog` and `make-splits` refuse to overwrite them unless you pass `--force`.

| step | quick-start (50 stars, ~3 min) | full experiment (9,865 stars) |
|---|---|---|
| catalogue snapshot | `exofinder fetch-catalog --config configs/quickstart.yaml` | same, `configs/full.yaml` |
| star-grouped split | `exofinder make-splits --config configs/quickstart.yaml` | same |
| download + preprocess (resumable) | `exofinder fetch --config configs/quickstart.yaml` | `exofinder fetch --config configs/full.yaml` |
| dataset + manifest | `exofinder build-dataset --config …` | |
| baselines | `exofinder train-baseline --config …` | |
| CNN (3 seeds in full) | `exofinder train-cnn --config …` | |
| global-only ablation | `exofinder train-cnn --config … --global-only` | |
| evaluation + report | `exofinder evaluate --config …` | writes `reports/<experiment>/` |
| release + demo bundle | `exofinder release …` then `exofinder bundle-examples …` | |

The quick-start configuration exercises the whole pipeline on real data, but its 67 examples are far too few for
meaningful metrics.

**Measured cost of the full download** (this machine, 12 worker processes): ≈ 2,300–2,500 stars/hour,
≈ 7.0 MB of FITS per star (≈ 69 GB transferred in total). Raw files are deleted once each star's views are saved and
verified, so the peak raw cache stayed below ~130 MB. Processed views total ≈ 100–200 MB. Interrupted runs resume
from per-star ledgers in `data/processed/ledger/`.

---

## Data and labels

| item | choice |
|---|---|
| Catalogue | Kepler Q1–Q17 **DR24** TCE table `q1_q17_dr24_tce`, NASA Exoplanet Archive TAP service |
| Snapshot | `data/catalog/dr24_tce_snapshot.csv` + `.provenance.json` (query, URL, UTC retrieval time, SHA-256) |
| Label column | `av_training_set` (Autovetter training set, as used by AstroNet) |
| Label mapping | **PC → 1**; **AFP, NTP → 0**; **UNK excluded** (never promoted). The original category is kept per example. |
| Example unit | one TCE (candidate event); ID `KKKKKKKKK_NN` = zero-padded KIC ID + TCE number |
| Light curves | MAST Kepler archive `archive.stsci.edu/missions/kepler/lightcurves/`, long cadence `*_llc.fits` |
| Flux | `PDCSAP_FLUX`; cadences with `SAP_QUALITY & 1130799 ≠ 0` (Lightkurve default bitmask) removed |
| Time | BKJD = BJD_TDB − 2454833 (FITS `TIME`, checked via `TIMESYS = TDB`, `BJDREFI = 2454833`) |
| Units | period: days · epoch: BKJD days · duration: **hours** in the catalogue, converted to days internally |

All 20,367 DR24 TCEs: PC 3,600 (2,716 stars), AFP 9,596 (5,286), NTP 2,541 (2,260), UNK 4,630. Labelled
examples: **15,737 TCEs on 9,865 stars** (positive prevalence 22.9%).

**What the labels mean.** "PC" means a DR24 planet candidate, *not* a confirmed planet. AFP lumps together
eclipsing binaries, background blends and contamination. NTP covers instrumental and stellar artefacts. Labels can be
wrong, and later catalogues (DR25) disagree for some TCEs. The task is therefore described as **reproducing DR24
training-set dispositions**. No other catalogue release is mixed into the labels.

**Release mismatch.** MAST now serves the final Kepler light curves (`DATA_REL = 25`, SOC pipeline 9.3), while the
DR24 TCEs and labels came from earlier processing. Ephemerides still line up for almost all TCEs, but a TCE's signal
in DR25 light curves can differ from what vetters saw. Every example records its light-curve files and data release.

**Splits** (`data/splits/split_manifest.csv`, seed 20171215): grouped by host star and stratified by whether the star
hosts at least one PC. 70% / 15% / 15% of stars go to train / validation / test. All TCEs of a star share a split,
and so do its views and augmented copies. **Kepler-90 (KIC 11442793) is a separate `demo_holdout`**: it is excluded
from training, validation and the main benchmark and is reported separately.

| split | stars | TCEs | PC | AFP+NTP |
|---|---|---|---|---|
| train | 6,904 | 11,014 | 2,518 | 8,496 |
| val | 1,479 | 2,360 | 536 | 1,824 |
| test | 1,481 | 2,362 | 545 | 1,817 |
| demo_holdout | 1 | 1 | 1 | 0 |

*(Catalogue-level counts. Usable counts after download and preprocessing exclusions are in the dataset manifest and the
results section.)*

## Preprocessing (shared by training and the demo)

One implementation, `src/exofinder/preprocess.py`, is used for the training set, the bundled examples and uploaded
CSVs. Its version and configuration fingerprint are stored in every checkpoint and checked when it is loaded.
`tests/test_real_inference.py` checks that the upload path and the training path produce bit-identical views
for real data.

1. **Clean**: finite values, sort, drop duplicate timestamps, median-normalise each quarter.
2. **Segment** at quarter boundaries and gaps > 0.75 d.
3. **Detrend**: iteratively 3σ-clipped least-squares cubic spline (knots every 1.5 d) per segment, fitted with the
   candidate's transits **masked** (±1.5 durations). Flux ÷ trend.
4. **Remove only upward outliers** (> 5σ robust); downward points are never clipped.
5. **Phase-fold**: `φ = ((t − t₀ + P/2) mod P) − P/2`.
6. **Views** (median binning, AstroNet geometry): global 2001 bins of width P/2001 over the full orbit; local 201 bins
   of width 0.16·D over ±4·D. Empty bins are linearly interpolated and flagged in a **coverage-mask channel**.
7. **Normalise**: `s = median(local) − min(local)`; both views → `(view − median)/s` (transit at −1; a shared scale
   keeps secondary eclipse depth relative to the primary); clipped to ±10.
8. **Reject**, with a reason code, inputs with < 100 points, < 3 in-transit points, local coverage < 50%, global
   coverage < 30%, depth `s` < 1 ppm, or invalid parameters.

The [walkthrough](docs/WALKTHROUGH.md) explains each step for beginners.

## Models

**Baselines** (`baseline.py`): 13 interpretable features computed by the shared preprocessing (log period, duration,
duty cycle, depth, noise, SNR, number of transits, odd/even depth difference, secondary-eclipse ratio, transit shape,
coverage). Two models: logistic regression (imputation + standardisation fitted on training data, balanced class
weights) and histogram gradient boosting (balanced class weights).

**CNN** (`model.py`): dual-branch 1D CNN, 1,557,633 parameters.

| part | layers | output |
|---|---|---|
| global branch, input 2 × 2001 | 5 × [Conv1d k=5 → ReLU → Conv1d k=5 → ReLU → MaxPool 5/2], channels 16-32-64-64-64 | 64 × 59 = 3,776 |
| local branch, input 2 × 201 | 2 blocks, channels 16-32 | 32 × 48 = 1,536 |
| head | concat 5,312 → FC 256 → ReLU → Dropout 0.3 → FC 256 → ReLU → Dropout 0.3 → FC 1 | logit |

The global-only ablation drops the local branch (1,155,201 parameters).

**Training** (`train.py`): `BCEWithLogitsLoss` with `pos_weight = n_neg/n_pos` from the training split; AdamW (lr 1e-3,
weight decay 1e-4); batch size 128; up to 40 epochs; early stopping (patience 8) on **validation average precision**.
Time-reversal augmentation (both views and masks flipped together, keeping the transit centred) is applied only to
training data. Checkpoints are written atomically and runs resume from `last.pt`. Seeds, configuration, dataset hashes and
package versions are stored in each checkpoint. The decision threshold is chosen on validation data (max F1) and saved
with the model. Three seeds are trained on the same fixed split.

## Evaluation protocol

* Every model is scored on the **same** test examples. Test results were not used for any choice.
* Primary metric: **average precision** (AP = Σₙ (Rₙ − Rₙ₋₁)·Pₙ, scikit-learn), with the PR curve. Also reported:
  ROC-AUC, precision / recall / F1 and confusion matrix at the validation threshold, and accuracy as a secondary metric.
  Uncertainty comes from a star-level bootstrap of the test split.
* Release model = the global+local CNN run with the highest **validation** AP.
* Per-example predictions for all splits and models are saved in `reports/full/predictions.csv`, so every number can
  be recomputed.

## Demo app

```bash
streamlit run app/streamlit_app.py
```

* **Score a candidate**: choose a bundled real Kepler TCE (selected by fixed rules after the model was frozen, and
  including failures) or Kepler-90 h / i, or upload a CSV (`time`, `flux`, optional `quality`, `segment`; time in BKJD or
  BJD_TDB; ≤ 25 MB). The page shows catalogue information separately from the model output. You can inspect the raw
  light curve with its trend, the folded transit, both model-input views, the score and the threshold, and
  download a JSON report.
* **Measured results**: metrics, an interactive PR curve, seed variation, bootstrap intervals and subgroups.
* **How it works & limits.**

If no checkpoint exists, the app says so and shows the commands that create it. Bundled examples need no network.
**The app does not search for periods.** It scores a candidate whose period, epoch and duration you supply. A blind
period search (e.g. Box Least Squares) is a possible extension.

## Differences from AstroNet

| | AstroNet (Shallue & Vanderburg 2018) | this project |
|---|---|---|
| split | "randomly partitioned" 80/10/10 by TCE: one star's TCEs can fall in different splits | **grouped by host star** (70/15/15), Kepler-90 held out |
| light curves | Kepler PDC-MAP light curves from MAST at the time (2017) | DR25 (`DATA_REL 25`) files currently served by MAST |
| other planets | transits of other confirmed planets in the system removed | not removed (that would require labels from a later catalogue) |
| detrending | paper: spline fitted with in-transit points removed, 3σ clipping; public `preprocess.py`: spline without a transit mask | spline with the candidate's transits masked; only upward outliers removed from the data |
| empty bins | filled with the median (public code) | interpolated, plus an explicit coverage-mask channel |
| normalisation | each view separately (median 0, minimum −1) | shared depth scale for both views |
| network | 5+2 conv blocks, filters 16→256 (global) / 16→32 (local); 4 × 512 FC | 5+2 blocks, filters 16→64 / 16→32; 2 × 256 FC with dropout (1.56 M parameters) |
| training | Adam, lr 1e-5, batch 64, 50 epochs, no dropout; hyperparameters tuned for validation AUC | AdamW, lr 1e-3, batch 128, early stopping on validation AP; no hyperparameter search |
| ensembling | average of 10 independently trained copies | single release model; 3 seeds reported individually |
| reported test result | AUC 0.988, accuracy 0.960 at threshold 0.5 (Tables 1–2) | see Results |

Because of these differences, and a harder star-level split, **our metrics should not be compared directly with the
paper's**.

## Repository layout

```
configs/              quickstart.yaml, full.yaml
src/exofinder/        catalog, splits, lightcurve, preprocess, pipeline, dataset,
                      baseline, model, train, metrics, evaluate, inference, upload, release, cli
app/                  Streamlit app (pages/), bundled real light curves (examples/)
data/catalog/         DR24 snapshot + provenance            (committed)
data/splits/          split manifest + info                 (committed)
data/{cache,processed,datasets}/                            (git-ignored, regenerated)
models/release/       released CNN + baseline + card        (committed)
reports/full/         metrics, predictions, figures, RESULTS.md
docs/                 WALKTHROUGH.md, figures
tests/                pytest suite
```

## Limitations

See [MODEL_CARD.md](MODEL_CARD.md). In short: the labels are DR24 dispositions, not ground truth. The model has only
been tested on Kepler long-cadence PDCSAP data. It needs a correct ephemeris. The score is uncalibrated. And DR24
labels are paired with DR25 light curves.

## References

* C. J. Shallue & A. Vanderburg (2018), *Identifying Exoplanets with Deep Learning: A Five-planet Resonant Chain around
  Kepler-80 and an Eighth Planet around Kepler-90*, AJ 155, 94 — [arXiv:1712.05044](https://arxiv.org/abs/1712.05044)
* AstroNet code: [google-research/exoplanet-ml](https://github.com/google-research/exoplanet-ml) (Apache-2.0; archived)
* NASA Exoplanet Archive: [Kepler TCE documentation](https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_TCE_docs.html),
  [TCE column definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_tce_columns.html),
  [KOI documentation](https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html),
  [TAP service](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html)
* J. H. Catanzarite (2015), *Autovetter Planet Candidate Catalog for Q1–Q17 Data Release 24*, KSCI-19090-001 (source
  of `av_training_set`)
* J. L. Coughlin et al. (2016), *Planetary Candidates Observed by Kepler. VII. The First Fully Uniform Catalog Based on
  the Entire 48-month Data Set (Q1–Q17 DR24)*, ApJS 224, 12
* MAST Kepler archive: <https://archive.stsci.edu/missions-and-data/kepler>
* Lightkurve: <https://docs.lightkurve.org/>

Licence: Apache-2.0 (see `LICENSE` and `NOTICE`). Author: Muhammad Abu Bakr Siddique.
