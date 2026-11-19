# LinkedIn draft (for your review; edit freely before posting)

Every number below comes from `reports/full/metrics.json` and `docs/RESULTS_REPORT.md`. Please re-check them if you
re-run anything.

---

In 2017, Christopher Shallue and Andrew Vanderburg trained a neural network on NASA Kepler data and found Kepler-90 i, an eighth planet around a star that already had seven. I wanted to understand how that works, so for my astronomy ML project I rebuilt a smaller version from scratch. 🔭

**Exoplanet Finder** takes a Kepler light curve (a star's brightness over four years) and a candidate signal's period, epoch and duration. It detrends the star's own variability, folds every transit on top of each other, and gives a 1D CNN two "views" of the dip: the whole orbit and a close-up. It returns a score for how closely the signal resembles transits that NASA's DR24 catalogue labelled planet candidates, as opposed to eclipsing binaries or instrument artefacts.

What I actually built:
• A resumable pipeline that downloaded real light curves for all 9,865 labelled stars (over 66 GB from NASA's MAST archive). After preprocessing, 15,733 of 15,737 signals, on 9,861 stars, were usable
• Train/validation/test splits grouped by star, so the model can't "recognise" a star it saw in training
• Two scikit-learn baselines, a dual-branch PyTorch CNN (1.56 M parameters), and a Streamlit demo that runs the real model

Results on 2,362 held-out signals from 1,481 stars the model never saw:
• Average precision 0.913 (a random guess scores about 0.23 here, because only 23% of signals are planet candidates)
• 91% of planet candidates recovered, at 83% precision
• Gradient-boosting baseline on hand-made features: AP 0.873

Kepler-90 was held out from all training. The model scores both Kepler-90 h (0.98) and Kepler-90 i (0.99) above its threshold. That's a nice sanity check, not a discovery: Kepler-90 i was validated by its discoverers with much more evidence than a CNN score.

What I learned:
→ Accuracy can lie. A model that always says "not a planet" is 77% accurate here and completely useless.
→ Leakage hides in the data structure. One star can have several signals, so I split by star: otherwise the test set would contain stars the model had already seen.
→ Seeds matter. My three training runs scored 0.908, 0.869 and 0.913, so I report all of them instead of the best one.
→ The model's confident mistakes taught me the most. Some catalogue "false positives" look like clean transits to the model. My working hypothesis is that they were flagged using evidence a light curve can't show, such as pixel-level centroid shifts, but I haven't yet checked that against the catalogue's diagnostics.

The code, model card and a beginner walkthrough are in the repo [add link]. Huge credit to Shallue & Vanderburg (arXiv:1712.05044), the AstroNet code from Google Research, and NASA's Exoplanet Archive and MAST for making all this data public.

#MachineLearning #DeepLearning #Astronomy #Exoplanets #PyTorch #DataScience

---

## Resume bullets

* Built an end-to-end PyTorch pipeline that classifies NASA Kepler transit signals with a dual-branch 1D CNN.
  It downloaded light curves for 9,865 stars (over 66 GB) and trained and evaluated on 15,733 usable labelled
  candidates from 9,861 stars. It reached test average precision 0.913 on star-held-out data, against 0.873 for a
  gradient-boosting baseline.
* Designed a leakage-safe evaluation: star-grouped splits, validation-only model and threshold selection,
  a protocol committed before testing, and bootstrap confidence intervals. Shipped the model as an offline Streamlit
  app with bit-identical training and inference preprocessing.

## Demo recording outline (60–90 s)

1. **Hook (5 s)**: the Kepler-90 system and its eight planets; "can a CNN spot Kepler-90 i?"
2. **Choose an example (10 s)**: open *Score a candidate* and pick *Kepler-90 h*. Point out the catalogue card
   (DR24 PC, held out from training) and that it is separate from the model output.
3. **Raw → folded (15 s)**: raw light curve with the spline trend, then the folded transit.
4. **Model inputs (10 s)**: global view and local view, including the coverage mask.
5. **Score (10 s)**: 0.982 vs threshold 0.671. Read the caveat aloud: a score is not a confirmation.
6. **Kepler-90 i (10 s)**: switch to it and, in *Folded transit*, show the ~116 ppm dip in the binned median that
   only appears after folding 74 transits.
7. **Honest failure (10 s)**: pick *Test PC with the lowest CNN score* and show the residual stellar variability
   left after detrending. Present it as the likely reason for the low score (a hypothesis from the views, not a
   checked fact).
8. **Measured results page (10 s)**: PR curve vs baselines and seed variation.
9. **Close (5 s)**: credit Shallue & Vanderburg and NASA, and add the repo link.
