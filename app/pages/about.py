import streamlit as st
from ui import setup_page

setup_page()
st.title("How it works & limitations")

st.markdown(
    """
### Transits in one paragraph
When a planet passes in front of its star, the star's measured brightness dips by roughly
(planet radius / star radius)² — about 84 ppm for an Earth-sized planet around a Sun-like star — for a few hours,
once per orbit. Kepler recorded the brightness of ~200,000 stars every ~29.4 minutes for four years. The Kepler pipeline
flagged repeating dips as **Threshold Crossing Events (TCEs)**. Many TCEs are not planets: eclipsing binary stars,
light from a neighbouring star (contamination), or instrumental artefacts can mimic a transit.

### What this app does
1. **Clean**: keep Kepler `PDCSAP_FLUX` cadences that pass the default quality bitmask; sort, remove duplicates,
   and normalise each quarter by its median.
2. **Detrend**: fit a smooth spline to each continuous segment with the candidate's transits *masked out*, and
   divide by it. Only upward outliers are removed, so real dips survive.
3. **Fold**: convert every time to an orbital phase using the supplied period and epoch, so all transits line up.
4. **Bin** into two fixed-length views (the AstroNet design): a **global view** (2001 bins across the whole orbit)
   that shows secondary eclipses and other out-of-transit structure, and a **local view** (201 bins across ±4
   transit durations) that resolves the transit's shape. Empty bins are interpolated and flagged in a coverage mask.
5. **Score**: a two-branch 1D convolutional network reads both views and outputs a score in [0, 1].

### What it does **not** do
* It does **not search for periods**. You must supply a candidate's period, epoch and duration (a blind period
  search, e.g. Box Least Squares, is a possible later extension).
* It does **not confirm planets**. The score says how much a signal resembles DR24 Autovetter-training-set
  *planet candidates* versus *false positives / non-transiting phenomena*. Confirmation needs independent evidence.
* The score is **not calibrated**: 0.8 does not mean an 80% chance of being a planet.

### Labels and their limits
* Labels: Kepler Q1–Q17 **DR24** TCE table, column `av_training_set`: PC → 1; AFP and NTP → 0; UNK excluded.
* A PC label means *planet candidate* at the time of DR24 vetting, not *confirmed planet*. Some PCs were later
  shown to be false positives and vice versa; later catalogues (e.g. DR25) can disagree.
* AFP mixes eclipsing binaries, background eclipsing binaries and contamination; NTP covers instrumental and stellar
  variability. Errors are reported separately for each.
* Light curves come from the final Kepler data release served by MAST (`DATA_REL = 25`, SOC 9.3), while labels come
  from DR24 vetting of earlier processing. This mismatch is documented and can shift some signals.

### Distribution limits
Trained only on Kepler long-cadence PDCSAP light curves of DR24 TCEs. Expect degraded or meaningless scores for
TESS/K2 data, short cadence, other flux definitions, single-transit events, or wrong ephemerides.

### Kepler-90 and attribution
Kepler-90 i was announced by **Christopher Shallue and Andrew Vanderburg (2018)**, who trained the AstroNet CNN on
DR24 TCEs and searched multi-planet systems for weak signals. This project is an independent, smaller
reimplementation inspired by that work: [arXiv:1712.05044](https://arxiv.org/abs/1712.05044) ·
[google-research/exoplanet-ml](https://github.com/google-research/exoplanet-ml) (Apache-2.0).
Kepler-90 (KIC 11442793) was reserved before training for demonstration; Kepler-90 h is its DR24 TCE and Kepler-90 i
is not in the DR24 TCE table at all.

Data: NASA Exoplanet Archive (Kepler DR24 TCE table, Planetary Systems table) and the Mikulski Archive for Space
Telescopes (Kepler light curves). This research uses data collected by the Kepler mission, funded by the NASA
Science Mission Directorate.
"""
)
