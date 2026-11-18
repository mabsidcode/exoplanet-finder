"""Make the walkthrough figures from the bundled REAL Kepler-90 light curve.

Usage: python scripts/make_walkthrough_figures.py
Writes docs/figures/walkthrough_*.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from exofinder.lightcurve import BKJD_OFFSET, light_curve_from_raw_arrays  # noqa: E402
from exofinder.preprocess import CandidateParams, PreprocessConfig, preprocess_candidate  # noqa: E402
from exofinder.release import load_bundled_raw  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_bundled_raw(ROOT / "app" / "examples" / "kplr011442793.npz")
    lc = light_curve_from_raw_arrays(raw)
    cfg = PreprocessConfig()
    h = CandidateParams(331.603, 140.48, 14.49)  # DR24 TCE 011442793_01 (Kepler-90 h)
    i = CandidateParams(14.44912, 2455644.3488 - BKJD_OFFSET, 2.80)  # Kepler-90 i, Shallue & Vanderburg 2018
    ph = preprocess_candidate(lc, h, cfg, keep_intermediates=True)
    pi = preprocess_candidate(lc, i, cfg, keep_intermediates=True)

    fig, ax = plt.subplots(3, 1, figsize=(10, 8.5))
    q = (lc.time > 538) & (lc.time < 630)
    ax[0].plot(lc.time[q], lc.flux[q], ".", ms=1, color="k")
    sel = (ph.time > 538) & (ph.time < 630)
    ax[0].plot(ph.time[sel], ph.trend[sel], "-", color="tab:orange", lw=1, label="spline trend (Kepler-90 h transits masked)")
    ax[0].set(title="1. Raw PDCSAP light curve, one Kepler quarter (Q6), each quarter median-normalised",
              xlabel="time (BKJD days)", ylabel="relative flux")
    ax[0].legend(fontsize=8)
    ax[1].plot(ph.time, ph.detrended, ".", ms=0.6, color="k")
    ax[1].set(title="2. All quarters after detrending: the deepest dips are Kepler-90 h; others are its sibling planets",
              xlabel="time (BKJD days)", ylabel="flux / trend")
    w = np.abs(pi.phase) < cfg.local_num_durations * i.duration_days
    ax[2].plot(pi.phase[w] * 24, (pi.detrended[w] - 1) * 1e6, ".", ms=1, color="0.6", label="individual observations")
    lv = pi.local_view
    ax[2].plot(pi.local_centers * 24, lv[0] * pi.diagnostics["depth_ppm"], "-", color="tab:blue", lw=1.5,
               label="local view (binned median, rescaled to ppm)")
    ax[2].set(ylim=(-600, 400), title="3. Folded on Kepler-90 i's period (14.449 d): ~74 transits stack into one ~100 ppm dip",
              xlabel="hours from transit centre", ylabel="flux change (ppm)")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "walkthrough_stages.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(11, 5))
    for col, (name, pc, p) in enumerate([("Kepler-90 h (P = 331.6 d)", ph, h), ("Kepler-90 i (P = 14.45 d)", pi, i)]):
        ax[0, col].plot(pc.global_centers / p.period_days, pc.global_view[0], lw=0.6, color="k")
        ax[0, col].set(title=f"{name}: global view (2001 bins)", xlabel="phase (fraction of period)")
        ax[1, col].plot(pc.local_centers / p.duration_days, pc.local_view[0], ".-", ms=2, lw=0.8, color="tab:blue")
        ax[1, col].set(title=f"{name}: local view (201 bins)", xlabel="time from centre (transit durations)")
    for a in ax.flat:
        a.set_ylabel("normalised flux")
    fig.tight_layout()
    fig.savefig(OUT / "walkthrough_views.png", dpi=110)
    plt.close(fig)
    print("depth h", ph.diagnostics["depth_ppm"], "depth i", pi.diagnostics["depth_ppm"], "snr i", pi.diagnostics["snr_est"],
          "transits i", pi.diagnostics["n_transits_observed"])


if __name__ == "__main__":
    main()
