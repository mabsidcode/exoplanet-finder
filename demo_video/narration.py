"""Narration for the Exoplanet Finder demo, anchored to the shots in build.SCENES.

Each clip is one ElevenLabs request. It starts ``lead`` seconds after the start of segment ``seg`` of scene
``scene`` (indices into build.SCENES), so the words land on the visuals they describe. A clip is a list of caption
chunks ``(display, spoken)``: the display text is burned into the caption band and the SRT (digits, catalogue names);
the spoken text is what the voice reads (numbers written out so they are pronounced predictably).

Every claim is taken from the project's own files:
  score 0.987 / threshold 0.671 / 116 ppm / 74 transits ...... app output for Kepler-90 i (release model)
  DR24 AFP -> DR25 CANDIDATE (K00339.03) -> Kepler-529 d ...... app/examples/later_catalogue.json
  2,362 signals, 1,481 stars, AP 0.913 vs 0.873 ............... reports/full/metrics.json
"""

from __future__ import annotations

# Applied to the spoken text only. Checked with the pronunciation sample before full generation.
PRONOUNCE = {
    "Kepler-90 i": "Kepler ninety i",
    "Kepler-529 d": "Kepler five twenty-nine d",
    "DR24": "D R twenty-four",
    "DR25": "D R twenty-five",
    "CNN": "C N N",
}

CLIPS = [
    {"id": "01_intro", "scene": 0, "seg": 0, "lead": 0.6, "chunks": [
        ("Exoplanet Finder is a PyTorch model that scores possible planet signals", None),
        ("in NASA Kepler light curves.", None)]},
    {"id": "02_kepler90i", "scene": 1, "seg": 0, "lead": 0.3, "chunks": [
        ("Here's Kepler-90 i, a known planet we use as a demo.", None)]},
    {"id": "03_raw", "scene": 1, "seg": 2, "lead": 0.1, "chunks": [
        ("Four years of Kepler data:", None),
        ("the planet's dip is only about 116 parts per million.",
         "the planet's dip is only about a hundred and sixteen parts per million.")]},
    {"id": "04_folded", "scene": 1, "seg": 4, "lead": 0.5, "chunks": [
        ("Folding at the orbital period stacks 74 transits, and the dip appears.",
         "Folding at the orbital period stacks seventy-four transits, and the dip appears.")]},
    {"id": "05_global", "scene": 2, "seg": 0, "lead": 0.4, "chunks": [
        ("The network reads the whole orbit, which can reveal eclipsing binaries,", None)]},
    {"id": "06_local", "scene": 2, "seg": 2, "lead": 0.5, "chunks": [
        ("and a close-up of the dip.", None)]},
    {"id": "07_score", "scene": 3, "seg": 0, "lead": 0.3, "chunks": [
        ("It scores 0.987, above the validation threshold of 0.671:",
         "It scores zero point nine eight seven, above the validation threshold of zero point six seven one:"),
        ("it resembles labelled planet candidates.", None),
        ("That's not a probability, or a confirmation.", None)]},
    {"id": "08_disagree", "scene": 4, "seg": 0, "lead": 0.5, "chunks": [
        ("The DR24 training labels call this signal a false positive,", None),
        ("yet it scores 0.999.", "yet it scores zero point nine nine nine.")]},
    {"id": "09_dr25", "scene": 4, "seg": 2, "lead": 0.15, "chunks": [
        ("Later DR25 vetting calls it a candidate,", None),
        ("and NASA's Exoplanet Archive lists it as confirmed planet Kepler-529 d.", None),
        ("The benchmark still counts it as an error.", None)]},
    {"id": "10_eval", "scene": 5, "seg": 0, "lead": 0.35, "chunks": [
        ("On 2,362 signals from 1,481 unseen stars,",
         "On two thousand three hundred and sixty-two signals from fourteen hundred and eighty-one unseen stars,")]},
    {"id": "11_ap", "scene": 5, "seg": 2, "lead": 0.15, "chunks": [
        ("the CNN reaches an average precision of 0.913,", "the CNN reaches an average precision of zero point nine one three,"),
        ("versus 0.873 for a gradient-boosting baseline.", "versus zero point eight seven three for a gradient-boosting baseline.")]},
    {"id": "12_github", "scene": 6, "seg": 0, "lead": 0.3, "chunks": [
        ("The code, trained model and evaluation are on GitHub.", None)]},
    {"id": "13_credit", "scene": 7, "seg": 0, "lead": 0.3, "chunks": [
        ("Thanks to NASA's Kepler mission", None),
        ("and to Shallue and Vanderburg's AstroNet research.", None)]},
]

SAMPLE_TEXT = ("Exoplanet Finder is a PyTorch model for NASA Kepler light curves. "
               "Kepler ninety i scores zero point nine eight seven. "
               "On two thousand, three hundred and sixty-two test signals, the C N N reaches an average precision "
               "of zero point nine one three.")
SAMPLE_EXPECT = ["exoplanet", "pytorch", "kepler", "90", "0.987", "2,362", "cnn", "0.913"]


def spoken(chunk: tuple[str, str | None]) -> str:
    text = chunk[1] or chunk[0]
    for k, v in PRONOUNCE.items():
        text = text.replace(k, v)
    return text


def clip_text(clip: dict) -> tuple[str, list[tuple[int, int]]]:
    """Full spoken text of a clip and the [start, end) character span of each chunk within it."""
    parts, spans, pos = [], [], 0
    for ch in clip["chunks"]:
        s = spoken(ch)
        spans.append((pos, pos + len(s)))
        parts.append(s)
        pos += len(s) + 1
    return " ".join(parts), spans


def word_count() -> int:
    return sum(len(clip_text(c)[0].split()) for c in CLIPS)
