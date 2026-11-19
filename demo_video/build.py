"""Edit the recorded shots into the LinkedIn demo video (1920x1080, 30 fps, H.264 yuv420p) plus SRT and thumbnail.

Every output frame is composed here with Pillow and piped to FFmpeg. The app capture fills the top 1920x960 and the
captions sit in a separate 120 px band below it, so they never cover the scores, charts or transit dips.

Usage:  python demo_video/build.py            (after demo_video/record.py)
Output: demo_video/exoplanet_finder_linkedin.mp4, demo_video/captions.srt, demo_video/thumbnail.jpg
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
OUT_MP4 = HERE / "exoplanet_finder_linkedin.mp4"
OUT_SRT = HERE / "captions.srt"
OUT_THUMB = HERE / "thumbnail.jpg"
W, H, APP_H, FPS = 1920, 1080, 960, 30
CSS_W, CSS_H, DSF = 1440, 720, 2.0
BG, PANEL, TEXT, MUTED, ACCENT = (10, 14, 26), (19, 26, 44), (228, 232, 241), (167, 176, 197), (143, 179, 255)
FONT = "/System/Library/Fonts/Avenir Next.ttc"
F_DEMI, F_MED, F_REG, F_BOLD = 2, 5, 7, 0
REPO = "https://github.com/mabsidcode/exoplanet-finder"


def font(size, face=F_DEMI):
    return ImageFont.truetype(FONT, size, index=face)


# --------------------------------------------------------------------------- timeline model


@dataclass
class Seg:
    kind: str                      # "motion" | "hold" | "card"
    dur: float
    shot: str = ""                 # motion: rec/<shot>; hold: stills/<shot>.png; card: renderer name
    speed: float = 1.0             # motion playback speed
    zoom: tuple | None = None      # hold: target crop (x, y, w, h) in CSS px, w:h = 2:1
    fade_in: float = 0.0           # crossfade from the previous segment's last frame
    edge_fade: tuple = (0, 0)      # hold: soft fade (px) at the top / bottom of the app area for clipped content
    captions: list = field(default_factory=list)  # [(start, end, line1, line2)] relative to segment start


SCENES = [
    # 1. Opening
    [Seg("card", 4.6, "title")],
    # 2. Kepler-90 i: select, raw, folded
    [Seg("motion", 0, "k90_select", speed=1.35, fade_in=0.5,
         captions=[(0, 4.4, "Kepler-90 i: a planet already known from published work, used here as a demonstration",
                    "Real Kepler light curve bundled with the app; the Kepler-90 star was held out from training")]),
     Seg("hold", 1.3, "k90_top", fade_in=0.15),
     Seg("motion", 0, "k90_scroll_raw", speed=1.1,
         captions=[(0, 4.4, "Raw light curve: four years of Kepler brightness measurements",
                    "Kepler-90 i dims its star by only about 116 parts per million, invisible at this scale")]),
     Seg("hold", 3.2, "k90_raw", zoom=(350, 50, 1040, 520), fade_in=0.15),
     Seg("motion", 0, "k90_folded", speed=1.25, fade_in=0.3,
         captions=[(0, 6.6, "Folding at the 14.45-day period stacks 74 observed transits on top of each other",
                    "The repeated dips line up, so the transit becomes visible (blue line: binned median)")]),
     Seg("hold", 4.4, "k90_folded", zoom=(350, 50, 1040, 520), fade_in=0.15)],
    # 3. Model inputs
    [Seg("motion", 0, "k90_global", speed=1.25, fade_in=0.3,
         captions=[(0, 5.8, "Model input 1, the global view: the whole orbit in 2,001 bins",
                    "It shows structure away from the transit, such as eclipses of a companion star")]),
     Seg("hold", 3.6, "k90_global", zoom=(350, 50, 1040, 520), fade_in=0.15),
     Seg("motion", 0, "k90_local", speed=1.25, fade_in=0.3,
         captions=[(0, 5.9, "Model input 2, the local view: a close-up of the dip in 201 bins",
                    "The 1D convolutional network reads both views together")]),
     Seg("hold", 3.6, "k90_local", zoom=(350, 50, 1040, 520), fade_in=0.15)],
    # 4. Prediction
    [Seg("motion", 0, "k90_score", speed=1.15, fade_in=0.3,
         captions=[(0, 5.3, "Model score 0.987, above the validation-selected threshold of 0.671",
                    "The threshold was chosen on validation data, never on the test set"),
                   (5.3, 9.5, "Scores indicate resemblance to labelled planet candidates",
                    "Not a probability, and not a planet confirmation")]),
     Seg("hold", 7.7, "k90_score", zoom=(845, 86, 595, 297.5), fade_in=0.15)],  # model-output card only
    # 5. Catalogue disagreement (Kepler-529 d)
    [Seg("motion", 0, "afp_select", speed=1.35, fade_in=0.4,
         captions=[(0, 6.5, "KIC 10587105 TCE 3 is labelled AFP (astrophysical false positive) in Kepler DR24",
                    "The model scores it 0.999, so the DR24-based benchmark counts it as an error")]),
     Seg("hold", 3.7, "afp_top", zoom=(360, 190, 1040, 520), fade_in=0.35, edge_fade=(0, 70)),
     Seg("motion", 0, "afp_later", speed=1.1, fade_in=0.3,
         captions=[(0, 4.6, "Later DR25 vetting lists the matched KOI K00339.03 as CANDIDATE",
                    "NASA Exoplanet Archive, Planetary Systems table: confirmed planet Kepler-529 d"),
                   (4.6, 8.9, "Later-catalogue context only: the evaluation still uses the original DR24 labels",
                    "Matched by star, period and transit timing; catalogues checked 19 Nov 2026")]),
     Seg("hold", 7.7, "afp_later", zoom=(240, 56, 1200, 600), fade_in=0.15, edge_fade=(80, 0))],
    # 6. Evaluation
    [Seg("motion", 0, "results_nav", speed=1.2, fade_in=0.4,
         captions=[(0, 6.0, "Evaluation: 2,362 test signals from 1,481 held-out host stars",
                    "Splits are grouped by host star, so no star appears in both training and test")]),
     Seg("hold", 4.6, "results_top", zoom=(360, 40, 1040, 520), fade_in=0.35),
     Seg("motion", 0, "results_pr", speed=1.1, fade_in=0.3,
         captions=[(0, 7.3, "Average precision (AP) on the test set: CNN 0.913 vs gradient-boosting baseline 0.873",
                    "Precision–recall curve; random ranking would score about 0.23, the planet-candidate share")]),
     Seg("hold", 6.1, "results_pr", zoom=(350, 52, 1060, 530), fade_in=0.15)],
    # 7. The public repository (real GitHub page, signed out)
    [Seg("hold", 2.6, "github_top", zoom=(100, 40, 1240, 620), fade_in=0.5,
         captions=[(0, 2.6, "Open source on GitHub", "Code, trained model, evaluation and a beginner walkthrough")])],
    # 8. Closing
    [Seg("card", 5.8, "closing", fade_in=0.5)],
]

# --------------------------------------------------------------------------- frame sources

_cache: dict = {}


def still(name) -> Image.Image:
    if name not in _cache:
        _cache[name] = Image.open(BUILD / "stills" / f"{name}.png").convert("RGB")
    return _cache[name]


def app_view(name, rect=None) -> Image.Image:
    """Crop (CSS px, 2:1) of a full-resolution still, scaled to 1920x960."""
    im = still(name)
    x, y, w, h = rect or (0, 0, CSS_W, CSS_H)
    box = tuple(round(v * DSF) for v in (x, y, x + w, y + h))
    return im.resize((W, APP_H), Image.Resampling.LANCZOS, box=box)


def motion_frames(shot, speed) -> list[Image.Image]:
    meta = json.loads((BUILD / "rec" / shot / "frames.json").read_text())
    ts = np.array([m["t"] for m in meta])
    dur = (ts[-1] - ts[0]) / speed
    out = []
    for k in range(max(1, round(dur * FPS))):
        t = ts[0] + k / FPS * speed
        i = int(np.searchsorted(ts, t, side="right") - 1)
        key = (shot, i)
        if key not in _cache:
            _cache[key] = Image.open(BUILD / "rec" / shot / meta[i]["file"]).convert("RGB").resize(
                (W, APP_H), Image.Resampling.LANCZOS)
        out.append(_cache[key])
    return out


_fade_cache: dict = {}


def _edge_fade(img: Image.Image, top: int, bottom: int) -> Image.Image:
    key = (id(img), top, bottom)
    if key not in _fade_cache:
        mask = np.zeros((APP_H, 1), np.float32)
        if top:
            mask[:top, 0] = np.linspace(1, 0, top) ** 1.5
        if bottom:
            mask[APP_H - bottom:, 0] = np.linspace(0, 1, bottom) ** 1.5
        a = np.asarray(img, np.float32)
        out = a * (1 - mask[..., None]) + np.array(BG, np.float32) * mask[..., None]
        _fade_cache[key] = Image.fromarray(out.astype(np.uint8))
    return _fade_cache[key]


def ease(k):
    k = min(1.0, max(0.0, k))
    return 4 * k**3 if k < 0.5 else 1 - (-2 * k + 2) ** 3 / 2


# --------------------------------------------------------------------------- caption band and cards


def _fit(text, size, face, max_w):
    while size > 20:
        f = font(size, face)
        if f.getlength(text) <= max_w:
            return f
        size -= 1
    return font(size, face)


_band_cache: dict = {}


def caption_band(line1, line2, alpha=1.0) -> Image.Image:
    key = (line1, line2, round(alpha, 2))
    if key in _band_cache:
        return _band_cache[key]
    band = Image.new("RGB", (W, H - APP_H), BG)
    d = ImageDraw.Draw(band)
    d.rectangle([0, 0, W, 2], fill=(34, 44, 70))
    if line1 and alpha > 0:
        txt = Image.new("RGBA", band.size, (0, 0, 0, 0))
        dt = ImageDraw.Draw(txt)
        f1 = _fit(line1, 40, F_DEMI, W - 160)
        dt.text((W / 2, 40), line1, font=f1, fill=TEXT + (int(255 * alpha),), anchor="mm")
        if line2:
            f2 = _fit(line2, 31, F_MED, W - 160)
            dt.text((W / 2, 88), line2, font=f2, fill=MUTED + (int(255 * alpha),), anchor="mm")
        band = Image.alpha_composite(band.convert("RGBA"), txt).convert("RGB")
    _band_cache[key] = band
    return band


def frame(app_img: Image.Image, cap) -> Image.Image:
    out = Image.new("RGB", (W, H), BG)
    out.paste(app_img, (0, 0))
    out.paste(caption_band(*cap), (0, APP_H))
    return out


def _background(name, rect, blur, dim) -> Image.Image:
    im = still(name)
    x, y, w, h = rect
    bg = im.resize((W, H), Image.Resampling.LANCZOS, box=tuple(round(v * DSF) for v in (x, y, x + w, y + h)))
    if blur:
        bg = bg.filter(ImageFilter.GaussianBlur(blur))
    return Image.blend(bg, Image.new("RGB", (W, H), BG), dim)


def title_card() -> Image.Image:
    # genuine app frame (Kepler-90 i folded transit) behind the title
    img = _background("k90_folded", (300, 40, 1080, 607.5), 7, 0.72)
    d = ImageDraw.Draw(img)
    d.text((W / 2, 455), "Exoplanet Finder", font=font(124, F_BOLD), fill=TEXT, anchor="mm")
    d.rectangle([W / 2 - 70, 545, W / 2 + 70, 549], fill=ACCENT)
    d.text((W / 2, 615), "A 1D CNN for classifying signals in NASA Kepler light curves", font=font(50, F_MED),
           fill=TEXT, anchor="mm")
    return img


def closing_card() -> Image.Image:
    img = _background("github_top", (80, 0, 1280, 720), 9, 0.93)
    d = ImageDraw.Draw(img)
    d.text((W / 2, 330), "Built by Muhammad Abu Bakr Siddique", font=font(76, F_BOLD), fill=TEXT, anchor="mm")
    d.text((W / 2, 425), "Code, trained model and evaluation", font=font(46, F_MED), fill=MUTED, anchor="mm")
    d.rounded_rectangle([W / 2 - 600, 485, W / 2 + 600, 575], radius=14, fill=PANEL, outline=(46, 74, 122), width=2)
    d.text((W / 2, 530), REPO, font=font(50, F_DEMI), fill=ACCENT, anchor="mm")
    d.text((W / 2, 720), "Data: NASA Kepler mission, via MAST and the NASA Exoplanet Archive", font=font(34, F_MED),
           fill=TEXT, anchor="mm")
    d.text((W / 2, 772), "Inspired by Shallue & Vanderburg (2018), “Identifying Exoplanets with Deep Learning” (AstroNet)",
           font=font(34, F_MED), fill=TEXT, anchor="mm")
    return img


CARDS = {"title": title_card, "closing": closing_card}
CARD_TEXT = {
    "title": ("Exoplanet Finder", "A 1D CNN for classifying signals in NASA Kepler light curves"),
    "closing": ("Built by Muhammad Abu Bakr Siddique · Code, trained model and evaluation", REPO,
                "Data: NASA Kepler mission, via MAST and the NASA Exoplanet Archive",
                "Inspired by Shallue & Vanderburg (2018), AstroNet"),
}

# --------------------------------------------------------------------------- render


def caption_at(caps, t):
    for a, b, l1, l2 in caps:
        if a <= t < b:
            fade = 0.25
            alpha = min(1.0, (t - a) / fade if a > 0 else 1.0, (b - t) / fade)
            return (l1, l2, max(0.0, alpha))
    return ("", "", 0.0)


def _wrap2(text, size, face, max_w):
    """Split a caption into at most two balanced lines that fit max_w."""
    f = font(size, face)
    if f.getlength(text) <= max_w:
        return [text], f
    words = text.split()
    best = None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        w = max(f.getlength(a), f.getlength(b))
        if best is None or w < best[0]:
            best = (w, [a, b])
    if best[0] > max_w:
        return best[1], _fit(max(best[1], key=f.getlength), size, face, max_w)
    return best[1], f


def narration_band(text, alpha=1.0) -> Image.Image:
    """Caption band for the narrated version: the spoken sentence, one or two lines of equal weight."""
    key = ("narr", text, round(alpha, 2))
    if key in _band_cache:
        return _band_cache[key]
    band = Image.new("RGB", (W, H - APP_H), BG)
    ImageDraw.Draw(band).rectangle([0, 0, W, 2], fill=(34, 44, 70))
    if text and alpha > 0:
        txt = Image.new("RGBA", band.size, (0, 0, 0, 0))
        dt = ImageDraw.Draw(txt)
        lines, f = _wrap2(text, 40, F_DEMI, W - 160)
        ys = [60] if len(lines) == 1 else [36, 86]
        for y, line in zip(ys, lines, strict=True):
            dt.text((W / 2, y), line, font=f, fill=TEXT + (int(255 * alpha),), anchor="mm")
        band = Image.alpha_composite(band.convert("RGBA"), txt).convert("RGB")
    _band_cache[key] = band
    return band


def narration_caption_at(cues, t):
    for a, b, text in cues:
        if a <= t < b:
            fade = 0.18
            return text, max(0.0, min(1.0, (t - a) / fade, (b - t) / fade))
    return "", 0.0


def render(scenes=None, out_mp4=OUT_MP4, out_srt=OUT_SRT, narration=None):
    """narration: None for the caption-led silent cut, or {"cues": [(start, end, text)], "audio": wav_path}."""
    scenes = SCENES if scenes is None else scenes
    frames_written = 0
    srt = []
    audio_in = (["-i", str(narration["audio"])] if narration else
                ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-shortest"])
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-", *audio_in, "-map", "0:v", "-map", "1:a",
         "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
         "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-profile:v", "high", "-level", "4.1",
         "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
         "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709", "-g", "60",
         "-c:a", "aac", "-b:a", "192k" if narration else "96k", "-ar", "48000", "-movflags", "+faststart",
         str(out_mp4)],
        stdin=subprocess.PIPE)
    prev_last: Image.Image | None = None

    def emit(img):
        nonlocal frames_written
        ff.stdin.write(img.tobytes())
        frames_written += 1

    def with_caption(img, app=None):
        """Narrated cut: caption the spoken sentence at the current output time, on every frame."""
        text, alpha = narration_caption_at(narration["cues"], frames_written / FPS)
        out = img.copy() if app is None else img
        out.paste(narration_band(text, alpha), (0, APP_H))
        return out

    for scene in scenes:
        caps_scene, t_scene = [], frames_written / FPS
        # captions are attached to a segment but may run on into following segments of the same scene
        seg_start = []
        t = 0.0
        for seg in scene:
            seg_start.append(t)
            n = len(motion_frames(seg.shot, seg.speed)) if seg.kind == "motion" else round(seg.dur * FPS)
            t += n / FPS
        for seg, s0 in zip(scene, seg_start, strict=True):
            caps_scene += [(s0 + a, s0 + b, l1, l2) for a, b, l1, l2 in seg.captions]
        # each caption runs until the next one starts (or the scene ends): no gaps, no overlaps
        caps_scene.sort()
        caps_scene = [(a, caps_scene[i + 1][0] if i + 1 < len(caps_scene) else t, l1, l2)
                      for i, (a, _b, l1, l2) in enumerate(caps_scene)]
        for a, b, l1, l2 in caps_scene:
            srt.append((t_scene + a, t_scene + b, l1, l2))
        for seg, s0 in zip(scene, seg_start, strict=True):
            if seg.kind == "card":
                card = CARDS[seg.shot]()
                n = round(seg.dur * FPS)
                lines = CARD_TEXT[seg.shot]
                if not narration:
                    srt.append((frames_written / FPS, (frames_written + n) / FPS, lines[0], "\n".join(lines[1:])))
                for k in range(n):
                    img = card if not narration else with_caption(card)
                    if seg.shot == "title" and k < 0.6 * FPS:                       # fade in from black
                        img = Image.blend(Image.new("RGB", (W, H), (0, 0, 0)), card, ease(k / (0.6 * FPS)))
                    if seg.shot == "closing" and k >= n - 0.9 * FPS:                 # fade out to black
                        img = Image.blend(card, Image.new("RGB", (W, H), (0, 0, 0)), ease((k - (n - 0.9 * FPS)) / (0.9 * FPS)))
                    if seg.fade_in and prev_last is not None and k < seg.fade_in * FPS:
                        img = Image.blend(prev_last, img, ease(k / (seg.fade_in * FPS)))
                    emit(img)
                    last = img
            else:
                if seg.kind == "motion":
                    apps = motion_frames(seg.shot, seg.speed)
                    seg.fade_in = max(seg.fade_in, 0.17)  # hide the screencast's late first frame
                else:
                    n = round(seg.dur * FPS)
                    full = app_view(seg.shot)
                    target = app_view(seg.shot, seg.zoom) if seg.zoom else None
                    apps = []
                    # stronger zooms take longer, so the motion per frame stays gentle
                    zoom_frames = round(FPS * (0.9 + 0.25 * np.log2(CSS_W / seg.zoom[2]))) if seg.zoom else 1
                    for k in range(n):
                        if target is None or k == 0:
                            apps.append(full)
                        elif k >= zoom_frames:
                            apps.append(target)
                        else:  # smooth zoom from the full page to the region the caption talks about
                            e = ease(k / zoom_frames)
                            x, y, w, h = seg.zoom
                            r = (x * e, y * e, CSS_W + (w - CSS_W) * e, CSS_H + (h - CSS_H) * e)
                            apps.append(app_view(seg.shot, r))
                if seg.kind == "hold" and any(seg.edge_fade):
                    apps = [_edge_fade(a, *seg.edge_fade) for a in apps]
                for k, app in enumerate(apps):
                    tt = s0 + k / FPS
                    img = frame(app, caption_at(caps_scene, tt)) if not narration else with_caption(frame(app, ("", "", 0)), app)
                    if seg.fade_in and prev_last is not None and k < seg.fade_in * FPS:
                        img = Image.blend(prev_last, img, ease(k / (seg.fade_in * FPS)))
                    emit(img)
                    last = img
            prev_last = last
    ff.stdin.close()
    ff.wait()
    if ff.returncode:
        raise SystemExit("ffmpeg failed")
    if narration:
        srt = [(a, b, *(_wrap2(text, 40, F_DEMI, W - 160)[0] + [""])[:2]) for a, b, text in narration["cues"]]
    with open(out_srt, "w", encoding="utf-8") as f:
        for i, (a, b, l1, l2) in enumerate(srt, 1):
            f.write(f"{i}\n{_ts(a)} --> {_ts(b)}\n{l1}\n{l2}\n\n" if l2 else f"{i}\n{_ts(a)} --> {_ts(b)}\n{l1}\n\n")
    print(f"wrote {out_mp4} ({frames_written} frames, {frames_written / FPS:.2f} s) and {out_srt} ({len(srt)} cues)")
    return frames_written / FPS


def _ts(t):
    ms = round(t * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def thumbnail():
    """Genuine app frame (Kepler-90 i folded-transit chart, real model output) under a clean title band."""
    img = Image.new("RGB", (W, H), BG)
    x, y, w, h = 380, 129.6, 980, 370            # the folded-transit chart, CSS px (k90_folded still)
    chart = still("k90_folded").crop(tuple(round(v * DSF) for v in (x, y, x + w, y + h)))
    ch = round(W * h / w)
    img.paste(chart.resize((W, ch), Image.Resampling.LANCZOS), (0, H - ch))
    d = ImageDraw.Draw(img)
    d.text((80, 112), "Exoplanet Finder", font=font(120, F_BOLD), fill=TEXT, anchor="lm")
    d.text((84, 214), "A 1D CNN for classifying signals in NASA Kepler light curves", font=font(48, F_MED),
           fill=MUTED, anchor="lm")
    label = "Kepler-90 i · folded transit · model score 0.987"
    f = font(34, F_DEMI)
    x0 = 84
    x1 = x0 + f.getlength(label) + 50
    d.rounded_rectangle([x0, 262, x1, 320], radius=12, fill=PANEL, outline=(46, 74, 122), width=2)
    d.text(((x0 + x1) / 2, 291), label, font=f, fill=ACCENT, anchor="mm")
    img.save(OUT_THUMB, quality=92)
    print(f"wrote {OUT_THUMB}")


if __name__ == "__main__":
    render()
    thumbnail()
