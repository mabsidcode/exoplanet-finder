"""Build the narrated cut: align narration clips to their shots, extend holds so every sentence finishes over
its own visuals, mix and loudness-normalise the voice track, and render with narration-matched captions.

Usage:  python demo_video/narrated.py [--backend elevenlabs|preview]
Output: elevenlabs -> demo_video/exoplanet_finder_linkedin_narrated.mp4 + demo_video/captions_narrated.srt
        preview    -> demo_video/build/preview/narrated_preview.mp4 (+ .srt)   (stand-in voice, timing test only)
The original caption-led silent video (demo_video/exoplanet_finder_linkedin.mp4) is not touched.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build as B  # noqa: E402
import narration as N  # noqa: E402

SR = 48000
GAP = 0.35            # silence between consecutive sentences in a scene
TAIL = 0.45           # silence after the last sentence of a scene, before the transition
FINAL_TAIL = 1.6      # last sentence ends this long before the end (the closing card fades out over 0.9 s)
TARGET_LUFS, TARGET_TP = -16.0, -1.5
MAX_LEN = 90.0
MIN_CARD = {"title": 3.2, "closing": 4.5}   # seconds the title / closing card stay up even without speech


def seg_frames(seg) -> int:
    return len(B.motion_frames(seg.shot, seg.speed)) if seg.kind == "motion" else round(seg.dur * B.FPS)


def trim_trailing_silence(clips_meta: dict, keep=0.08) -> None:
    """Use each clip only up to where its speech ends (+keep s): the solver adds its own pauses between sentences."""
    for m in clips_meta["clips"]:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-i", m["file"], "-af", "silencedetect=noise=-45dB:d=0.12",
                            "-f", "null", "-"], capture_output=True, text=True).stderr
        starts = [float(v) for v in re.findall(r"silence_start: ([\d.]+)", r)]
        ends = [float(v) for v in re.findall(r"silence_end: ([\d.]+)", r)]
        if starts and (len(ends) < len(starts) or ends[-1] >= m["duration"] - 0.02) and starts[-1] > 0.5:
            m["full_duration"] = m["duration"]
            m["duration"] = min(m["duration"], starts[-1] + keep)
            m["chunks"] = [(a, min(b, m["duration"])) for a, b in m["chunks"]]


def plan(clips_meta: dict):
    """Return adjusted scenes and absolute clip placements [(clip_meta, start_s)]."""
    scenes = copy.deepcopy(B.SCENES)
    # Narration sets the pace: start every hold at a readable minimum (its zoom finishes and is seen for 0.9 s),
    # then lengthen holds only as far as the speech needs.
    for sc in scenes:
        for seg in sc:
            if seg.kind == "hold":
                zoom_s = (0.9 + 0.25 * np.log2(B.CSS_W / seg.zoom[2])) if seg.zoom else 0.0
                seg.dur = min(seg.dur, max(1.0, zoom_s + 0.9))
            elif seg.kind == "card":
                seg.dur = min(seg.dur, MIN_CARD[seg.shot])
    by_id = {c["id"]: c for c in clips_meta["clips"]}
    placements = []
    t_scene = 0.0
    prev_end_abs = 0.0
    for si, scene in enumerate(scenes):
        clips = sorted((c for c in N.CLIPS if c["scene"] == si), key=lambda c: (c["seg"], c["lead"]))

        def starts(scene=scene):
            out, t = [], 0.0
            for seg in scene:
                out.append(t)
                t += seg_frames(seg) / B.FPS
            return out, t

        prev_end = prev_end_abs - t_scene
        for c in clips:
            m = by_id[c["id"]]
            lead = c["lead"]
            st, _ = starts()
            want = max(st[c["seg"]] + lead, prev_end + GAP)
            delta = want - (st[c["seg"]] + lead)
            if delta > 1e-6:
                holds = [j for j in range(c["seg"]) if scene[j].kind in ("hold", "card")]
                if holds:
                    scene[holds[-1]].dur += delta      # hold the previous shot until the last sentence is done
                else:
                    lead += delta                      # first segment of a scene: start a little later
            st, _ = starts()
            start = st[c["seg"]] + lead
            placements.append((m, t_scene + start))
            prev_end = start + m["duration"]
        tail = FINAL_TAIL if si == len(scenes) - 1 else TAIL
        _, total = starts()
        if clips and prev_end + tail > total:
            last = [j for j, s in enumerate(scene) if s.kind in ("hold", "card")][-1]
            scene[last].dur += prev_end + tail - total
        _, total = starts()
        prev_end_abs = t_scene + prev_end if clips else prev_end_abs
        t_scene += total
    total_frames = sum(seg_frames(s) for sc in scenes for s in sc)
    return scenes, placements, total_frames / B.FPS


def cues(placements, video_len) -> list[tuple[float, float, str]]:
    """One caption per chunk, shown from just before it is spoken until the next chunk (or a short hold)."""
    raw = []
    clip_by_id = {c["id"]: c for c in N.CLIPS}
    for m, start in placements:
        for (a, b), chunk in zip(m["chunks"], clip_by_id[m["id"]]["chunks"], strict=True):
            raw.append([start + a - 0.12, start + b, chunk[0]])
    out = []
    for i, (a, b, text) in enumerate(raw):
        nxt = raw[i + 1][0] if i + 1 < len(raw) else video_len
        end = nxt if nxt - b < 1.2 else b + 0.8
        out.append((max(0.0, a), min(end, nxt, video_len), text))
    return out


def decode(path) -> np.ndarray:
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(pcm, np.float32).copy()


def mix(placements, video_len, out_wav: Path) -> None:
    track = np.zeros(int(round(video_len * SR)) + SR, np.float32)
    for m, start in placements:
        x = decode(m["file"])[: int(round(m["duration"] * SR))]
        f = int(0.01 * SR)
        x[:f] *= np.linspace(0, 1, f)
        x[-f:] *= np.linspace(1, 0, f)
        i = int(round(start * SR))
        track[i:i + len(x)] += x[: max(0, len(track) - i)]
    track = track[: int(round(video_len * SR))]
    raw = out_wav.with_suffix(".raw.wav")
    _write_wav(raw, track)
    # two-pass EBU R128 loudness normalisation to -16 LUFS integrated, -1.5 dBTP
    first = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(raw), "-af",
                            f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA=11:print_format=json", "-f", "null", "-"],
                           capture_output=True, text=True).stderr
    meas = json.loads(first[first.rindex("{"): first.rindex("}") + 1])
    af = (f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA=11:measured_I={meas['input_i']}:"
          f"measured_TP={meas['input_tp']}:measured_LRA={meas['input_lra']}:measured_thresh={meas['input_thresh']}:"
          f"offset={meas['target_offset']}:linear=true,aresample={SR}")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-af", af, "-ac", "2", "-c:a", "pcm_s16le",
                    str(out_wav)], check=True)


def _write_wav(path, x):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
                    "-c:a", "pcm_f32le", str(path)], input=x.astype(np.float32).tobytes(), check=True)


def verify(mp4: Path, placements, video_len) -> dict:
    """Loudness, peaks/clipping, audible speech in every clip window, silence elsewhere, sentences end in time."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(mp4), "-map", "0:a", "-af", "ebur128=peak=true",
                        "-f", "null", "-"], capture_output=True, text=True).stderr
    integ = float(re.findall(r"I:\s+(-?[\d.]+) LUFS", r)[-1])
    peak = float(re.findall(r"Peak:\s+(-?[\d.]+) dBFS", r)[-1])
    x = decode(mp4)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(mp4)], capture_output=True, text=True).stdout)
    clipped = int((np.abs(x) >= 0.999).sum())

    def rms_db(a, b):
        seg = x[int(a * SR): int(b * SR)]
        return float(20 * np.log10(np.sqrt(np.mean(seg**2)) + 1e-9)) if len(seg) else -120.0

    windows = [(s, s + m["duration"], m["id"]) for m, s in placements]
    speech = {cid: round(rms_db(a, b), 1) for a, b, cid in windows}
    gaps = [(windows[i][1] + 0.05, windows[i + 1][0] - 0.05) for i in range(len(windows) - 1)
            if windows[i + 1][0] - windows[i][1] > 0.2]
    gap_db = max((rms_db(a, b) for a, b in gaps), default=-120.0)
    last_end = max(b for _, b, _ in windows)
    res = {"duration_s": round(dur, 2), "integrated_lufs": integ, "true_peak_dbfs": peak, "clipped_samples": clipped,
           "quietest_clip_rms_dbfs": min(speech.values()), "loudest_gap_rms_dbfs": round(gap_db, 1),
           "last_sentence_ends_s": round(last_end, 2), "seconds_after_last_sentence": round(dur - last_end, 2),
           "per_clip_rms_dbfs": speech}
    res["checks"] = {
        "length 60-90 s": 60 <= dur <= MAX_LEN,
        "loudness -16 +/- 1 LUFS": abs(integ - TARGET_LUFS) <= 1.0,
        "true peak <= -1 dBTP": peak <= -1.0,
        "no clipped samples": clipped == 0,
        "every clip audible (> -35 dBFS RMS)": min(speech.values()) > -35,
        "gaps silent (< -50 dBFS RMS)": gap_db < -50,
        "last sentence ends >= 1 s before the end": dur - last_end >= 1.0,
    }
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["elevenlabs", "preview"], default="elevenlabs")
    a = ap.parse_args()
    meta = json.loads((HERE / "build" / "narration" / a.backend / "clips.json").read_text())
    trim_trailing_silence(meta)
    scenes, placements, video_len = plan(meta)
    print(f"planned length {video_len:.2f} s (silent cut {sum(seg_frames(s) for sc in B.SCENES for s in sc) / B.FPS:.2f} s)")
    for sc_old, sc_new in zip(B.SCENES, scenes, strict=True):
        for o, n in zip(sc_old, sc_new, strict=True):
            if n.kind != "motion" and abs(n.dur - o.dur) > 0.01:
                print(f"  {n.shot}: {o.dur:.2f} -> {n.dur:.2f} s")
    if video_len > MAX_LEN:
        raise SystemExit(f"Narrated cut would be {video_len:.1f} s (> {MAX_LEN} s): shorten the script or raise --speed")
    if a.backend == "elevenlabs":
        out_mp4, out_srt = HERE / "exoplanet_finder_linkedin_narrated.mp4", HERE / "captions_narrated.srt"
    else:
        (HERE / "build" / "preview").mkdir(parents=True, exist_ok=True)
        out_mp4, out_srt = HERE / "build" / "preview" / "narrated_preview.mp4", HERE / "build" / "preview" / "narrated_preview.srt"
    wav = HERE / "build" / f"narration_{a.backend}.wav"
    mix(placements, video_len, wav)
    cue_list = cues(placements, video_len)
    B.render(scenes, out_mp4, out_srt, narration={"cues": cue_list, "audio": wav})
    res = verify(out_mp4, placements, video_len)
    (HERE / "build" / f"narrated_check_{a.backend}.json").write_text(json.dumps(
        {**res, "voice": meta.get("voice"), "licence": meta.get("licence"),
         "placements": [(m["id"], round(s, 3), round(m["duration"], 3)) for m, s in placements]}, indent=2))
    for k, v in res["checks"].items():
        print(f"  {'PASS' if v else 'FAIL'} {k}")
    print({k: v for k, v in res.items() if k not in ("checks", "per_clip_rms_dbfs")})
    return 0 if all(res["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
