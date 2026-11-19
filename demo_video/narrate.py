"""Generate the demo narration with ElevenLabs (or a local macOS `say` stand-in for timing tests).

Credentials: the API key is read from the ELEVENLABS_API_KEY environment variable, or else from the macOS Keychain
item "elevenlabs-api-key" (see README instructions in the hand-off). It is never printed, logged or written to disk.

Usage:
  python demo_video/narrate.py check            plan, credits, voices (no credits used)
  python demo_video/narrate.py sample           short pronunciation/pacing sample + speech-to-text check
  python demo_video/narrate.py generate         all narration clips (cached; refuses if credits are short)
  python demo_video/narrate.py preview          local stand-in voice (macOS `say`) to test timing only

Options: --voice NAME (default: first available of George, Daniel, Brian, Alice), --speed 0.7-1.2 (default 1.05),
         --model ID (default eleven_multilingual_v2)
Output:  demo_video/build/narration/<backend>/<clip>.mp3|.aiff + clips.json (durations, chunk timings, voice)
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import narration as N  # noqa: E402

API = "https://api.elevenlabs.io/v1"
OUT = HERE / "build" / "narration"
VOICE_PREFERENCE = ["George", "Daniel", "Brian", "Alice"]
KEYCHAIN_SERVICE = "elevenlabs-api-key"


# --------------------------------------------------------------------------- credentials and HTTP


def api_key() -> str:
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not key and sys.platform == "darwin":
        r = subprocess.run(["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
                           capture_output=True, text=True)
        key = r.stdout.strip() if r.returncode == 0 else ""
    if not key:
        raise SystemExit(
            "No ElevenLabs API key found. Set ELEVENLABS_API_KEY for this shell, or store it in the macOS Keychain:\n"
            f"  security add-generic-password -a \"$USER\" -s {KEYCHAIN_SERVICE} -w\n"
            "(you are prompted for the key; it never appears in shell history).")
    return key


def _request(method: str, path: str, **kw) -> requests.Response:
    headers = {"xi-api-key": api_key(), **kw.pop("headers", {})}
    r = requests.request(method, f"{API}{path}", headers=headers, timeout=120, **kw)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text[:300])
        except ValueError:
            detail = r.text[:300]
        raise SystemExit(f"ElevenLabs {method} {path} failed: HTTP {r.status_code}: {detail}")  # no key in message
    return r


# --------------------------------------------------------------------------- account checks


def subscription() -> dict:
    s = _request("GET", "/user/subscription").json()
    used, limit = s.get("character_count", 0), s.get("character_limit", 0)
    return {"tier": s.get("tier"), "status": s.get("status"), "used": used, "limit": limit,
            "remaining": max(0, limit - used), "resets_unix": s.get("next_character_count_reset_unix"),
            "can_extend_character_limit": s.get("can_extend_character_limit")}


def licence_note(tier: str | None) -> str:
    if (tier or "").lower() == "free":
        return ("FREE plan: output is licensed for non-commercial use only and must be attributed, with "
                "'elevenlabs.io' in the title of the published content (ElevenLabs help centre).")
    return f"Paid plan ({tier}): includes a commercial licence for generated audio (check your plan's terms)."


def choose_voice(name: str | None) -> dict:
    voices = _request("GET", "/voices").json().get("voices", [])
    stock = [v for v in voices if v.get("category") in ("premade", "default")]
    by_name = {v["name"].split(" ")[0].split("-")[0].strip(): v for v in stock}
    for want in ([name] if name else VOICE_PREFERENCE):
        if want in by_name:
            v = by_name[want]
            return {"voice_id": v["voice_id"], "name": v["name"], "labels": v.get("labels", {})}
    raise SystemExit(f"None of {name or VOICE_PREFERENCE} is available. Stock voices: {sorted(by_name)}")


def credits_needed(texts: list[str], model: str) -> int:
    per_char = 0.5 if ("flash" in model or "turbo" in model) else 1.0
    return int(sum(len(t) for t in texts) * per_char) + 1


def require_credits(texts, model):
    sub = subscription()
    need = credits_needed(texts, model)
    print(f"plan {sub['tier']} ({sub['status']}): {sub['remaining']:,} of {sub['limit']:,} credits left; "
          f"this request needs about {need:,}")
    print(licence_note(sub["tier"]))
    if need > sub["remaining"]:
        raise SystemExit("Not enough credits. Stopping without generating (no overage or purchase is triggered).")
    return sub


# --------------------------------------------------------------------------- synthesis


def tts(text, voice, model, speed, previous_text=None, next_text=None) -> tuple[bytes, dict]:
    body = {"text": text, "model_id": model,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "style": 0.0, "use_speaker_boost": True,
                               "speed": speed}}
    if previous_text:
        body["previous_text"] = previous_text
    if next_text:
        body["next_text"] = next_text
    r = _request("POST", f"/text-to-speech/{voice['voice_id']}/with-timestamps",
                 params={"output_format": "mp3_44100_128"}, json=body).json()
    return base64.b64decode(r["audio_base64"]), (r.get("alignment") or r.get("normalized_alignment") or {})


def transcribe(path: Path) -> str:
    with open(path, "rb") as f:
        r = _request("POST", "/speech-to-text", data={"model_id": "scribe_v1", "language_code": "en"},
                     files={"file": (path.name, f, "audio/mpeg")})
    return r.json().get("text", "")


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out)


def chunk_times(clip, alignment, dur) -> list[tuple[float, float]]:
    """Start/end time of each caption chunk, from ElevenLabs character alignment (or proportional fallback)."""
    text, spans = N.clip_text(clip)
    chars = alignment.get("characters") or []
    if len(chars) == len(text):
        st, en = alignment["character_start_times_seconds"], alignment["character_end_times_seconds"]
        out = []
        for a, b in spans:
            idx = [i for i in range(a, b) if not text[i].isspace()]
            out.append((st[idx[0]], en[idx[-1]]))
        return out
    total = len(text)
    return [(dur * a / total, dur * b / total) for a, b in spans]


def generate(backend: str, voice_name: str | None, speed: float, model: str) -> None:
    out_dir = OUT / backend
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = [N.clip_text(c)[0] for c in N.CLIPS]
    meta = {"backend": backend, "speed": speed, "model": model, "clips": []}
    if backend == "elevenlabs":
        voice = choose_voice(voice_name)
        todo = []
        for i, c in enumerate(N.CLIPS):
            key = _cache_key(voice["voice_id"], model, speed, texts, i)
            if not (out_dir / f"{c['id']}.{key}.mp3").exists():
                todo.append(texts[i])
        sub = require_credits(todo, model) if todo else subscription()
        meta.update({"voice": voice, "tier": sub["tier"], "licence": licence_note(sub["tier"])})
        print(f"voice: {voice['name']} ({voice['voice_id']}), model {model}, speed {speed}; {len(todo)} clips to generate")
    else:
        meta["voice"] = {"name": "macOS say: Samantha (timing stand-in only)"}
    for i, c in enumerate(N.CLIPS):
        if backend == "elevenlabs":
            key = _cache_key(voice["voice_id"], model, speed, texts, i)
            path = out_dir / f"{c['id']}.{key}.mp3"
            align_path = path.with_suffix(".align.json")
            if not path.exists():
                audio, align = tts(texts[i], voice, model, speed,
                                   previous_text=texts[i - 1] if i else None,
                                   next_text=texts[i + 1] if i + 1 < len(texts) else None)
                path.write_bytes(audio)
                align_path.write_text(json.dumps(align))
                print(f"  generated {path.name}")
            align = json.loads(align_path.read_text())
        else:
            path = out_dir / f"{c['id']}.aiff"
            subprocess.run(["say", "-v", "Samantha", "-r", str(round(165 * speed)), "-o", str(path), texts[i]], check=True)
            align = {}
        d = duration(path)
        meta["clips"].append({"id": c["id"], "file": str(path), "duration": d, "text": texts[i],
                              "chunks": chunk_times(c, align, d), "timed_by": "alignment" if align else "proportional"})
    (out_dir / "clips.json").write_text(json.dumps(meta, indent=2))
    tot = sum(c["duration"] for c in meta["clips"])
    print(f"{len(meta['clips'])} clips, {tot:.1f} s of speech -> {out_dir / 'clips.json'}")


def _cache_key(voice_id, model, speed, texts, i) -> str:
    ctx = json.dumps([voice_id, model, speed, texts[i], texts[i - 1] if i else "", texts[i + 1] if i + 1 < len(texts) else ""])
    return hashlib.sha256(ctx.encode()).hexdigest()[:10]


def sample(voice_name, speed, model) -> None:
    voice = choose_voice(voice_name)
    require_credits([N.SAMPLE_TEXT], model)
    audio, _ = tts(N.SAMPLE_TEXT, voice, model, speed)
    path = OUT / "sample" / f"sample_{voice['name'].split()[0]}_{speed}.mp3"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audio)
    d = duration(path)
    words = len(N.SAMPLE_TEXT.split())
    print(f"sample: {path} ({d:.2f} s, {60 * words / d:.0f} words/min) voice {voice['name']}")
    try:
        heard = transcribe(path)
    except SystemExit as e:  # optional check: never block on it
        print(f"speech-to-text unavailable, skipping the wording check ({e})")
        return
    print(f"speech-to-text: {heard}")
    norm = heard.lower().replace(",", "")
    for term in N.SAMPLE_EXPECT:
        t = term.lower().replace(",", "")
        alt = {"0.987": "nine eight seven", "0.913": "nine one three", "2362": "sixty-two", "90": "ninety",
               "cnn": "c n n", "pytorch": "py torch"}.get(t, t)
        print(f"  {'OK ' if (t in norm or alt in norm) else 'CHECK'} {term}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["check", "sample", "generate", "preview"])
    ap.add_argument("--voice")
    ap.add_argument("--speed", type=float, default=1.05)
    ap.add_argument("--model", default="eleven_multilingual_v2")
    a = ap.parse_args()
    if a.action == "check":
        sub = subscription()
        print(json.dumps({k: v for k, v in sub.items()}, indent=2))
        print(licence_note(sub["tier"]))
        v = choose_voice(a.voice)
        print(f"voice: {v['name']} {v['labels']}")
        texts = [N.clip_text(c)[0] for c in N.CLIPS]
        print(f"full narration needs about {credits_needed(texts, a.model):,} credits; sample about "
              f"{credits_needed([N.SAMPLE_TEXT], a.model):,}")
    elif a.action == "sample":
        sample(a.voice, a.speed, a.model)
    elif a.action == "generate":
        generate("elevenlabs", a.voice, a.speed, a.model)
    else:
        generate("preview", None, a.speed, a.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
