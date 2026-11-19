"""Record the Exoplanet Finder demo from the running app with headless Chrome (DevTools screencast).

Only the page is captured (no desktop, tabs or notifications). Real mouse events drive the app; a drawn cursor
overlay makes the pointer visible. Each interaction is recorded as a short motion clip; after the app has finished
rerunning and rendering, a full-resolution still is captured. build.py edits these into the video.

Usage:  python demo_video/record.py [--port 8650]
Output: demo_video/build/rec/<shot>/NNNNN.jpg + frames.json, demo_video/build/stills/<name>.png, build/shots.json
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import connect

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
VIEW_W, VIEW_H, DSF = 1440, 720, 2.0          # CSS viewport; stills are 2880x1440 device pixels
SCREENCAST_W, SCREENCAST_H = 1920, 960         # motion frames
GITHUB_URL = "https://github.com/mabsidcode/exoplanet-finder"

CURSOR_JS = r"""
(() => {
  if (window.__cur) return 'exists';
  const c = document.createElement('div');
  c.innerHTML = '<svg width="30" height="30" viewBox="0 0 28 28"><path d="M3 2 L3 22 L8.5 16.8 L12.3 25.5 L15.6 24 L11.9 15.5 L19.5 15.5 Z" fill="#ffffff" stroke="#0a0e1a" stroke-width="1.6" stroke-linejoin="round"/></svg>';
  Object.assign(c.style, {position: 'fixed', left: '0px', top: '0px', zIndex: 2147483647, pointerEvents: 'none',
                          filter: 'drop-shadow(0 2px 4px rgba(0,0,0,.6))', willChange: 'transform'});
  document.documentElement.appendChild(c);
  const ring = document.createElement('div');
  Object.assign(ring.style, {position: 'fixed', width: '36px', height: '36px', marginLeft: '-18px', marginTop: '-18px',
    borderRadius: '50%', border: '3px solid #8fb3ff', opacity: '0', zIndex: 2147483646, pointerEvents: 'none'});
  document.documentElement.appendChild(ring);
  let x = 1100, y = 560;
  const place = () => { c.style.transform = `translate(${x - 3}px, ${y - 2}px)`; };
  place();
  const ease = k => k < .5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
  window.__cur = {
    move: (tx, ty, ms) => new Promise(r => { const x0 = x, y0 = y, t0 = performance.now();
      const st = t => { const k = Math.min(1, (t - t0) / ms), e = ease(k); x = x0 + (tx - x0) * e; y = y0 + (ty - y0) * e; place();
        if (k < 1) requestAnimationFrame(st); else r([x, y]); }; requestAnimationFrame(st); }),
    ripple: () => new Promise(r => { ring.style.left = x + 'px'; ring.style.top = y + 'px'; const t0 = performance.now();
      const st = t => { const k = Math.min(1, (t - t0) / 450); ring.style.opacity = String(0.9 * (1 - k));
        ring.style.transform = `scale(${0.5 + 0.9 * k})`; if (k < 1) requestAnimationFrame(st); else r(); }; requestAnimationFrame(st); }),
    hide: () => { c.style.display = 'none'; }, pos: () => [x, y],
  };
  window.__scroller = () => {
    for (const s of ['[data-testid="stMain"]', '[data-testid="stAppViewContainer"]', 'section.main']) {
      const el = document.querySelector(s);
      if (el && el.scrollHeight > el.clientHeight + 5 && /(auto|scroll)/.test(getComputedStyle(el).overflowY)) return el;
    }
    return document.scrollingElement;
  };
  window.__scrollBy = (dy, ms) => new Promise(r => { const s = __scroller(), y0 = s.scrollTop, t0 = performance.now();
    const st = t => { const k = Math.min(1, (t - t0) / ms); s.scrollTop = y0 + dy * ease(k); if (k < 1) requestAnimationFrame(st); else r(s.scrollTop); };
    requestAnimationFrame(st); });
  window.__rect = (sel, text) => { const el = [...document.querySelectorAll(sel)].find(e => !text || e.innerText.toLowerCase().includes(text.toLowerCase()));
    if (!el) return null; const b = el.getBoundingClientRect(); return [b.x, b.y, b.width, b.height]; };
  return 'ok';
})()
"""


class CDP:
    def __init__(self, ws_url: str):
        self.ws = connect(ws_url, max_size=None)
        self.lock = threading.Lock()
        self.n = 0
        self.pending: dict[int, list] = {}
        self.recording: list | None = None
        threading.Thread(target=self._reader, daemon=True).start()

    def _send(self, method, params, wait=True):
        with self.lock:
            self.n += 1
            i = self.n
            ev = [threading.Event(), None]
            self.pending[i] = ev
            self.ws.send(json.dumps({"id": i, "method": method, "params": params}))
        if wait:
            if not ev[0].wait(120):
                raise TimeoutError(method)
            if "error" in ev[1]:
                raise RuntimeError(f"{method}: {ev[1]['error']}")
            return ev[1].get("result", {})
        return None

    def call(self, method, **params):
        return self._send(method, params)

    def _reader(self):
        for raw in self.ws:
            m = json.loads(raw)
            if "id" in m:
                ev = self.pending.pop(m["id"], None)
                if ev:
                    ev[1] = m
                    ev[0].set()
            elif m.get("method") == "Page.screencastFrame":
                p = m["params"]
                if self.recording is not None:
                    self.recording.append((p["metadata"]["timestamp"], p["data"]))
                self._send("Page.screencastFrameAck", {"sessionId": p["sessionId"]}, wait=False)

    def js(self, expr):
        r = self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(f"JS error: {r['exceptionDetails']}")
        return r.get("result", {}).get("value")


class Recorder:
    def __init__(self, cdp: CDP):
        self.c = cdp
        self.shots: dict[str, dict] = {}
        (BUILD / "rec").mkdir(parents=True, exist_ok=True)
        (BUILD / "stills").mkdir(parents=True, exist_ok=True)

    def start(self):
        self.c.recording = []
        self.t0 = time.time()

    def stop(self, name: str):
        frames, self.c.recording = self.c.recording, None
        d = BUILD / "rec" / name
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        meta = []
        for i, (ts, data) in enumerate(frames):
            (d / f"{i:05d}.jpg").write_bytes(base64.b64decode(data))
            meta.append({"file": f"{i:05d}.jpg", "t": ts})
        (d / "frames.json").write_text(json.dumps(meta))
        self.shots[name] = {"n_frames": len(meta), "duration": (meta[-1]["t"] - meta[0]["t"]) if meta else 0}
        print(f"  shot {name}: {len(meta)} frames over {self.shots[name]['duration']:.2f} s")

    def still(self, name: str, rects: dict | None = None):
        png = base64.b64decode(self.c.call("Page.captureScreenshot", format="png")["data"])
        (BUILD / "stills" / f"{name}.png").write_bytes(png)
        self.shots.setdefault("_stills", {})[name] = {"rects": rects or {}}
        print(f"  still {name} {rects or ''}")

    # ------------------------------------------------------------------ interactions
    def move(self, x, y, ms=800):
        self.c.js(f"__cur.move({x}, {y}, {ms})")

    def click(self):
        x, y = self.c.js("__cur.pos()")
        self.c.js("__cur.ripple(); 0")
        for t in ("mousePressed", "mouseReleased"):
            self.c.call("Input.dispatchMouseEvent", type=t, x=x, y=y, button="left", clickCount=1)

    def center(self, sel, text=None):
        r = self.c.js(f"__rect({json.dumps(sel)}, {json.dumps(text)})")
        if not r:
            print(self.c.js("document.body.innerHTML.length"), self.c.js(
                "[...document.querySelectorAll('[role=option],[role=listbox]')].map(e => e.outerHTML.slice(0, 200))"))
            raise RuntimeError(f"element not found: {sel} {text}")
        return r[0] + r[2] / 2, r[1] + r[3] / 2

    def rect(self, sel, text=None):
        return self.c.js(f"__rect({json.dumps(sel)}, {json.dumps(text)})")

    def scroll_to(self, sel, text, top_offset, ms=1100):
        r = self.rect(sel, text)
        self.c.js(f"__scrollBy({r[1] - top_offset}, {ms})")

    def wait_for(self, expr, timeout=90, settle=1.5):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.c.js(expr):
                time.sleep(settle)
                if self.c.js(expr):
                    return
            time.sleep(0.25)
        raise TimeoutError(expr)


APP_READY = """(() => {
  const big = document.querySelector('.card .big');
  const plots = document.querySelectorAll('.js-plotly-plot');
  const busy = document.querySelector('[data-testid="stSpinner"]') ||
               document.querySelector('[data-testid="stStatusWidget"] [data-testid="stStatusWidgetRunningIcon"]');
  return !!big && big.innerText.trim() === '%s' && plots.length > 0 && !busy;
})()"""

RESULTS_READY = """(() => {
  const m = [...document.querySelectorAll('[data-testid="stMetricValue"]')].map(e => e.innerText);
  const imgs = [...document.querySelectorAll('img')];
  return m.includes('0.913') && document.querySelectorAll('.js-plotly-plot').length > 0 &&
         imgs.length >= 4 && imgs.every(i => i.complete && i.naturalWidth > 0);
})()"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8650)
    args = ap.parse_args()
    app_url = f"http://localhost:{args.port}/"
    urllib.request.urlopen(app_url + "_stcore/health", timeout=10)
    profile = BUILD / "chrome-profile"
    shutil.rmtree(profile, ignore_errors=True)
    chrome = subprocess.Popen([CHROME, "--headless=new", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                               "--remote-debugging-port=9377", f"--user-data-dir={profile}", "--hide-scrollbars",
                               "--no-first-run", "--disable-extensions", "--mute-audio",
                               f"--window-size={VIEW_W},{VIEW_H}", "about:blank"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen("http://127.0.0.1:9377/json"))
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.1)
        c = CDP([t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"])
        rec = Recorder(c)
        c.call("Page.enable")
        c.call("Runtime.enable")
        c.call("Emulation.setDeviceMetricsOverride", width=VIEW_W, height=VIEW_H, deviceScaleFactor=DSF, mobile=False)
        c.call("Page.navigate", url=app_url)
        rec.wait_for(APP_READY % "0.987", settle=3)
        assert c.js("document.title").startswith("Exoplanet Finder")
        c.js(CURSOR_JS)
        webgl = c.js("!!document.createElement('canvas').getContext('webgl')")
        print("WebGL available:", webgl)
        assert webgl, "WebGL is required for the light-curve charts"
        c.call("Page.startScreencast", format="jpeg", quality=92, maxWidth=SCREENCAST_W, maxHeight=SCREENCAST_H,
               everyNthFrame=1)
        time.sleep(0.5)

        # ---- Scene 2: Kepler-90 i -------------------------------------------------------------
        rec.still("k90_initial")
        rec.start()
        time.sleep(0.4)
        x, y = rec.center('[data-testid="stSelectbox"] button[aria-label="Open"]')
        rec.move(x, y, 900)
        rec.click()
        time.sleep(0.7)
        ox, oy = rec.center('[role="option"]', "Kepler-90 i")
        rec.move(ox, oy, 750)
        time.sleep(0.15)
        rec.click()
        time.sleep(0.8)
        rec.move(1330, 520, 600)
        time.sleep(0.2)
        rec.stop("k90_select")
        rec.wait_for(APP_READY % "0.987")
        rec.still("k90_top", {"later": rec.rect(".card", "Later catalogues"), "catalogue": rec.rect(".card", "Catalogue information")})

        rec.start()
        rec.scroll_to('[role="tablist"]', None, 74, 1300)  # below Streamlit's fixed header
        time.sleep(0.4)
        rec.stop("k90_scroll_raw")
        time.sleep(1.0)
        rec.still("k90_raw", {"chart": rec.rect('[data-testid="stPlotlyChart"]')})

        for name, tab in (("k90_folded", "Folded transit"), ("k90_global", "Global view"), ("k90_local", "Local view")):
            rec.start()
            x, y = rec.center('[role="tab"]', tab)
            rec.move(x, y, 750)
            time.sleep(0.1)
            rec.click()
            time.sleep(1.2)
            rec.move(1395, 300, 600)
            time.sleep(0.3)
            rec.stop(name)
            time.sleep(1.0)
            sel = c.js("(document.querySelector('[role=tab][aria-selected=true]') || {}).innerText")
            assert sel and tab in sel, f"tab switch failed: selected={sel!r}, wanted {tab!r}"
            vis = '[role="tabpanel"]:not([hidden]) [data-testid="stPlotlyChart"]'
            rec.still(name, {"chart": rec.rect(vis)})

        # ---- Scene 4: prediction ------------------------------------------------------------------
        rec.start()
        rec.scroll_to(".card:has(.tag.model)", None, 96, 1300)
        sx, sy = rec.center(".card .big")
        rec.move(1400, sy + 10, 800)  # park in the margin, off the text
        time.sleep(0.3)
        rec.stop("k90_score")
        rec.still("k90_score", {"model": rec.rect(".card:has(.tag.model)"), "big": rec.rect(".card .big")})

        # ---- Scene 5: catalogue disagreement ---------------------------------------------------------
        rec.start()
        r = rec.rect('[data-testid="stSelectbox"]')
        if r[1] < 70:
            c.js(f"__scrollBy({r[1] - 150}, 900)")
        x, y = rec.center('[data-testid="stSelectbox"] button[aria-label="Open"]')
        rec.move(x, y, 850)
        rec.click()
        time.sleep(0.7)
        ox, oy = rec.center('[role="option"]', "KIC 10587105")
        rec.move(ox, oy, 750)
        time.sleep(0.15)
        rec.click()
        time.sleep(0.5)
        rec.stop("afp_select")
        rec.wait_for(APP_READY % "0.999")
        rec.move(1330, 560, 10)
        time.sleep(0.3)
        rec.still("afp_top", {"model": rec.rect(".card:has(.tag.model)"),
                              "catalogue": rec.rect(".card", "Catalogue information")})

        rec.start()
        rec.scroll_to(".card", "Later catalogues", 72, 1400)
        time.sleep(0.4)
        rec.stop("afp_later")
        time.sleep(0.8)
        rec.still("afp_later", {"later": rec.rect(".card", "Later catalogues")})

        # ---- Scene 6: evaluation -------------------------------------------------------------------
        rec.start()
        x, y = rec.center('[data-testid="stSidebarNav"] a', "Measured results")
        rec.move(x, y, 1000)
        time.sleep(0.15)
        rec.click()
        time.sleep(0.6)
        rec.stop("results_nav")
        rec.wait_for(RESULTS_READY, settle=2.5)
        c.js("__scroller().scrollTop = 0; 0")
        rec.move(1330, 600, 10)
        time.sleep(0.6)
        rec.still("results_top", {"metrics": rec.rect('[data-testid="stHorizontalBlock"]'),
                                  "table": rec.rect('[data-testid="stDataFrame"]')})

        rec.start()
        rec.scroll_to('[data-testid="stPlotlyChart"]', None, 76, 1400)
        time.sleep(0.4)
        rec.stop("results_pr")
        time.sleep(1.0)
        rec.still("results_pr", {"chart": rec.rect('[data-testid="stPlotlyChart"]')})
        c.call("Page.stopScreencast")

        # ---- Scene 7: the public GitHub repository -------------------------------------------------
        c.call("Page.navigate", url=GITHUB_URL)
        rec.wait_for("document.readyState === 'complete' && !!document.querySelector('article.markdown-body')",
                     timeout=60, settle=2.5)
        rec.still("github_top")
        c.js("window.scrollTo(0, document.querySelector('article.markdown-body').getBoundingClientRect().top + "
             "window.scrollY - 70); 0")
        time.sleep(1.5)
        rec.still("github_readme")

        (BUILD / "shots.json").write_text(json.dumps(rec.shots, indent=2))
        print("done")
        return 0
    finally:
        chrome.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
