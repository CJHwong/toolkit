# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pywebview==6.2.1",
#     "pyobjc-core==12.1 ; sys_platform == 'darwin'",
#     "pyobjc-framework-Cocoa==12.1 ; sys_platform == 'darwin'",
#     "pyobjc-framework-WebKit==12.1 ; sys_platform == 'darwin'",
#     "transformers==5.4.0",
#     "mlx-audio==0.4.2",
#     "click==8.3.1",
#     "numpy==2.4.4",
#     "soundfile==0.13.1",
#     "librosa==0.11.0",
# ]
# ///
"""
Auto Dub GUI — pywebview frontend for auto_dub.py.

Runs as a native macOS app: the WebKit view and the Python pipeline share
one process. JS calls Python via pywebview's js_api bridge; Python pushes
progress back to JS via window.evaluate_js. No external server, no IPC —
the tiny localhost HTTP server embedded here exists solely to stream
media files (videos, preview clips) to <video>/<audio> elements with
Range-request support. file:// would not seek large MP4s reliably.

Launch:
    uv run python/auto_dub_gui.py

Requirements match auto_dub.py (ffmpeg, ffprobe, uv, optional claude).
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

# ─── Self-bootstrap ───────────────────────────────────────────────
# Enable the one-liner form:
#   uv run https://raw.githubusercontent.com/CJHwong/toolkit/<ref>/python/auto_dub_gui.py
# uv only fetches THIS script, so the sibling `auto_dub` module and the
# `auto_dub_web/` asset tree aren't on disk. If that's the case, fetch
# them from the same repo ref into ~/.cache/auto-dub/src-<ref>/ once and
# point the rest of the code at that dir. Running from a clone is still
# detected (sibling files present) and uses the clone directly.
_SRC_REPO = os.environ.get("AUTO_DUB_REPO", "CJHwong/toolkit")
_SRC_REF = os.environ.get("AUTO_DUB_REF", "main")
_SRC_ASSETS = [
    "auto_dub.py",
    "auto_dub_web/index.html",
    "auto_dub_web/styles.css",
    "auto_dub_web/overrides.css",
    "auto_dub_web/data.jsx",
    "auto_dub_web/icons.jsx",
    "auto_dub_web/player.jsx",
    "auto_dub_web/side-panel.jsx",
    "auto_dub_web/app.jsx",
]


def _fetch_asset(url: str, dst: Path, pinned: bool) -> None:
    """Populate `dst` from `url`, using an ETag sidecar for cheap freshness
    checks on mutable refs. When `pinned` (a SHA), the file is immutable,
    so skip the network as soon as we have a local copy.
    """
    etag_file = dst.with_suffix(dst.suffix + ".etag")
    if dst.exists() and pinned:
        return

    req = urllib.request.Request(url)
    if dst.exists() and etag_file.exists():
        req.add_header("If-None-Match", etag_file.read_text().strip())

    try:
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
            etag = resp.headers.get("ETag", "")
        # Only announce actual downloads — 304 bypasses this (raises below).
        print(f"[auto-dub] Fetched {dst.name}", file=sys.stderr, flush=True)
        dst.write_bytes(data)
        if etag:
            etag_file.write_text(etag)
    except urllib.error.HTTPError as e:
        if e.code == 304:
            return  # cached copy is current
        if dst.exists():
            return  # serve stale rather than blow up mid-launch
        raise
    except urllib.error.URLError:
        if dst.exists():
            return  # offline: serve stale
        raise


def _bootstrap_src_dir() -> Path:
    """Resolve a directory that contains auto_dub.py + auto_dub_web/.

    Preference order:
      1. $AUTO_DUB_SRC — explicit override (useful for local dev).
      2. The script's own parent, if it already contains the assets
         (e.g. running from a clone).
      3. ~/.cache/auto-dub/src-<ref>/ — cache keyed by ref. SHA refs
         are treated as immutable; branch refs re-check with ETag each
         launch so `main` stays current.
    """
    override = os.environ.get("AUTO_DUB_SRC")
    if override:
        p = Path(override).expanduser().resolve()
        if (p / "auto_dub.py").is_file() and (p / "auto_dub_web").is_dir():
            return p

    here = Path(__file__).resolve().parent
    if (here / "auto_dub.py").is_file() and (here / "auto_dub_web").is_dir():
        return here

    pinned = bool(re.fullmatch(r"[0-9a-f]{7,40}", _SRC_REF))
    cache = Path.home() / ".cache" / "auto-dub" / f"src-{_SRC_REF}"
    (cache / "auto_dub_web").mkdir(parents=True, exist_ok=True)
    for rel in _SRC_ASSETS:
        url = f"https://raw.githubusercontent.com/{_SRC_REPO}/{_SRC_REF}/python/{rel}"
        _fetch_asset(url, cache / rel, pinned)
    return cache


_SRC_DIR = _bootstrap_src_dir()
sys.path.insert(0, str(_SRC_DIR))

import webview

import auto_dub  # noqa: E402
from auto_dub import Callbacks, TranscriptPrep  # noqa: E402

APP_ROOT = _SRC_DIR
WEB_ROOT = _SRC_DIR / "auto_dub_web"
RECENTS_FILE = Path.home() / ".cache" / "auto-dub" / "recents.json"
RECENTS_MAX = 10

# Babel-standalone fetches .jsx files at runtime; serve as JavaScript so
# WebKit doesn't complain about an unknown MIME type.
mimetypes.add_type("application/javascript", ".jsx")
mimetypes.add_type("application/wasm", ".wasm")

# ─── Embedded media server ─────────────────────────────────────────
# Authorized absolute paths. Only these can be fetched via GET /media.
_ALLOWED_MEDIA: set[str] = set()
_ALLOWED_LOCK = threading.Lock()


def _register_media_path(p: Path) -> str:
    """Whitelist a path for /media and return the URL to hand to the UI.

    The `v` param is a cache-buster: when the same path gets regenerated
    (regen → re-mux produces a new MP4 at the same location), WKWebView
    would otherwise keep serving the stale cached bytes.
    """
    sp = str(Path(p).resolve())
    with _ALLOWED_LOCK:
        _ALLOWED_MEDIA.add(sp)
    return f"/media?path={quote(sp)}&v={int(time.time() * 1000)}"


def _is_allowed(sp: str) -> bool:
    with _ALLOWED_LOCK:
        return sp in _ALLOWED_MEDIA


class _Handler(BaseHTTPRequestHandler):
    """Serves static UI from WEB_ROOT and whitelisted media with Range support."""

    # Silence default access-log spam on stderr
    def log_message(self, *args, **kwargs) -> None:  # noqa: ARG002
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/media":
            self._serve_media(parse_qs(parsed.query))
            return

        # Default: static UI asset
        rel = parsed.path.lstrip("/") or "index.html"
        if rel.endswith("/"):
            rel += "index.html"
        candidate = (WEB_ROOT / rel).resolve()
        # Path-traversal guard
        if not str(candidate).startswith(str(WEB_ROOT.resolve())):
            self.send_error(403)
            return
        if not candidate.is_file():
            self.send_error(404)
            return
        self._serve_file(candidate)

    def _serve_media(self, query: dict[str, list[str]]) -> None:
        path = (query.get("path") or [""])[0]
        if not path or not _is_allowed(path):
            self.send_error(403, "Path not authorized")
            return
        p = Path(path)
        if not p.is_file():
            self.send_error(404)
            return
        self._serve_file(p)

    def _serve_file(self, path: Path) -> None:
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "application/octet-stream"
        size = path.stat().st_size

        range_header = self.headers.get("Range")
        if range_header:
            m = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if not m:
                self.send_error(416, "Invalid Range")
                return
            start = int(m.group(1)) if m.group(1) else 0
            end_s = m.group(2)
            end = int(end_s) if end_s else size - 1
            end = min(end, size - 1)
            if start > end:
                self.send_error(416, "Invalid Range")
                return
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(length))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with path.open("rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    remaining -= len(chunk)
            return

        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with path.open("rb") as f:
            while True:
                chunk = f.read(64 * 1024)
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return


def _start_server() -> int:
    """Boot the internal HTTP server on 127.0.0.1 with an OS-chosen port."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, name="auto_dub_http", daemon=True)
    thread.start()
    return port


# ─── Recents ──────────────────────────────────────────────────────
def _load_recents() -> list[dict]:
    if not RECENTS_FILE.exists():
        return []
    try:
        data = json.loads(RECENTS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict) and "path" in r]
    except (json.JSONDecodeError, OSError):
        pass
    return []


def _save_recents(recents: list[dict]) -> None:
    try:
        RECENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        RECENTS_FILE.write_text(json.dumps(recents, indent=2), encoding="utf-8")
    except OSError:
        pass


def _bump_recent(recents: list[dict], entry: dict) -> list[dict]:
    out = [entry] + [r for r in recents if r.get("path") != entry.get("path")]
    return out[:RECENTS_MAX]


# ─── Friendly error messages ─────────────────────────────────────
def _friendly_error(e: Exception) -> str:
    """Translate low-level subprocess/OS errors into something the banner
    can show without leaking tracebacks or raw CalledProcessError reprs.
    """
    if isinstance(e, FileNotFoundError):
        tool = e.filename or ""
        if isinstance(tool, bytes):
            tool = tool.decode("utf-8", errors="replace")
        # subprocess raises with filename set to the missing exe
        short = Path(tool).name if tool else "command"
        hint = auto_dub.INSTALL_HINTS.get(short, "")
        tail = f" · Install: {hint}" if hint else ""
        return f"Required tool not found: {short}{tail}"
    if isinstance(e, subprocess.CalledProcessError):
        buf = e.stderr or e.stdout or b""
        if isinstance(buf, bytes):
            buf = buf.decode("utf-8", errors="replace")
        lines = [ln for ln in buf.strip().splitlines() if ln.strip()]
        detail = lines[-1] if lines else f"exit code {e.returncode}"
        cmd_name = Path(e.cmd[0]).name if e.cmd else "command"
        return f"{cmd_name} failed: {detail}"
    return str(e) or type(e).__name__


# ─── GUI-flavored pipeline callbacks ──────────────────────────────
def _push_event(window, evt: dict) -> None:
    """Send a single event to the JS side. Shared by WebCallbacks and Api
    so there's exactly one place that knows the JS bridge contract."""
    if window is None:
        return
    try:
        window.evaluate_js(
            f"window.__autoDubEvent && window.__autoDubEvent({json.dumps(evt)})"
        )
    except Exception:
        # evaluate_js can fail if the window is mid-teardown; ignore.
        pass


class WebCallbacks(Callbacks):
    """Pushes pipeline events to the JS side via window.evaluate_js."""

    def __init__(self, window, cancel_event: threading.Event):
        self._window = window
        self._cancel_event = cancel_event

    def log(self, level: str, msg: str) -> None:
        _push_event(self._window, {
            "type": "log", "level": level, "msg": msg,
            "t": time.strftime("%H:%M:%S"),
        })

    def stage(self, key: str) -> None:
        _push_event(self._window, {"type": "stage", "key": key})

    def segment(self, idx: int, status: str) -> None:
        _push_event(self._window, {"type": "segment", "idx": idx, "status": status})

    def cancelled(self) -> bool:
        return self._cancel_event.is_set()


# ─── JS-facing API ────────────────────────────────────────────────
def _segments_for_ui(segments: list[dict]) -> list[dict]:
    """Convert pipeline segment dicts ({index,...}) to the UI shape ({i,...})."""
    return [
        {"i": s["index"], "start": s["start"], "end": s["end"], "text": s["text"]}
        for s in segments
    ]


class Api:
    """Methods on this class are reachable from JS as window.pywebview.api.*.

    Names with a leading underscore are private to Python; pywebview won't
    expose them. Long-running operations dispatch to a worker thread and
    stream progress back via WebCallbacks / _push.
    """

    def __init__(self) -> None:
        self._window = None
        self._prep: TranscriptPrep | None = None
        self._video_path: Path | None = None
        self._lock = threading.Lock()
        self._busy = False
        self._cancel = threading.Event()
        self._recents = _load_recents()
        self._ref_audio: Path | None = None
        # Reference transcript is mode-specific: the auto-extracted clip
        # and the user-uploaded clip are two different audios and must
        # not share text. Keyed by voice mode ('auto' | 'upload').
        self._ref_text: dict[str, str] = {"auto": "", "upload": ""}
        # TTS model is heavy (~5–20s load). Cache across run_dub and
        # regenerate_segment calls; evict when the user flips 0.6B / 1.7B.
        self._tts_model = None
        self._tts_model_small: bool | None = None

    def _set_window(self, window) -> None:
        self._window = window

    def _start_worker(self, target) -> None:
        threading.Thread(target=target, daemon=True).start()

    def _push(self, evt: dict) -> None:
        _push_event(self._window, evt)

    def _begin(self) -> WebCallbacks:
        """Acquire busy flag + fresh cancel event. Raises if already running."""
        with self._lock:
            if self._busy:
                raise RuntimeError("A pipeline step is already running.")
            self._busy = True
            self._cancel.clear()
        return WebCallbacks(self._window, self._cancel)

    def _end(self) -> None:
        with self._lock:
            self._busy = False

    def _run_async(self, phase: str, body, *, error_extra: dict | None = None) -> dict:
        """Run `body(cbs)` in a worker thread with the shared try/except/finally
        plumbing: cooperative cancel → 'cancelled' event; any other exception
        → traceback + friendly 'error' event; always _end() afterward.
        """
        cbs = self._begin()

        def work() -> None:
            try:
                body(cbs)
            except auto_dub._Cancelled:
                self._push({"type": "cancelled"})
            except Exception as e:
                traceback.print_exc()
                evt = {"type": "error", "phase": phase, "msg": _friendly_error(e)}
                if error_extra:
                    evt.update(error_extra)
                self._push(evt)
            finally:
                self._end()

        self._start_worker(work)
        return {"ok": True}

    def _emit_transcript_ready(self, prep: TranscriptPrep, *, from_cache: bool) -> None:
        """Emit the transcript_ready event from whichever path built `prep`.

        Both probe_cache (fast) and the transcribe worker (slow, whisper +
        claude) need to push the same shape to the UI; centralizing avoids
        silent drift in the payload.
        """
        ref_clip_url = (
            _register_media_path(prep.ref_audio_path)
            if prep.ref_audio_path.exists() else None
        )
        self._push({
            "type": "transcript_ready",
            "segments": _segments_for_ui(prep.segments),
            "original": {s["index"]: s["text"] for s in prep.original_segments},
            "output_name": prep.output_path.name,
            "ref": {
                "index": prep.ref_segment.get("index"),
                "start": prep.ref_segment.get("start", 0.0),
                "end": prep.ref_segment.get("end", 0.0),
                "text": prep.ref_text,
                "url": ref_clip_url,
            },
            "language": prep.language,
            "target_lang": prep.target_lang,
            "from_cache": from_cache,
        })

    def _emit_dub_done(self, prep: TranscriptPrep, *, from_cache: bool = False) -> None:
        self._push({
            "type": "done",
            "output": str(prep.output_path),
            "name": prep.output_path.name,
            "url": _register_media_path(prep.output_path),
            "from_cache": from_cache,
        })

    def _get_tts_model(self, small: bool, cbs: Callbacks):
        if self._tts_model is not None and self._tts_model_small == small:
            cbs.log("info", f"Reusing cached {'0.6B' if small else '1.7B'} model")
            return self._tts_model
        # Drop the previous model first so its memory is reclaimed before
        # the new one is loaded (1.7B + 0.6B together won't fit comfortably).
        self._tts_model = None
        self._tts_model_small = None
        model = auto_dub.load_tts_model(small, cbs)
        self._tts_model = model
        self._tts_model_small = small
        return model

    # ── Sync methods ──

    def get_recents(self) -> list[dict]:
        return self._recents

    def check_environment(self, needs_claude: bool = True) -> dict:
        """Return {ok, missing, hints} for external tools. The UI calls
        this on mount and after toggling Translate (claude is only needed
        when translating)."""
        return auto_dub.check_environment(needs_claude=bool(needs_claude))

    def pick_file(self) -> dict | None:
        if self._window is None:
            return None
        result = self._window.create_file_dialog(
            webview.FileDialog.OPEN,
            allow_multiple=False,
            file_types=(
                "Video files (*.mp4;*.mov;*.mkv;*.m4v;*.webm)",
                "Audio files (*.wav;*.m4a;*.mp3;*.flac)",
                "All files (*.*)",
            ),
        )
        if not result:
            return None
        return self.open_file(result[0])

    def open_file(self, path: str) -> dict:
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        meta = auto_dub.probe_video(p)
        meta["url"] = _register_media_path(p)
        self._video_path = p
        self._prep = None
        self._ref_audio = None
        self._ref_text = {"auto": "", "upload": ""}

        self._recents = _bump_recent(self._recents, {
            "name": p.name,
            "path": str(p),
            "meta": time.strftime("Opened %Y-%m-%d"),
        })
        _save_recents(self._recents)
        return meta

    def pick_ref_audio(self) -> dict | None:
        if self._window is None:
            return None
        result = self._window.create_file_dialog(
            webview.FileDialog.OPEN,
            allow_multiple=False,
            file_types=(
                "Audio files (*.wav;*.mp3;*.m4a;*.flac;*.ogg)",
                "All files (*.*)",
            ),
        )
        if not result:
            return None
        p = Path(result[0])
        self._ref_audio = p
        return {"name": p.name, "path": str(p)}

    def set_ref_text(self, text: str, mode: str = "auto") -> None:
        """Store ref transcript per voice mode.

        If a prep already exists and matches this mode, also propagate the
        edit into prep.ref_text so subsequent run_dub / regen use it; and
        for auto mode, rewrite the ref_clip.txt sidecar so try_load_from_cache
        on a later session doesn't see stale text.
        """
        mode = mode if mode in ("auto", "upload") else "auto"
        self._ref_text[mode] = text or ""
        if self._prep is not None:
            prep_is_upload = self._prep.voice_name != "original"
            if prep_is_upload == (mode == "upload"):
                self._prep.ref_text = text or ""
                if mode == "auto":
                    sidecar = self._prep.ref_audio_path.with_suffix(".txt")
                    try:
                        sidecar.write_text(text or "", encoding="utf-8")
                    except OSError:
                        pass

    def set_ref_mode_auto(self) -> None:
        """Switch back to auto-extracted ref voice. Clears any uploaded clip."""
        self._ref_audio = None

    def reveal_in_finder(self, path: str) -> None:
        if path:
            subprocess.Popen(["open", "-R", str(path)])

    def open_path(self, path: str) -> None:
        if path:
            subprocess.Popen(["open", str(path)])

    def cancel(self) -> None:
        # Cooperative cancel: the worker checks this flag between segments
        # and between subprocess stderr lines. Mid-TTS of a segment the
        # check won't fire for a few seconds; push a log line so the user
        # sees that the request was received.
        if self._cancel.is_set():
            return
        self._cancel.set()
        self._push({
            "type": "log",
            "level": "warn",
            "msg": "Cancellation requested — waiting for current step to finish…",
            "t": time.strftime("%H:%M:%S"),
        })
        self._push({"type": "cancelling"})

    def reset(self) -> None:
        self._cancel.set()
        self._prep = None
        self._video_path = None

    # ── Async methods (return immediately; progress via events) ──

    def probe_cache(self, opts: dict) -> dict:
        """Try to load an existing transcript from disk for the current
        language/voice config. No whisper/claude calls. If cache exists,
        emits the same `transcript_ready` event as a full transcribe run
        (with from_cache=True) so the UI can reuse the same handler.
        """
        if not self._video_path:
            return {"cached": False}
        use_ref_upload = bool(opts.get("use_ref_upload"))
        prep = auto_dub.try_load_from_cache(
            video=self._video_path,
            language=opts.get("language", "auto"),
            target_lang=opts.get("target_lang"),
            ref_audio=self._ref_audio if use_ref_upload else None,
            ref_text=self._ref_text["upload"] if use_ref_upload else None,
            small=bool(opts.get("small")),
        )
        if prep is None:
            return {"cached": False}

        self._prep = prep
        self._emit_transcript_ready(prep, from_cache=True)

        # If a full dub already lives on disk, restore the player to its
        # post-dub state: segments marked rendered, Dubbed track toggle
        # available, RunBar showing Complete.
        if prep.output_path.exists():
            for seg in prep.segments:
                if (prep.segments_dir / f"{seg['index']:03d}.wav").exists():
                    self._push({"type": "segment", "idx": seg["index"], "status": "done"})
            self._emit_dub_done(prep, from_cache=True)
        return {"cached": True}

    def transcribe(self, opts: dict) -> dict:
        """Kick off prepare_transcript() in a worker thread.

        opts = {language, target_lang|None, small, use_ref_upload: bool}
        Events emitted: log, stage, segment, then transcript_ready or error.
        """
        if not self._video_path:
            raise RuntimeError("No file opened.")
        video_path = self._video_path
        ref_audio = self._ref_audio if opts.get("use_ref_upload") else None
        ref_text = self._ref_text["upload"] if opts.get("use_ref_upload") else None
        language = opts.get("language") or "auto"
        target_lang = opts.get("target_lang")
        small = bool(opts.get("small"))

        def body(cbs: Callbacks) -> None:
            prep = auto_dub.prepare_transcript(
                video=video_path, ref_audio=ref_audio, ref_text=ref_text,
                language=language, target_lang=target_lang, small=small, cbs=cbs,
            )
            self._prep = prep
            self._emit_transcript_ready(prep, from_cache=False)

        return self._run_async("transcribe", body)

    def edit_segment(self, idx: int, patch: dict) -> dict:
        """Mutate the in-memory segment, invalidate its TTS cache, and
        persist the change back to the on-disk SRT so a later probe_cache
        load (new app session, etc.) sees the same text.
        """
        if not self._prep:
            return {"ok": False, "reason": "no_transcript"}
        prep = self._prep
        for seg in prep.segments:
            if seg["index"] == idx:
                if "text" in patch:
                    seg["text"] = str(patch["text"])
                if "start" in patch:
                    seg["start"] = max(0.0, float(patch["start"]))
                if "end" in patch:
                    seg["end"] = max(seg["start"] + 0.1, float(patch["end"]))
                auto_dub.invalidate_segment_cache(prep.segments_dir, [idx])
                active_srt = prep.translated_srt if prep.target_lang else prep.srt_file
                if active_srt is not None:
                    try:
                        auto_dub.write_srt(prep.segments, active_srt)
                    except OSError:
                        pass  # disk write failure is non-fatal
                return {"ok": True, "seg": _segments_for_ui([seg])[0]}
        return {"ok": False, "reason": "not_found"}

    def transcribe_ref_audio(self, language: str | None = None, mode: str = "auto") -> dict:
        """Run Whisper on the current reference clip and push the text back.

        Works in both modes:
          - Upload mode: transcribes the user-picked file (self._ref_audio).
          - Auto mode: transcribes the clip auto-extracted during
            prepare_transcript (prep.ref_audio_path).

        The resulting text is written to the per-mode slot and into the
        active prep (so subsequent run_dub picks it up).
        """
        slot = "upload" if mode == "upload" else "auto"
        if slot == "upload":
            if self._ref_audio is None:
                raise RuntimeError("No reference audio available yet.")
            audio = self._ref_audio
        else:
            if self._prep is None or not self._prep.ref_audio_path.exists():
                raise RuntimeError("No reference audio available yet.")
            audio = self._prep.ref_audio_path

        lang_canonical = auto_dub.resolve_language(language) if language else "auto"
        whisper_lang = auto_dub.to_whisper_lang(lang_canonical)

        def body(cbs: Callbacks) -> None:
            text = auto_dub.transcribe_audio_to_text(audio, whisper_lang, cbs)
            self.set_ref_text(text, slot)
            self._push({"type": "ref_transcribed", "text": text, "mode": slot})

        return self._run_async("ref_transcribe", body)

    def regenerate_segment(self, idx: int) -> dict:
        """Re-TTS a single segment. Emits segment events + final 'segment_ready'.
        Also re-muxes the output video if every segment is already rendered,
        so the preview element reflects the new voice without a full Dub.
        """
        if not self._prep:
            raise RuntimeError("No transcript yet.")
        prep = self._prep

        def body(cbs: Callbacks) -> None:
            model = self._get_tts_model(prep.small, cbs)
            auto_dub.regenerate_segment(prep, idx, prep.segments, cbs, model=model)
            clip = prep.segments_dir / f"{idx:03d}.wav"
            url = _register_media_path(clip) if clip.exists() else None
            self._push({"type": "segment_ready", "idx": idx, "url": url})

            all_rendered = all(
                (prep.segments_dir / f"{s['index']:03d}.wav").exists()
                for s in prep.segments
            )
            if all_rendered and prep.output_path.exists():
                cbs.log("info", "Refreshing dubbed video with regenerated segment...")
                auto_dub.finalize_dub(prep, prep.segments, cbs)
                self._emit_dub_done(prep)

        return self._run_async("regenerate", body, error_extra={"idx": idx})

    def run_dub(self, opts: dict | None = None) -> dict:
        """Run stages 3+4. Uses whatever is currently in prep.segments (edits included).

        opts may include `small` to override the model-size choice made at
        transcribe time — the user may have flipped the toggle between
        Transcribe and Dub and expects that to take effect.
        """
        if not self._prep:
            raise RuntimeError("No transcript yet.")
        if opts and "small" in opts:
            self._prep.small = bool(opts["small"])
        prep = self._prep

        def body(cbs: Callbacks) -> None:
            model = self._get_tts_model(prep.small, cbs)
            auto_dub.run_dub(prep, prep.segments, cbs, model=model)
            self._emit_dub_done(prep)

        return self._run_async("dub", body)


# ─── Signal handling ──────────────────────────────────────────────
def _install_signal_handlers() -> None:
    """Make Ctrl+C exit cleanly instead of crashing the Cocoa event loop.

    Python's default SIGINT handler raises KeyboardInterrupt, but when
    the main thread is parked inside NSApp.run() the exception can't
    unwind the Cocoa stack, so macOS shows a "Python quit unexpectedly"
    crash dialog. The fix is a watchdog thread blocked on a pipe fd
    written-to by set_wakeup_fd (the pipe byte is emitted by the C signal
    trampoline, which IS async-signal-safe). The watchdog then calls
    os._exit, skipping Python shutdown entirely — safe here because our
    only long-lived resources are daemon threads and an mmap'd MLX model
    the OS will reap on exit.
    """
    # No-op handlers — the real work happens in the watchdog below.
    signal.signal(signal.SIGINT, lambda *_: None)
    signal.signal(signal.SIGTERM, lambda *_: None)

    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    signal.set_wakeup_fd(write_fd)

    def watchdog() -> None:
        try:
            while True:
                data = os.read(read_fd, 1)
                if data:
                    print("\nAuto Dub shutting down.", flush=True)
                    os._exit(0)
        except OSError:
            pass

    threading.Thread(target=watchdog, name="auto_dub_sigint", daemon=True).start()


# ─── Entry ────────────────────────────────────────────────────────
def main() -> None:
    if not WEB_ROOT.is_dir():
        raise SystemExit(f"Missing UI assets at {WEB_ROOT}")

    _install_signal_handlers()
    port = _start_server()
    api = Api()
    window = webview.create_window(
        title="Auto Dub",
        url=f"http://127.0.0.1:{port}/",
        js_api=api,
        width=1280,
        height=900,
        min_size=(1080, 820),
    )
    api._set_window(window)

    # Quit paths that DON'T come through SIGINT: ⌘Q, the window's red
    # close button, macOS logout. These route through
    # NSApplication.terminate: → libc exit() → C++ static destructors.
    # libmlx.dylib's CompilerCache destructor dereferences Python tuples
    # from a Python interpreter that's already tearing down, SIGSEGVing
    # in tupledealloc ("Python quit unexpectedly"). Skip that whole path
    # by hard-exiting the moment the window closes — we have no state
    # that needs graceful shutdown (HTTP server is a daemon thread, TTS
    # model is just mmap'd files the OS reaps).
    def _hard_exit(*_):
        os._exit(0)
    try:
        window.events.closed += _hard_exit
    except Exception:
        # Older pywebview event APIs exposed .closed as a plain list.
        try:
            window.events.closed.append(_hard_exit)
        except Exception:
            pass

    try:
        webview.start()
    finally:
        _hard_exit()


if __name__ == "__main__":
    main()
