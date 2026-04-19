# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "transformers==5.4.0",
#     "mlx-audio==0.4.2",
#     "click==8.3.1",
#     "numpy==2.4.4",
#     "soundfile==0.13.1",
#     "librosa==0.11.0",
# ]
# ///
"""
Auto Dub - Automatically dub videos using Whisper + Qwen3-TTS.

Pipeline:
    1. Extract audio from video (ffmpeg)
    2. Transcribe to SRT (whisper.py)
    3. (Optional) Translate SRT via claude -p
    4. Generate TTS per segment (Qwen3-TTS, model loaded once)
    5. Combine TTS clips and mux with silent video (ffmpeg)

USAGE:
    # Auto-extract voice from video, translate to English
    uv run auto_dub.py -l Chinese --target-lang English video.mp4

    # Use external reference voice
    uv run auto_dub.py -r voice.wav -t "ref text" -l Chinese --target-lang English video.mp4

    # Re-voice with external voice, same language
    uv run auto_dub.py -r voice.wav -t "ref text" -l Chinese video.mp4

    # Use a manually edited SRT
    uv run auto_dub.py --srt edited.srt -l Chinese video.mp4

REQUIREMENTS:
    - ffmpeg on PATH
    - uv on PATH (runs whisper.py from GitHub for transcription)
    - claude CLI on PATH (only for --target-lang translation)

LIBRARY USAGE:
    The pipeline can also be driven programmatically (e.g. from a GUI).
    Pass a Callbacks subclass to receive log/stage/segment events and to
    signal cooperative cancellation:

        from auto_dub import Callbacks, prepare_transcript, run_dub

        class MyCallbacks(Callbacks):
            def log(self, level, msg): ...
            def stage(self, key): ...
            def segment(self, idx, status): ...

        prep = prepare_transcript(video_path, language='chinese',
                                  target_lang='english', cbs=MyCallbacks())
        out = run_dub(prep, cbs=MyCallbacks())
"""
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import click
import numpy as np
import soundfile as sf

SAMPLE_RATE = 24000  # Qwen3-TTS output sample rate
REF_CLIP_TARGET_SECONDS = 10  # target duration for auto-extracted reference clip
WHISPER_SCRIPT = "https://raw.githubusercontent.com/CJHwong/toolkit/main/python/whisper.py"

# Supported languages: Qwen3-TTS name -> whisper code
# Qwen3-TTS supports: auto, chinese, english, french, german, italian,
#                      japanese, korean, portuguese, russian, spanish
SUPPORTED_LANGUAGES = {
    "auto":       None,
    "chinese":    "zh",
    "english":    "en",
    "french":     "fr",
    "german":     "de",
    "italian":    "it",
    "japanese":   "ja",
    "korean":     "ko",
    "portuguese": "pt",
    "russian":    "ru",
    "spanish":    "es",
}

# Also accept short codes as aliases
SHORT_CODE_TO_LANG = {v: k for k, v in SUPPORTED_LANGUAGES.items() if v is not None}


# ─── Callbacks ───────────────────────────────────────────────────
class Callbacks:
    """Pipeline callbacks. Override for GUI/progress use; default is no-op."""

    def log(self, level: str, msg: str) -> None:
        """level in {'info', 'warn', 'error', 'stage'}."""

    def stage(self, key: str) -> None:
        """key in {'extract', 'transcribe', 'translate', 'ref', 'tts', 'mux', 'done'}."""

    def segment(self, idx: int, status: str) -> None:
        """status in {'gen', 'done'}."""

    def cancelled(self) -> bool:
        """Cooperative cancellation. Return True to abort the pipeline."""
        return False


class StderrCallbacks(Callbacks):
    """CLI-style callbacks that print to stderr, matching the original behavior."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def log(self, level: str, msg: str) -> None:
        # Always show warnings and errors, even when quiet
        if self.verbose or level in ("warn", "error"):
            tag = level.upper()
            click.echo(f"[{tag}] {msg}", err=True)

    def stage(self, key: str) -> None:
        self.log("stage", f"=== Stage: {key} ===")


class _Cancelled(Exception):
    """Raised internally when cbs.cancelled() returns True."""


def _check_cancel(cbs: Callbacks) -> None:
    if cbs.cancelled():
        raise _Cancelled()


# ─── Language helpers ────────────────────────────────────────────
def resolve_language(lang: str) -> str:
    """Normalize language input to canonical Qwen3-TTS name (lowercase).

    Accepts: 'Chinese', 'chinese', 'zh', 'ZH', 'en', 'English', etc.
    Returns: 'chinese', 'english', etc.
    Raises click.UsageError on unsupported language.
    """
    lower = lang.lower()
    if lower in SUPPORTED_LANGUAGES:
        return lower
    if lower in SHORT_CODE_TO_LANG:
        return SHORT_CODE_TO_LANG[lower]
    supported = ", ".join(
        f"{name} ({code})" for name, code in SUPPORTED_LANGUAGES.items() if code
    )
    raise click.UsageError(
        f"Unsupported language: '{lang}'. Supported: {supported}"
    )


def to_whisper_lang(lang: str) -> str | None:
    """Convert canonical language name to whisper short code."""
    return SUPPORTED_LANGUAGES[lang]


# ─── Preflight / probe ───────────────────────────────────────────
INSTALL_HINTS = {
    "ffmpeg":  "brew install ffmpeg",
    "ffprobe": "brew install ffmpeg (provides ffprobe)",
    "uv":      "curl -LsSf https://astral.sh/uv/install.sh | sh",
    "claude":  "npm i -g @anthropic-ai/claude-code",
}


def check_environment(needs_claude: bool = True) -> dict:
    """Report missing external tools and install hints for the GUI.

    Always treated as non-fatal here — the CLI's preflight_check is still
    the gate for command-line use. The GUI calls this on startup so it
    can show a persistent banner instead of only discovering the problem
    mid-pipeline.
    """
    tools = ["ffmpeg", "ffprobe", "uv"]
    if needs_claude:
        tools.append("claude")
    missing = [t for t in tools if not shutil.which(t)]
    return {
        "ok": not missing,
        "missing": missing,
        "hints": {t: INSTALL_HINTS.get(t, "") for t in missing},
    }


def preflight_check(needs_claude: bool = False) -> None:
    """Verify required external tools are on PATH; raise if any are missing."""
    env = check_environment(needs_claude=needs_claude)
    if not env["ok"]:
        raise click.UsageError(
            f"Required tools not found on PATH: {', '.join(env['missing'])}"
        )


def get_video_duration(video_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def probe_video(video_path: Path) -> dict:
    """Return metadata for UI display: name, duration, size, codec, resolution."""
    duration = get_video_duration(video_path)
    size_bytes = video_path.stat().st_size

    # Probe codec + dimensions via ffprobe (best effort; non-fatal if missing)
    codec = ""
    height = 0
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,height",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, check=True,
        ).stdout.splitlines()
        if len(out) >= 2:
            codec = out[0].strip()
            try:
                height = int(out[1].strip())
            except ValueError:
                height = 0
    except subprocess.CalledProcessError:
        pass

    # Format a human-readable size string (MB/GB)
    if size_bytes >= 1024 ** 3:
        size_str = f"{size_bytes / 1024 ** 3:.1f} GB"
    else:
        size_str = f"{size_bytes / 1024 ** 2:.0f} MB"
    parts = [size_str]
    if height:
        parts.append(f"{height}p")
    if codec:
        parts.append(codec.upper())
    meta = " · ".join(parts)

    return {
        "name": video_path.name,
        "path": str(video_path),
        "duration": duration,
        "size_bytes": size_bytes,
        "size": meta,
        "codec": codec,
        "height": height,
    }


# ─── Stage helpers ───────────────────────────────────────────────
def extract_audio(video_path: Path, output_path: Path, cbs: Callbacks) -> None:
    """Extract audio from video as 16kHz mono WAV for whisper."""
    if output_path.exists():
        cbs.log("info", f"Using cached audio: {output_path}")
        return
    cbs.log("info", f"Extracting audio from {video_path.name}...")
    subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", str(output_path), "-y"],
        capture_output=True, check=True,
    )


def extract_original_audio_m4a(video_path: Path, output_path: Path, cbs: Callbacks) -> None:
    """Extract the video's original audio as playable AAC/m4a for the UI player."""
    if output_path.exists():
        return
    cbs.log("info", f"Extracting original audio track for preview: {output_path.name}")
    subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-vn", "-c:a", "aac",
         "-b:a", "128k", str(output_path), "-y"],
        capture_output=True, check=True,
    )


def pick_ref_segment(segments: list[dict]) -> dict:
    """Pick the best segment for voice reference (~10s, clear speech)."""
    return min(
        segments,
        key=lambda s: abs((s["end"] - s["start"]) - REF_CLIP_TARGET_SECONDS),
    )


def extract_ref_clip(
    audio_path: Path,
    segments: list[dict],
    output_path: Path,
    cbs: Callbacks,
) -> tuple[str, dict]:
    """Extract a ~10s clip from the video audio to use as voice reference.

    Picks the segment closest to REF_CLIP_TARGET_SECONDS. Caches the clip
    and its transcript text as a sidecar .txt file. Returns (ref_text, best_seg).
    """
    text_sidecar = output_path.with_suffix(".txt")
    meta_sidecar = output_path.with_suffix(".meta")

    if output_path.exists() and text_sidecar.exists():
        cbs.log("info", f"Using cached ref clip: {output_path}")
        ref_text = text_sidecar.read_text(encoding="utf-8")
        # Best-effort: re-pick for metadata so UI can show the clip range
        best = pick_ref_segment(segments) if segments else {"index": 0, "start": 0.0, "end": 0.0, "text": ref_text}
        return ref_text, best

    best = pick_ref_segment(segments)
    duration = best["end"] - best["start"]

    cbs.log(
        "info",
        f"Auto-extracting ref clip: segment {best['index']} "
        f"({duration:.1f}s at {best['start']:.1f}s)",
    )
    subprocess.run(
        ["ffmpeg", "-i", str(audio_path), "-ss", str(best["start"]),
         "-t", str(duration), "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", str(output_path), "-y"],
        capture_output=True, check=True,
    )
    text_sidecar.write_text(best["text"], encoding="utf-8")
    meta_sidecar.write_text(
        f"{best['index']}\n{best['start']}\n{best['end']}\n",
        encoding="utf-8",
    )
    return best["text"], best


def transcribe_to_srt(
    audio_path: Path,
    srt_path: Path,
    language: str | None,
    prompt: str | None,
    cbs: Callbacks,
) -> None:
    """Shell out to whisper.py (via uv run from GitHub) to generate SRT.

    Streams whisper's stderr to cbs.log('info', …) so the GUI can show
    live progress. Honors cooperative cancellation between lines.
    """
    if srt_path.exists():
        cbs.log("info", f"Using cached SRT: {srt_path}")
        return

    cbs.log("info", "Transcribing audio to SRT...")
    cmd = ["uv", "run", WHISPER_SCRIPT, "-f", "srt", "-o", str(srt_path)]
    if language:
        cmd.extend(["-l", language])
    if prompt:
        cmd.extend(["--prompt", prompt])
    cmd.extend(["-v", str(audio_path)])

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        assert proc.stderr is not None
        for line in proc.stderr:
            _check_cancel(cbs)
            line = line.rstrip()
            if line:
                cbs.log("info", line)
    except _Cancelled:
        proc.terminate()
        proc.wait(timeout=5)
        raise
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"whisper.py failed with exit code {proc.returncode}")


def transcribe_audio_to_text(
    audio_path: Path,
    language: str | None,
    cbs: Callbacks,
    prompt: str | None = None,
) -> str:
    """Run whisper.py on a short clip and return plain text (no timestamps).

    Used by the GUI to fill in the reference voice transcript for both
    auto-extracted clips and user-uploaded reference audio. Streams
    whisper's stderr to cbs.log so the UI can show live progress.
    """
    import tempfile

    cbs.log("info", f"Transcribing reference clip: {audio_path.name}...")
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
        tmp = Path(tf.name)
    try:
        cmd = ["uv", "run", WHISPER_SCRIPT, "-f", "txt", "-o", str(tmp), "-R"]
        if language:
            cmd.extend(["-l", language])
        if prompt:
            cmd.extend(["--prompt", prompt])
        cmd.append(str(audio_path))

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            assert proc.stderr is not None
            for line in proc.stderr:
                _check_cancel(cbs)
                line = line.rstrip()
                if line:
                    cbs.log("info", line)
        except _Cancelled:
            proc.terminate()
            proc.wait(timeout=5)
            raise
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"whisper.py failed with exit code {proc.returncode}")
        return tmp.read_text(encoding="utf-8").strip()
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def translate_srt(
    srt_path: Path,
    translated_path: Path,
    target_lang: str,
    cbs: Callbacks,
) -> None:
    """Translate SRT text lines via claude -p, preserving timestamps."""
    if translated_path.exists():
        cbs.log("info", f"Using cached translation: {translated_path}")
        return

    srt_content = srt_path.read_text(encoding="utf-8")
    prompt = (
        f"Translate the following SRT subtitle file to {target_lang}. "
        "Keep the SRT format exactly: preserve segment numbers, timestamps, "
        "and blank lines between blocks. Only translate the text lines. "
        f"All numbers and dates must be written as {target_lang} words "
        "(e.g. 'April sixteenth' not '4/16', 'thirty' not '30'). "
        "This is critical because the output feeds into a TTS engine. "
        "Output ONLY the translated SRT, no explanation.\n\n"
        f"{srt_content}"
    )

    cbs.log("info", f"Translating SRT to {target_lang} via claude...")
    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True, text=True, check=True,
    )

    translated_path.write_text(result.stdout.strip() + "\n", encoding="utf-8")
    cbs.log("info", f"Translation saved: {translated_path}")


def parse_srt(srt_path: Path) -> list[dict]:
    """Parse SRT file into list of {index, start, end, text}."""
    content = srt_path.read_text(encoding="utf-8")
    segments = []

    for block in re.split(r"\n\n+", content.strip()):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        try:
            index = int(lines[0])
        except ValueError:
            continue
        time_match = re.match(
            r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})",
            lines[1],
        )
        if not time_match:
            continue

        g = [int(x) for x in time_match.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        text = " ".join(lines[2:]).strip()

        if text:
            segments.append({"index": index, "start": start, "end": end, "text": text})

    return segments


def _fmt_srt_time(t: float) -> str:
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = int(t % 60)
    millis = int(round((t - int(t)) * 1000))
    if millis == 1000:
        millis = 0
        seconds += 1
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def write_srt(segments: list[dict], srt_path: Path) -> None:
    """Serialize segments back to SRT. Used after the user edits in the GUI."""
    lines = []
    for seg in segments:
        lines.append(str(seg["index"]))
        lines.append(f"{_fmt_srt_time(seg['start'])} --> {_fmt_srt_time(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")
    srt_path.write_text("\n".join(lines), encoding="utf-8")


def load_ref_audio(ref_audio_path: Path, cbs: Callbacks):
    """Load and prepare reference audio for TTS (mono, 24kHz, mlx array)."""
    import mlx.core as mx

    ref_audio_data, ref_sr = sf.read(str(ref_audio_path))

    if ref_audio_data.ndim > 1:
        ref_audio_data = ref_audio_data.mean(axis=1)
        cbs.log("info", "Converted reference audio to mono")

    if ref_sr != SAMPLE_RATE:
        import librosa
        ref_audio_data = librosa.resample(
            ref_audio_data, orig_sr=ref_sr, target_sr=SAMPLE_RATE,
        )
        cbs.log("info", f"Resampled reference from {ref_sr}Hz to {SAMPLE_RATE}Hz")

    return mx.array(ref_audio_data.astype(np.float32))


def _hf_cache_dir(repo_id: str) -> Path:
    """HuggingFace hub snapshot directory for a given repo id."""
    safe = repo_id.replace("/", "--")
    return Path.home() / ".cache" / "huggingface" / "hub" / f"models--{safe}"


def _model_is_cached(repo_id: str) -> bool:
    """True when at least one snapshot of `repo_id` is on disk.

    HF stores weights under `<cache>/snapshots/<rev>/…`. A non-empty
    snapshots directory is a good-enough proxy for "already downloaded"
    without parsing the revision metadata.
    """
    snapshots = _hf_cache_dir(repo_id) / "snapshots"
    if not snapshots.is_dir():
        return False
    for rev in snapshots.iterdir():
        try:
            if any(rev.iterdir()):
                return True
        except OSError:
            continue
    return False


class _LogStream:
    """File-like sink that calls `emit(line)` for every line written.

    Treats both \\n and \\r as line terminators — tqdm rewrites its
    progress bar with \\r, so without this a 3 GB download would appear
    as one unbroken buffer in the UI log.
    """
    def __init__(self, emit):
        self._emit = emit
        self._buf = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        while self._buf:
            lf = self._buf.find("\n")
            cr = self._buf.find("\r")
            candidates = [i for i in (lf, cr) if i != -1]
            if not candidates:
                break
            i = min(candidates)
            line = self._buf[:i].strip()
            self._buf = self._buf[i + 1:]
            if line:
                try:
                    self._emit(line)
                except Exception:
                    pass
        return len(s)

    def flush(self) -> None:
        pass

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False


def load_tts_model(small: bool, cbs: Callbacks):
    """Load the Qwen3-TTS model. Caller may cache the returned object
    across multiple generate_tts_segments / regenerate_segment calls to
    avoid the ~20s reload cost per interaction.

    On a clean machine the weights (1.2 GB for 0.6B, 3.4 GB for 1.7B)
    come from HuggingFace. We surface that to the UI log instead of
    letting the download vanish silently into stderr.
    """
    from contextlib import redirect_stderr
    from mlx_audio.tts.utils import load_model

    model_name = (
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base" if small
        else "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    )
    size_label = "0.6B" if small else "1.7B"
    cbs.log("info", f"Loading {size_label} model...")
    if not _model_is_cached(model_name):
        approx = "~1.2 GB" if small else "~3.4 GB"
        cbs.log(
            "warn",
            f"{model_name} not cached locally. First run will download "
            f"{approx} from Hugging Face — this can take several minutes.",
        )

    sink = _LogStream(lambda ln: cbs.log("info", ln))
    try:
        with redirect_stderr(sink):
            return load_model(model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load TTS model: {e}") from e


def generate_tts_segments(
    segments: list[dict],
    ref_audio_path: Path,
    ref_text: str,
    cache_dir: Path,
    language: str,
    small: bool,
    cbs: Callbacks,
    model=None,
) -> None:
    """Generate TTS audio for each segment. Per-segment results are cached.

    Pass `model` from a long-lived cache (e.g. the GUI Api) to skip the
    per-call model load — otherwise a fresh model is loaded on demand.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    to_generate = []
    for seg in segments:
        cached = cache_dir / f"{seg['index']:03d}.wav"
        if cached.exists():
            cbs.log("info", f"Segment {seg['index']} cached, skipping")
            cbs.segment(seg["index"], "done")
        else:
            to_generate.append(seg)

    if not to_generate:
        cbs.log("info", "All segments cached, skipping TTS generation")
        return

    if model is None:
        model = load_tts_model(small, cbs)

    ref_audio_mx = load_ref_audio(ref_audio_path, cbs)

    for i, seg in enumerate(to_generate):
        _check_cancel(cbs)
        output_path = cache_dir / f"{seg['index']:03d}.wav"
        cbs.segment(seg["index"], "gen")
        cbs.log(
            "info",
            f"[{i + 1}/{len(to_generate)}] Segment {seg['index']}: "
            f"{seg['text'][:60]}{'...' if len(seg['text']) > 60 else ''}",
        )

        result = next(iter(model.generate(
            text=seg["text"],
            lang_code=language,
            ref_audio=ref_audio_mx,
            ref_text=ref_text,
            verbose=False,
        )))

        audio = np.array(result.audio)
        sf.write(str(output_path), audio, model.sample_rate)
        cbs.segment(seg["index"], "done")

    cbs.log("info", f"Generated {len(to_generate)} segments")


def invalidate_segment_cache(segments_dir: Path, seg_indices: list[int]) -> None:
    """Delete cached TTS clips for the given segment indices (for regen after edits)."""
    for idx in seg_indices:
        p = segments_dir / f"{idx:03d}.wav"
        if p.exists():
            p.unlink()


def combine_audio(
    segments: list[dict],
    cache_dir: Path,
    video_duration: float,
    output_path: Path,
    cbs: Callbacks,
) -> None:
    """Place cached TTS clips at SRT timestamps into a single audio track."""
    total_samples = int(video_duration * SAMPLE_RATE)
    combined = np.zeros(total_samples, dtype=np.float32)

    for seg in segments:
        clip_path = cache_dir / f"{seg['index']:03d}.wav"
        if not clip_path.exists():
            cbs.log("warn", f"Missing clip for segment {seg['index']}")
            continue

        clip_data, clip_sr = sf.read(str(clip_path))
        if clip_sr != SAMPLE_RATE:
            import librosa
            clip_data = librosa.resample(
                clip_data, orig_sr=clip_sr, target_sr=SAMPLE_RATE,
            )

        start_sample = int(seg["start"] * SAMPLE_RATE)
        end_sample = start_sample + len(clip_data)

        if end_sample > total_samples:
            clip_data = clip_data[:total_samples - start_sample]
            end_sample = total_samples

        combined[start_sample:end_sample] = clip_data

    sf.write(str(output_path), combined, SAMPLE_RATE)
    cbs.log("info", f"Combined audio: {output_path}")


# TODO: Integrate vocal-remover (https://github.com/tsurumeso/vocal-remover) as an option.
#       Instead of fully stripping original audio, use vocal-remover to
#       separate vocals from background (music/SFX/ambient), keep the
#       background track, and mix it with the TTS audio.
#       Toggle via a --keep-background flag.
def mux_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    cbs: Callbacks,
) -> None:
    """Replace video audio with the generated TTS track."""
    cbs.log("info", f"Muxing video + audio -> {output_path.name}...")
    subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-i", str(audio_path),
         "-map", "0:v", "-map", "1:a", "-c:v", "copy",
         str(output_path), "-y"],
        capture_output=True, check=True,
    )


# ─── Two-phase pipeline entry points ─────────────────────────────
@dataclass
class TranscriptPrep:
    """Everything produced by prepare_transcript(); consumed by run_dub()."""
    video_path: Path
    cache_path: Path
    extracted_audio: Path
    srt_file: Path
    translated_srt: Path | None
    segments: list[dict]          # final segments (translated if target_lang else source)
    original_segments: list[dict] # always source-language segments
    ref_audio_path: Path
    ref_text: str
    ref_segment: dict             # {index, start, end, text} for the ref clip
    language: str                 # canonical
    target_lang: str | None       # canonical
    lang_key: str                 # target_lang or language
    voice_name: str
    segments_dir: Path
    combined_audio: Path
    output_path: Path
    small: bool = False


def prepare_transcript(
    video: Path,
    ref_audio: Path | None = None,
    ref_text: str | None = None,
    language: str = "auto",
    target_lang: str | None = None,
    output: Path | None = None,
    srt_file_arg: Path | None = None,
    prompt: str | None = None,
    cache_dir: Path | None = None,
    small: bool = False,
    cbs: Callbacks | None = None,
) -> TranscriptPrep:
    """Stages 1, 2, 2.5 + reference clip extraction. No TTS yet.

    Returns a TranscriptPrep the caller can use to display segments to the
    user (editing if desired) and then feed into run_dub().
    """
    cbs = cbs or Callbacks()

    # Normalize languages
    language = resolve_language(language)
    if target_lang:
        target_lang = resolve_language(target_lang)
        if target_lang == "auto":
            raise click.UsageError("--target-lang cannot be 'auto'.")

    # Upload mode must come with a transcript — TTS needs it.
    if ref_audio and not ref_text:
        raise click.UsageError("ref_text is required when ref_audio is provided.")
    # The CLI requires at least one of ref_audio / target_lang (otherwise
    # the dub is a near-identical re-render). The GUI is permissive — the
    # user may transcribe first, decide later — so that gate lives in cli()
    # now, not here.

    preflight_check(needs_claude=bool(target_lang))

    layout = _resolve_cache_layout(video, language, target_lang, ref_audio, output, cache_dir)
    layout.cache_path.mkdir(parents=True, exist_ok=True)

    cbs.stage("extract")
    extract_audio(layout.video_path, layout.extracted_audio, cbs)
    _check_cancel(cbs)

    if srt_file_arg:
        cbs.log("info", f"Using provided SRT: {srt_file_arg}")
        layout.srt_file.write_text(Path(srt_file_arg).read_text(encoding="utf-8"), encoding="utf-8")
        original_segments = parse_srt(layout.srt_file)
        segments = original_segments
    else:
        whisper_lang = to_whisper_lang(language)

        # Auto-add Chinese prompt hint for zh to avoid translation mode
        effective_prompt = prompt
        if whisper_lang and whisper_lang.startswith("zh") and not prompt:
            effective_prompt = "以下是繁體中文的逐字稿"

        cbs.stage("transcribe")
        transcribe_to_srt(layout.extracted_audio, layout.srt_file, whisper_lang, effective_prompt, cbs)
        _check_cancel(cbs)
        original_segments = parse_srt(layout.srt_file)

        if target_lang:
            cbs.stage("translate")
            translate_srt(layout.srt_file, layout.translated_srt, target_lang, cbs)
            _check_cancel(cbs)
            segments = parse_srt(layout.translated_srt)
        else:
            segments = original_segments

    cbs.log("info", f"Found {len(segments)} segments")

    if not segments:
        raise RuntimeError("No segments found in transcription.")

    if layout.auto_ref:
        ref_audio_path = layout.cache_path / "ref_clip.wav"
        cbs.stage("ref")
        ref_text_resolved, ref_segment = extract_ref_clip(
            layout.extracted_audio, original_segments, ref_audio_path, cbs,
        )
    else:
        ref_audio_path = Path(ref_audio)
        ref_text_resolved = ref_text
        ref_segment = {"index": 0, "start": 0.0, "end": 0.0, "text": ref_text}

    _check_cancel(cbs)

    return _prep_from_layout(
        layout, segments, original_segments,
        ref_audio_path, ref_text_resolved, ref_segment, small,
    )


@dataclass
class _CacheLayout:
    """Resolved file paths shared by prepare_transcript + try_load_from_cache.

    Centralizing avoids silent drift — changing one naming convention
    only requires changing it here, and probe_cache can't disagree with
    the live pipeline about where a file should live.
    """
    video_path: Path
    cache_path: Path
    auto_ref: bool
    voice_name: str
    language: str
    target_lang: str | None
    lang_key: str
    extracted_audio: Path
    srt_file: Path
    translated_srt: Path | None
    segments_dir: Path
    combined_audio: Path
    output_path: Path


def _resolve_cache_layout(
    video: Path,
    language: str,
    target_lang: str | None,
    ref_audio: Path | None,
    output: Path | None,
    cache_dir: Path | None,
) -> _CacheLayout:
    video_path = Path(video)
    auto_ref = ref_audio is None
    voice_name = "original" if auto_ref else Path(ref_audio).stem
    lang_key = target_lang or language

    cache_path = Path(cache_dir) if cache_dir is not None else video_path.parent / video_path.stem
    return _CacheLayout(
        video_path=video_path,
        cache_path=cache_path,
        auto_ref=auto_ref,
        voice_name=voice_name,
        language=language,
        target_lang=target_lang,
        lang_key=lang_key,
        extracted_audio=cache_path / "audio.wav",
        srt_file=cache_path / "transcript.srt",
        translated_srt=cache_path / f"transcript_{target_lang}.srt" if target_lang else None,
        segments_dir=cache_path / f"segments_{lang_key}_{voice_name}",
        combined_audio=cache_path / f"combined_{lang_key}_{voice_name}.wav",
        output_path=(
            Path(output) if output is not None
            else video_path.with_stem(f"{video_path.stem}_{lang_key}_{voice_name}")
        ),
    )


def _prep_from_layout(
    layout: _CacheLayout,
    segments: list[dict],
    original_segments: list[dict],
    ref_audio_path: Path,
    ref_text: str,
    ref_segment: dict,
    small: bool,
) -> "TranscriptPrep":
    return TranscriptPrep(
        video_path=layout.video_path,
        cache_path=layout.cache_path,
        extracted_audio=layout.extracted_audio,
        srt_file=layout.srt_file,
        translated_srt=layout.translated_srt,
        segments=segments,
        original_segments=original_segments,
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        ref_segment=ref_segment,
        language=layout.language,
        target_lang=layout.target_lang,
        lang_key=layout.lang_key,
        voice_name=layout.voice_name,
        segments_dir=layout.segments_dir,
        combined_audio=layout.combined_audio,
        output_path=layout.output_path,
        small=small,
    )


def try_load_from_cache(
    video: Path,
    language: str = "auto",
    target_lang: str | None = None,
    cache_dir: Path | None = None,
    ref_audio: Path | None = None,
    ref_text: str | None = None,
    small: bool = False,
    output: Path | None = None,
) -> "TranscriptPrep | None":
    """Build a TranscriptPrep from existing cache files, or None if the
    cache is incomplete for the requested configuration. Never invokes
    whisper, claude, or ffmpeg — purely a filesystem probe + SRT parse.
    """
    language = resolve_language(language)
    if target_lang:
        target_lang = resolve_language(target_lang)
        if target_lang == "auto":
            return None

    layout = _resolve_cache_layout(video, language, target_lang, ref_audio, output, cache_dir)

    if not (layout.extracted_audio.exists() and layout.srt_file.exists()):
        return None
    if layout.translated_srt is not None and not layout.translated_srt.exists():
        return None

    if layout.auto_ref:
        ref_audio_path = layout.cache_path / "ref_clip.wav"
        text_sidecar = ref_audio_path.with_suffix(".txt")
        if not (ref_audio_path.exists() and text_sidecar.exists()):
            return None
        ref_text_resolved = text_sidecar.read_text(encoding="utf-8")
        ref_segment = {"index": 0, "start": 0.0, "end": 0.0, "text": ref_text_resolved}
        meta_sidecar = ref_audio_path.with_suffix(".meta")
        if meta_sidecar.exists():
            try:
                lines = meta_sidecar.read_text(encoding="utf-8").splitlines()
                if len(lines) >= 3:
                    ref_segment = {
                        "index": int(lines[0]),
                        "start": float(lines[1]),
                        "end": float(lines[2]),
                        "text": ref_text_resolved,
                    }
            except (ValueError, OSError):
                pass
    else:
        if ref_text is None:
            return None
        ref_audio_path = Path(ref_audio)
        ref_text_resolved = ref_text
        ref_segment = {"index": 0, "start": 0.0, "end": 0.0, "text": ref_text}

    original_segments = parse_srt(layout.srt_file)
    segments = parse_srt(layout.translated_srt) if layout.translated_srt else original_segments
    if not segments:
        return None

    return _prep_from_layout(
        layout, segments, original_segments,
        ref_audio_path, ref_text_resolved, ref_segment, small,
    )


def finalize_dub(
    prep: TranscriptPrep,
    segments: list[dict],
    cbs: Callbacks,
) -> Path:
    """Combine per-segment wavs and mux into the output video.

    Emits stage('mux') then stage('done'). Shared by run_dub (end of
    full pipeline) and the GUI's regen-refresh path (so a single segment
    edit updates the muxed preview without re-running TTS).
    """
    cbs.stage("mux")
    video_duration = get_video_duration(prep.video_path)
    combine_audio(segments, prep.segments_dir, video_duration, prep.combined_audio, cbs)
    mux_video(prep.video_path, prep.combined_audio, prep.output_path, cbs)
    cbs.stage("done")
    return prep.output_path


def run_dub(
    prep: TranscriptPrep,
    segments: list[dict] | None = None,
    cbs: Callbacks | None = None,
    model=None,
) -> Path:
    """Stages 3 and 4. If `segments` is passed (e.g. user-edited), it is
    written back to the SRT before TTS runs and supersedes prep.segments.
    Returns the final output video path.
    """
    cbs = cbs or Callbacks()
    segs = list(segments) if segments is not None else prep.segments

    if segments is not None:
        active_srt = prep.translated_srt if prep.target_lang else prep.srt_file
        if active_srt is not None:
            write_srt(segs, active_srt)
            cbs.log("info", f"Saved edited SRT to {active_srt.name}")

    cbs.stage("tts")
    generate_tts_segments(
        segs, prep.ref_audio_path, prep.ref_text,
        prep.segments_dir, prep.lang_key, prep.small, cbs,
        model=model,
    )
    _check_cancel(cbs)

    return finalize_dub(prep, segs, cbs)


def regenerate_segment(
    prep: TranscriptPrep,
    seg_idx: int,
    segments: list[dict] | None = None,
    cbs: Callbacks | None = None,
    model=None,
) -> None:
    """Invalidate and re-synthesize TTS for a single segment.

    Useful when the user edits one segment and wants to preview the new
    voice before committing to a full re-render. Pass a cached `model`
    to make this near-instant instead of paying the 20s reload.
    """
    cbs = cbs or Callbacks()
    segs = list(segments) if segments is not None else prep.segments
    target = next((s for s in segs if s["index"] == seg_idx), None)
    if target is None:
        raise ValueError(f"Segment {seg_idx} not found")

    invalidate_segment_cache(prep.segments_dir, [seg_idx])
    generate_tts_segments(
        [target], prep.ref_audio_path, prep.ref_text,
        prep.segments_dir, prep.lang_key, prep.small, cbs,
        model=model,
    )


def run_pipeline(
    video: Path,
    ref_audio: Path | None = None,
    ref_text: str | None = None,
    language: str = "auto",
    target_lang: str | None = None,
    output: Path | None = None,
    srt_file_arg: Path | None = None,
    prompt: str | None = None,
    cache_dir: Path | None = None,
    small: bool = False,
    cbs: Callbacks | None = None,
) -> Path:
    """Full pipeline: prepare_transcript + run_dub. Returns output video path."""
    cbs = cbs or Callbacks()
    prep = prepare_transcript(
        video=video, ref_audio=ref_audio, ref_text=ref_text,
        language=language, target_lang=target_lang, output=output,
        srt_file_arg=srt_file_arg, prompt=prompt, cache_dir=cache_dir,
        small=small, cbs=cbs,
    )
    return run_dub(prep, cbs=cbs)


# ─── CLI ─────────────────────────────────────────────────────────
@click.command()
@click.argument("video", type=click.Path(exists=True))
@click.option(
    "-r", "--ref-audio", default=None,
    type=click.Path(exists=True),
    help="Reference audio for dubbing voice (auto-extracted from video if omitted)",
)
@click.option(
    "-t", "--ref-text", default=None,
    help="Transcript of the reference audio (auto-detected if omitted with -r)",
)
@click.option(
    "-l", "--language", default="auto",
    help="Source language (default: auto). Accepts: "
    "auto, chinese/zh, english/en, french/fr, german/de, italian/it, "
    "japanese/ja, korean/ko, portuguese/pt, russian/ru, spanish/es",
)
@click.option(
    "--target-lang", default=None,
    help="Target language for translation. Same values as -l (except auto). "
    "Omit to keep original language.",
)
@click.option("-o", "--output", default=None, help="Output video path (default: <input>_<lang>_<voice>.mp4)")
@click.option(
    "--srt", "srt_file_arg", default=None,
    type=click.Path(exists=True), help="Pre-existing SRT file (skip whisper + translation)",
)
@click.option("--prompt", default=None, help="Whisper transcription prompt (e.g. language hint)")
@click.option("--cache-dir", default=None, help="Cache directory (default: <video_dir>/<video_stem>/)")
@click.option("--small", is_flag=True, help="Use faster 0.6B TTS model")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.version_option(version="0.1.0")
def cli(
    video: str,
    ref_audio: str | None,
    ref_text: str | None,
    language: str,
    target_lang: str | None,
    output: str | None,
    srt_file_arg: str | None,
    prompt: str | None,
    cache_dir: str | None,
    small: bool,
    verbose: bool,
):
    """Automatically dub a video using Whisper transcription + Qwen3-TTS.

    Transcribes VIDEO, optionally translates to --target-lang, then
    re-synthesizes speech using a reference voice. When -r is omitted,
    the original speaker's voice is auto-extracted from the video.

    Cache and output are stored next to the input video.

    \b
    Recommended workflow for best quality:
      1. Run once with -v to generate the SRT:
         auto_dub.py -l zh --target-lang en -v video.mp4
      2. Review the SRT in <video_name>/transcript.srt (or transcript_<lang>.srt
         for translations). AI transcription often misses names, numbers, or
         domain terms. Fix these before the TTS stage turns them into audio.
      3. Delete the segments you want to regenerate, or delete the whole
         segments_* folder, then re-run. Cached segments are skipped.
      4. To use a fully hand-edited SRT, pass --srt:
         auto_dub.py --srt corrected.srt video.mp4

    \b
    Examples:
      # Translate Chinese video to English (auto-extract speaker voice)
      auto_dub.py -l zh --target-lang en video.mp4
      # Use a specific voice for dubbing
      auto_dub.py -r speaker.wav -t "ref transcript" -l zh --target-lang en video.mp4
      # Re-voice in same language with a different voice
      auto_dub.py -r speaker.wav -t "ref transcript" -l zh video.mp4
      # Use a manually edited SRT (skip transcription + translation)
      auto_dub.py --srt edited.srt video.mp4
    """
    # CLI-only gate: producing a video that's identical-ish to the input
    # is almost never what a command-line invocation wants.
    if not ref_audio and not target_lang:
        raise click.UsageError(
            "Either provide -r/--ref-audio (re-voice) or --target-lang (translate). "
            "Nothing to do without both."
        )

    cbs = StderrCallbacks(verbose=verbose)
    try:
        output_path = run_pipeline(
            video=Path(video),
            ref_audio=Path(ref_audio) if ref_audio else None,
            ref_text=ref_text,
            language=language,
            target_lang=target_lang,
            output=Path(output) if output else None,
            srt_file_arg=Path(srt_file_arg) if srt_file_arg else None,
            prompt=prompt,
            cache_dir=Path(cache_dir) if cache_dir else None,
            small=small,
            cbs=cbs,
        )
    except _Cancelled:
        click.echo("Cancelled.", err=True)
        sys.exit(130)
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(str(output_path))


if __name__ == "__main__":
    cli()
