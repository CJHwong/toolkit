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
"""
import re
import shutil
import subprocess
import sys
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


def log(message: str, verbose: bool = True):
    if verbose:
        click.echo(f"[INFO] {message}", err=True)


def preflight_check(needs_claude: bool = False):
    """Verify required external tools are on PATH."""
    missing = []
    for tool in ("ffmpeg", "ffprobe", "uv"):
        if not shutil.which(tool):
            missing.append(tool)
    if needs_claude and not shutil.which("claude"):
        missing.append("claude")
    if missing:
        raise click.UsageError(f"Required tools not found on PATH: {', '.join(missing)}")


def get_video_duration(video_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def extract_audio(video_path: Path, output_path: Path, verbose: bool = False):
    """Extract audio from video as 16kHz mono WAV for whisper."""
    if output_path.exists():
        log(f"Using cached audio: {output_path}", verbose)
        return
    log(f"Extracting audio from {video_path.name}...", verbose)
    subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", str(output_path), "-y"],
        capture_output=True, check=True,
    )


def extract_ref_clip(
    audio_path: Path,
    segments: list[dict],
    output_path: Path,
    verbose: bool = False,
) -> str:
    """Extract a ~10s clip from the video audio to use as voice reference.

    Picks the segment closest to REF_CLIP_TARGET_SECONDS. Caches the clip
    and its transcript text as a sidecar .txt file. Returns ref_text.
    """
    text_sidecar = output_path.with_suffix(".txt")

    if output_path.exists() and text_sidecar.exists():
        log(f"Using cached ref clip: {output_path}", verbose)
        return text_sidecar.read_text(encoding="utf-8")

    best = pick_ref_segment(segments)
    duration = best["end"] - best["start"]

    log(
        f"Auto-extracting ref clip: segment {best['index']} "
        f"({duration:.1f}s at {best['start']:.1f}s)",
        verbose,
    )
    subprocess.run(
        ["ffmpeg", "-i", str(audio_path), "-ss", str(best["start"]),
         "-t", str(duration), "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", str(output_path), "-y"],
        capture_output=True, check=True,
    )
    text_sidecar.write_text(best["text"], encoding="utf-8")
    return best["text"]


def pick_ref_segment(segments: list[dict]) -> dict:
    """Pick the best segment for voice reference (~10s, clear speech)."""
    # Sort by how close the duration is to the target
    return min(
        segments,
        key=lambda s: abs((s["end"] - s["start"]) - REF_CLIP_TARGET_SECONDS),
    )


def transcribe_to_srt(
    audio_path: Path,
    srt_path: Path,
    language: str | None = None,
    prompt: str | None = None,
    verbose: bool = False,
):
    """Shell out to whisper.py (via uv run from GitHub) to generate SRT."""
    if srt_path.exists():
        log(f"Using cached SRT: {srt_path}", verbose)
        return

    log("Transcribing audio to SRT...", verbose)
    cmd = ["uv", "run", WHISPER_SCRIPT, "-f", "srt", "-o", str(srt_path)]
    if language:
        cmd.extend(["-l", language])
    if prompt:
        cmd.extend(["--prompt", prompt])
    if verbose:
        cmd.append("-v")
    cmd.append(str(audio_path))

    subprocess.run(cmd, check=True)


def translate_srt(
    srt_path: Path,
    translated_path: Path,
    target_lang: str,
    verbose: bool = False,
):
    """Translate SRT text lines via claude -p, preserving timestamps."""
    if translated_path.exists():
        log(f"Using cached translation: {translated_path}", verbose)
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

    log(f"Translating SRT to {target_lang} via claude...", verbose)
    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True, text=True, check=True,
    )

    translated_path.write_text(result.stdout.strip() + "\n", encoding="utf-8")
    log(f"Translation saved: {translated_path}", verbose)


def parse_srt(srt_path: Path) -> list[dict]:
    """Parse SRT file into list of {index, start, end, text}."""
    content = srt_path.read_text(encoding="utf-8")
    segments = []

    for block in re.split(r"\n\n+", content.strip()):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        index = int(lines[0])
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


def load_ref_audio(ref_audio_path: Path, verbose: bool = False):
    """Load and prepare reference audio for TTS (mono, 24kHz, mlx array)."""
    import mlx.core as mx

    ref_audio_data, ref_sr = sf.read(str(ref_audio_path))

    if ref_audio_data.ndim > 1:
        ref_audio_data = ref_audio_data.mean(axis=1)
        log("Converted reference audio to mono", verbose)

    if ref_sr != SAMPLE_RATE:
        import librosa
        ref_audio_data = librosa.resample(
            ref_audio_data, orig_sr=ref_sr, target_sr=SAMPLE_RATE,
        )
        log(f"Resampled reference from {ref_sr}Hz to {SAMPLE_RATE}Hz", verbose)

    return mx.array(ref_audio_data.astype(np.float32))


def generate_tts_segments(
    segments: list[dict],
    ref_audio_path: Path,
    ref_text: str,
    cache_dir: Path,
    language: str,
    small: bool = False,
    verbose: bool = False,
):
    """Generate TTS audio for each segment. Model loaded once, results cached."""
    from mlx_audio.tts.utils import load_model

    cache_dir.mkdir(parents=True, exist_ok=True)

    to_generate = []
    for seg in segments:
        cached = cache_dir / f"{seg['index']:03d}.wav"
        if cached.exists():
            log(f"Segment {seg['index']} cached, skipping", verbose)
        else:
            to_generate.append(seg)

    if not to_generate:
        log("All segments cached, skipping TTS generation", verbose)
        return

    model_name = (
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base" if small
        else "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    )
    log(f"Loading {'0.6B' if small else '1.7B'} model...", verbose)
    model = load_model(model_name)

    ref_audio_mx = load_ref_audio(ref_audio_path, verbose)

    for i, seg in enumerate(to_generate):
        output_path = cache_dir / f"{seg['index']:03d}.wav"
        log(
            f"[{i + 1}/{len(to_generate)}] Segment {seg['index']}: "
            f"{seg['text'][:60]}{'...' if len(seg['text']) > 60 else ''}",
            verbose,
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

    log(f"Generated {len(to_generate)} segments", verbose)


def combine_audio(
    segments: list[dict],
    cache_dir: Path,
    video_duration: float,
    output_path: Path,
    verbose: bool = False,
):
    """Place cached TTS clips at SRT timestamps into a single audio track."""
    total_samples = int(video_duration * SAMPLE_RATE)
    combined = np.zeros(total_samples, dtype=np.float32)

    for seg in segments:
        clip_path = cache_dir / f"{seg['index']:03d}.wav"
        if not clip_path.exists():
            log(f"Warning: missing clip for segment {seg['index']}", verbose=True)
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
    log(f"Combined audio: {output_path}", verbose)


# TODO: Integrate vocal-remover (https://github.com/tsurumeso/vocal-remover) as an option.
#       Instead of fully stripping original audio, use vocal-remover to
#       separate vocals from background (music/SFX/ambient), keep the
#       background track, and mix it with the TTS audio.
#       Toggle via a --keep-background flag.
def mux_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    verbose: bool = False,
):
    """Replace video audio with the generated TTS track."""
    log(f"Muxing video + audio -> {output_path.name}...", verbose)
    subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-i", str(audio_path),
         "-map", "0:v", "-map", "1:a", "-c:v", "copy",
         str(output_path), "-y"],
        capture_output=True, check=True,
    )


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
    # Validate languages
    language = resolve_language(language)
    if target_lang:
        target_lang = resolve_language(target_lang)
        if target_lang == "auto":
            raise click.UsageError("--target-lang cannot be 'auto'.")

    # Validation
    if ref_audio and not ref_text:
        raise click.UsageError("-t/--ref-text is required when -r/--ref-audio is provided.")
    if not ref_audio and not target_lang:
        raise click.UsageError(
            "Either provide -r/--ref-audio (re-voice) or --target-lang (translate). "
            "Nothing to do without both."
        )

    preflight_check(needs_claude=bool(target_lang))

    video_path = Path(video)
    auto_ref = ref_audio is None
    voice_name = "original" if auto_ref else Path(ref_audio).stem

    # Language key: target language if translating, else source language
    lang_key = target_lang or language

    # Cache dir: sibling to input video, named after the video
    if cache_dir is None:
        cache_path = video_path.parent / video_path.stem
    else:
        cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Output: <video_stem>_<lang>_<voice>.mp4
    if output is None:
        output_path = video_path.with_stem(
            f"{video_path.stem}_{lang_key}_{voice_name}"
        )
    else:
        output_path = Path(output)

    # Shared cache (language-independent)
    extracted_audio = cache_path / "audio.wav"
    srt_file = cache_path / "transcript.srt"

    # Per-language + per-voice cache
    segments_dir = cache_path / f"segments_{lang_key}_{voice_name}"
    combined_audio = cache_path / f"combined_{lang_key}_{voice_name}.wav"

    # Stage 1: Extract audio from video (always needed for whisper or auto-ref)
    log("=== Stage 1: Extract audio ===", verbose)
    extract_audio(video_path, extracted_audio, verbose)

    if srt_file_arg:
        # Manual SRT: skip whisper + translation entirely
        log(f"Using provided SRT: {srt_file_arg}", verbose)
        segments = parse_srt(Path(srt_file_arg))
    else:
        whisper_lang = to_whisper_lang(language)

        # Auto-add Chinese prompt hint for zh to avoid translation mode
        effective_prompt = prompt
        if whisper_lang and whisper_lang.startswith("zh") and not prompt:
            effective_prompt = "以下是繁體中文的逐字稿"

        # Stage 2: Transcribe to SRT (original language, always preserved)
        log("=== Stage 2: Transcribe ===", verbose)
        transcribe_to_srt(extracted_audio, srt_file, whisper_lang, effective_prompt, verbose)

        # Stage 2.5: Translate SRT if target language specified
        if target_lang:
            translated_srt = cache_path / f"transcript_{target_lang}.srt"
            log(f"=== Stage 2.5: Translate to {target_lang} ===", verbose)
            translate_srt(srt_file, translated_srt, target_lang, verbose)
            segments = parse_srt(translated_srt)
        else:
            segments = parse_srt(srt_file)

    log(f"Found {len(segments)} segments", verbose)

    if not segments:
        click.echo("No segments found in transcription.", err=True)
        sys.exit(1)

    # Resolve reference audio: auto-extract from video or use provided
    if auto_ref:
        ref_audio_path = cache_path / "ref_clip.wav"
        log("=== Auto-extracting voice reference ===", verbose)
        # Use original-language SRT segments to pick a clear clip
        original_segments = parse_srt(srt_file) if srt_file.exists() else segments
        ref_text_resolved = extract_ref_clip(
            extracted_audio, original_segments, ref_audio_path, verbose,
        )
    else:
        ref_audio_path = Path(ref_audio)
        ref_text_resolved = ref_text

    # Stage 3: Generate TTS per segment (model loaded once)
    log("=== Stage 3: Generate TTS ===", verbose)
    generate_tts_segments(
        segments, ref_audio_path, ref_text_resolved,
        segments_dir, lang_key, small, verbose,
    )

    # Stage 4: Combine clips and mux with video
    log("=== Stage 4: Combine and mux ===", verbose)
    video_duration = get_video_duration(video_path)
    combine_audio(segments, segments_dir, video_duration, combined_audio, verbose)
    mux_video(video_path, combined_audio, output_path, verbose)

    click.echo(str(output_path))


if __name__ == "__main__":
    cli()
