# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mlx-audio>=0.3.1",
#     "click>=8.0",
#     "pydub>=0.25.0",
#     "httpx>=0.25.0",
# ]
# ///
"""
Qwen3-ASR CLI - Audio transcription using mlx-audio on Apple Silicon.

USAGE:
    # Basic transcription (auto-detect language)
    uv run qwen3_asr.py audio.mp3

    # Chinese audio
    uv run qwen3_asr.py -l zh audio.mp3

    # Larger model for better accuracy
    uv run qwen3_asr.py -m mlx-community/Qwen3-ASR-1.7B-8bit audio.mp3

    # SRT subtitles output
    uv run qwen3_asr.py -f srt -o subtitles.srt video.mp4

    # From URL
    uv run qwen3_asr.py https://example.com/audio.mp3

    # JSON output with timing metadata
    uv run qwen3_asr.py -f json audio.mp3

    # Multiple files (concatenated)
    uv run qwen3_asr.py part1.mp3 part2.mp3

MODELS:
    mlx-community/Qwen3-ASR-0.6B-8bit   ~1.0 GB (default, fast)
    mlx-community/Qwen3-ASR-0.6B-4bit   ~0.3 GB (smallest)
    mlx-community/Qwen3-ASR-1.7B-8bit   ~2.5 GB (best accuracy)

REQUIREMENTS:
    - macOS 15+ (Sequoia), Apple Silicon (M1+)
    - ffmpeg: brew install ffmpeg
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import click
import httpx
from pydub import AudioSegment

DEFAULT_MODEL = "mlx-community/Qwen3-ASR-0.6B-8bit"

# Language name mapping for Qwen3-ASR prompt
LANG_NAMES = {
    "zh": "Chinese", "zhs": "Chinese", "zht": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
}


def log(message: str, verbose: bool = True, level: str = "INFO"):
    if verbose or level == "ERROR":
        click.echo(f"[{level}] {message}", err=(level == "ERROR"))


def download_if_url(input_path: str, temp_dir: Path, verbose: bool = False) -> Path:
    if input_path.startswith(("http://", "https://")):
        url_path = input_path.split("?")[0]
        filename = Path(url_path).name or "downloaded_audio"
        output_file = temp_dir / filename
        log(f"Downloading {input_path}...", verbose)
        with httpx.Client(follow_redirects=True, timeout=300.0) as client:
            response = client.get(input_path)
            response.raise_for_status()
            output_file.write_bytes(response.content)
        return output_file
    path = Path(input_path)
    if not path.exists():
        raise click.UsageError(f"Input file not found: {input_path}")
    return path


def convert_to_wav(input_path: Path, output_path: Path, verbose: bool = False) -> Path:
    log(f"Converting {input_path.name} to WAV...", verbose)
    audio = AudioSegment.from_file(str(input_path))
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(str(output_path), format="wav")
    return output_path


def get_audio_duration(wav_path: Path) -> float:
    audio = AudioSegment.from_file(str(wav_path))
    return len(audio) / 1000.0


def _format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _seconds_to_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _seconds_to_lrc_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"[{minutes:02d}:{secs:05.2f}]"


def transcribe(
    audio_path: Path,
    model_name: str,
    language: str | None = None,
    verbose: bool = False,
) -> dict:
    """Transcribe audio and return result with timing metadata."""
    from mlx_audio.stt.generate import generate_transcription
    from mlx_audio.stt.utils import load_model

    lang_name = LANG_NAMES.get(language.lower(), language) if language else None

    log(f"Loading model: {model_name}", verbose)
    load_start = time.monotonic()
    model = load_model(model_name)
    load_elapsed = time.monotonic() - load_start
    log(f"Model loaded in {load_elapsed:.1f}s", verbose)

    log(f"Transcribing {audio_path.name} (language={lang_name or 'auto'})...", verbose)

    kwargs = {}
    if lang_name:
        kwargs["language"] = lang_name

    infer_start = time.monotonic()
    result = generate_transcription(
        model=model,
        audio=str(audio_path),
        verbose=verbose,
        **kwargs,
    )
    infer_elapsed = time.monotonic() - infer_start

    audio_duration = get_audio_duration(audio_path)
    rtf = infer_elapsed / audio_duration if audio_duration > 0 else 0

    # Build segments from result if available
    segments = []
    if hasattr(result, "segments") and result.segments:
        for seg in result.segments:
            start = seg.get("start", 0) if isinstance(seg, dict) else getattr(seg, "start_time", 0)
            end = seg.get("end", 0) if isinstance(seg, dict) else getattr(seg, "end_time", 0)
            text = seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")
            if text.strip():
                segments.append({"start": start, "end": end, "text": text.strip()})

    # Fallback: single segment from full text
    if not segments and result.text.strip():
        segments.append({"start": 0, "end": audio_duration, "text": result.text.strip()})

    return {
        "text": result.text.strip(),
        "segments": segments,
        "audio_duration": audio_duration,
        "load_time": load_elapsed,
        "infer_time": infer_elapsed,
        "rtf": rtf,
        "model": model_name,
        "language": lang_name or "auto",
    }


def format_txt(segments: list[dict]) -> str:
    return "\n".join(seg["text"] for seg in segments if seg["text"])


def format_srt(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        if not seg["text"]:
            continue
        start = _seconds_to_srt_time(seg["start"])
        end = _seconds_to_srt_time(seg["end"])
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def format_json(result: dict) -> str:
    return json.dumps(result, indent=2, ensure_ascii=False)


def format_lrc(segments: list[dict]) -> str:
    lines = []
    for seg in segments:
        if not seg["text"]:
            continue
        time_str = _seconds_to_lrc_time(seg["start"])
        lines.append(f"{time_str}{seg['text']}")
    return "\n".join(lines)


def format_output(result: dict, fmt: str) -> str:
    if fmt == "json":
        return format_json(result)
    formatters = {"txt": format_txt, "srt": format_srt, "lrc": format_lrc}
    formatter = formatters.get(fmt, format_txt)
    return formatter(result["segments"])


@click.command()
@click.argument("inputs", nargs=-1, required=True)
@click.option("-o", "--output", default=None, help="Output file path")
@click.option("-l", "--language", default=None, help="Language code (default: auto-detect)")
@click.option("-m", "--model", default=None, help=f"Model name (default: {DEFAULT_MODEL})")
@click.option(
    "-f", "--format", "output_format",
    type=click.Choice(["txt", "srt", "json", "lrc"]),
    default="txt",
    help="Output format (default: txt)",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.version_option(version="0.1.0")
def cli(
    inputs: tuple[str, ...],
    output: str | None,
    language: str | None,
    model: str | None,
    output_format: str,
    verbose: bool,
):
    """Transcribe audio using Qwen3-ASR via mlx-audio (Apple Silicon).

    INPUTS can be file paths or HTTP(S) URLs. Multiple inputs are concatenated.

    \b
    Examples:
        qwen3_asr.py audio.mp3
        qwen3_asr.py -l zh chinese_audio.mp3
        qwen3_asr.py -f srt -o subtitles.srt video.mp4
        qwen3_asr.py part1.mp3 part2.mp3
        qwen3_asr.py https://example.com/audio.mp3
    """
    model_name = model or DEFAULT_MODEL

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        wav_path = temp_dir / "input.wav"
        if len(inputs) == 1:
            local_path = download_if_url(inputs[0], temp_dir, verbose)
            convert_to_wav(local_path, wav_path, verbose)
        else:
            combined = AudioSegment.empty()
            for input_path in inputs:
                local = download_if_url(input_path, temp_dir, verbose)
                combined += AudioSegment.from_file(str(local))
            combined = combined.set_frame_rate(16000).set_channels(1)
            combined.export(str(wav_path), format="wav")

        result = transcribe(wav_path, model_name, language, verbose)

    # Determine output file path
    if output is None:
        if len(inputs) == 1 and not inputs[0].startswith(("http://", "https://")):
            output_file = Path(inputs[0]).with_suffix(f".{output_format}")
        else:
            output_file = Path(f"qwen3_asr_output.{output_format}")
    else:
        output_file = Path(output)
        if not output_file.suffix:
            output_file = output_file.with_suffix(f".{output_format}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    formatted = format_output(result, output_format)
    output_file.write_text(formatted, encoding="utf-8")

    if verbose or sys.stderr.isatty():
        click.echo(
            f"[RTF={result['rtf']:.3f}] "
            f"[infer={result['infer_time']:.1f}s] "
            f"[audio={result['audio_duration']:.1f}s] "
            f"[model={result['model']}]",
            err=True,
        )

    click.echo(str(output_file))


if __name__ == "__main__":
    cli()
