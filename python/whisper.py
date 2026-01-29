# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pywhispercpp>=1.4.0",
#     "click>=8.0",
#     "pydub>=0.25.0",
#     "httpx>=0.25.0",
#     "huggingface_hub>=0.20.0",
# ]
# ///
"""
Whisper CLI - Audio transcription using pywhispercpp.

USAGE:
    # Run directly from GitHub (no clone needed):
    URL=https://raw.githubusercontent.com/CJHwong/toolkit/main/python/whisper.py

    # Basic transcription
    uv run $URL audio.mp3

    # Chinese audio (auto-selects breeze model)
    uv run $URL -l zh audio.mp3

    # Multiple files (concatenated)
    uv run $URL part1.mp3 part2.mp3

    # From URL
    uv run $URL https://example.com/audio.mp3

    # SRT subtitles output
    uv run $URL -f srt -o subtitles video.mp4

    # With chunking at silences (for long audio)
    uv run $URL -c long_audio.mp3

    # With transcription prompt
    uv run $URL --prompt "Technical terms: API, SDK" audio.mp3

    # Or run locally:
    uv run whisper.py [OPTIONS] INPUT [INPUT...]

OPTIONS:
    -o, --output PATH       Output directory (default: input filename or 'whisper_output')
    -l, --language TEXT     Language code (default: en)
    -m, --model TEXT        Model name/path (default: auto by language)
    -f, --format            Output format: txt, srt, json, lrc (default: txt)
    -c, --chunk             Enable chunking at silences
    -r, --realtime          Show realtime output (default: on if TTY, off if piped)
    -R, --no-realtime       Disable realtime output
    -v, --verbose           Verbose output
    --prompt TEXT           Initial transcription prompt
    --max-length INT        Max segment length in characters

MODELS:
    Standard models (auto-download via pywhispercpp):
        tiny, base, small, medium, large-v1, large-v2, large-v3, large-v3-turbo

    Breeze models for Chinese (auto-download from HuggingFace):
        breeze      - 3.09 GB (full precision)
        breeze-q8   - 1.66 GB (default for zh, good accuracy/size balance)
        breeze-q5   - 1.08 GB
        breeze-q4   - 889 MB (smallest)

    Custom model: provide absolute path to .bin file

REQUIREMENTS:
    - ffmpeg: brew install ffmpeg (macOS) / apt install ffmpeg (Linux)

NOTES:
    - First run downloads models (~1.5-3GB depending on model)
    - Models are cached in ~/Library/Application Support/pywhispercpp/models/
    - zh* languages auto-select breeze-q8, others use large-v3-turbo
    - Supports: mp3, mp4, m4a, wav, webm, mov, m3u8, and HTTP(S) URLs
"""
import json
import sys
import tempfile
from pathlib import Path

import click
import httpx
from huggingface_hub import hf_hub_download
from pydub import AudioSegment
from pydub.silence import detect_silence
from pywhispercpp.constants import MODELS_DIR
from pywhispercpp.model import Model

# HuggingFace repository for Breeze ASR model
BREEZE_REPO = "alan314159/Breeze-ASR-25-whispercpp"
BREEZE_VARIANTS = {
    "breeze": "ggml-model.bin",  # 3.09 GB full precision
    "breeze-q8": "ggml-model-q8_0.bin",  # 1.66 GB (default for zh)
    "breeze-q5": "ggml-model-q5_k.bin",  # 1.08 GB
    "breeze-q4": "ggml-model-q4_k.bin",  # 889 MB
}

# Default models by language
DEFAULT_MODEL_ZH = "breeze-q8"
DEFAULT_MODEL_OTHER = "large-v3-turbo"


def log(message: str, verbose: bool = True, level: str = "INFO"):
    """Log message if verbose or if error."""
    if verbose or level == "ERROR":
        click.echo(f"[{level}] {message}", err=(level == "ERROR"))


def resolve_model_path(model_arg: str | None, language: str, verbose: bool = False) -> str:
    """
    Resolve model argument to a path.

    Resolution order:
    1. If model_arg is absolute path → use directly
    2. If model_arg is None → pick default by language
    3. If model_arg is breeze variant → download from HuggingFace if needed
    4. Otherwise → return model name for pywhispercpp (auto-download)
    """
    # Determine effective model name
    if model_arg is None:
        lang_lower = language.lower()
        if lang_lower.startswith("zh"):
            model_arg = DEFAULT_MODEL_ZH
            log(f"Auto-selected model '{model_arg}' for language '{language}'", verbose)
        else:
            model_arg = DEFAULT_MODEL_OTHER
            log(f"Auto-selected model '{model_arg}' for language '{language}'", verbose)

    # Absolute path: use directly
    if Path(model_arg).is_absolute():
        if not Path(model_arg).exists():
            raise click.UsageError(f"Model file not found: {model_arg}")
        log(f"Using custom model: {model_arg}", verbose)
        return model_arg

    # Breeze variant: download from HuggingFace
    if model_arg in BREEZE_VARIANTS:
        filename = BREEZE_VARIANTS[model_arg]
        cache_dir = Path(MODELS_DIR) / "breeze"
        cache_dir.mkdir(parents=True, exist_ok=True)

        local_path = cache_dir / filename
        if local_path.exists():
            log(f"Using cached breeze model: {local_path}", verbose)
            return str(local_path)

        log(f"Downloading {model_arg} from HuggingFace ({filename})...", verbose)
        downloaded_path = hf_hub_download(
            repo_id=BREEZE_REPO,
            filename=filename,
            local_dir=cache_dir,
        )
        log(f"Model downloaded to: {downloaded_path}", verbose)
        return downloaded_path

    # Standard whisper model: return name for pywhispercpp
    log(f"Using standard model: {model_arg}", verbose)
    return model_arg


def download_if_url(input_path: str, temp_dir: Path, verbose: bool = False) -> Path:
    """Download URL to temp file if input is HTTP(S) URL, otherwise return as Path."""
    if input_path.startswith(("http://", "https://")):
        # Extract filename from URL
        url_path = input_path.split("?")[0]
        filename = Path(url_path).name or "downloaded_audio"
        output_file = temp_dir / filename

        log(f"Downloading {input_path}...", verbose)
        with httpx.Client(follow_redirects=True, timeout=300.0) as client:
            response = client.get(input_path)
            response.raise_for_status()
            output_file.write_bytes(response.content)

        log(f"Downloaded to: {output_file}", verbose)
        return output_file

    path = Path(input_path)
    if not path.exists():
        raise click.UsageError(f"Input file not found: {input_path}")
    return path


def convert_to_wav(input_path: Path, output_path: Path, verbose: bool = False) -> Path:
    """Convert any audio file to 16kHz mono WAV using pydub."""
    log(f"Converting {input_path.name} to WAV...", verbose)

    audio = AudioSegment.from_file(str(input_path))
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(str(output_path), format="wav")

    log(f"Converted to: {output_path}", verbose)
    return output_path


def concatenate_audio(input_files: list[Path], output_path: Path, verbose: bool = False) -> Path:
    """Concatenate multiple audio files into one WAV."""
    log(f"Concatenating {len(input_files)} audio files...", verbose)

    combined = AudioSegment.empty()
    for i, input_file in enumerate(input_files, 1):
        log(f"  Loading file {i}/{len(input_files)}: {input_file.name}", verbose)
        audio = AudioSegment.from_file(str(input_file))
        combined += audio

    # Convert to 16kHz mono
    combined = combined.set_frame_rate(16000).set_channels(1)
    combined.export(str(output_path), format="wav")

    log(f"Concatenated to: {output_path}", verbose)
    return output_path


def detect_silences_in_audio(
    audio_path: Path, min_silence_len: int = 3000, silence_thresh: int = -30, verbose: bool = False
) -> list[tuple[int, int]]:
    """Detect silence points for chunking."""
    log("Detecting silence points...", verbose)

    audio = AudioSegment.from_file(str(audio_path))
    silences = detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    log(f"Found {len(silences)} silence points", verbose)
    return silences


def chunk_audio_at_silences(
    audio_path: Path, output_dir: Path, verbose: bool = False
) -> list[Path]:
    """Split audio at silence points and return list of chunk paths."""
    audio = AudioSegment.from_file(str(audio_path))
    silences = detect_silences_in_audio(audio_path, verbose=verbose)

    chunks = []
    start = 0

    for i, (silence_start, silence_end) in enumerate(silences):
        # Use the middle of the silence as the split point
        split_point = (silence_start + silence_end) // 2
        if split_point > start:
            chunk = audio[start:split_point]
            chunk_path = output_dir / f"chunk_{i:03d}.wav"
            chunk.export(str(chunk_path), format="wav")
            chunks.append(chunk_path)
            log(f"  Created chunk {i}: {start}ms - {split_point}ms", verbose)
        start = split_point

    # Final chunk
    if start < len(audio):
        chunk = audio[start:]
        chunk_path = output_dir / f"chunk_{len(chunks):03d}.wav"
        chunk.export(str(chunk_path), format="wav")
        chunks.append(chunk_path)
        log(f"  Created final chunk: {start}ms - {len(audio)}ms", verbose)

    log(f"Created {len(chunks)} chunks", verbose)
    return chunks


def _format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm for realtime display."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _make_realtime_callback():
    """Create a callback that prints segments as they're transcribed."""
    def callback(segment):
        start = _format_timestamp(segment.t0 / 100.0)
        end = _format_timestamp(segment.t1 / 100.0)
        text = segment.text.strip()
        if text:
            click.echo(f"[{start} --> {end}]  {text}")
    return callback


def transcribe(
    audio_path: Path,
    model_path: str,
    language: str,
    prompt: str | None = None,
    max_length: int = 0,
    verbose: bool = False,
    realtime: bool = False,
) -> list[dict]:
    """Transcribe audio file using pywhispercpp."""
    log(f"Transcribing {audio_path.name}...", verbose)

    # Build model parameters
    model_params = {
        "print_progress": verbose and not realtime,
        "print_realtime": False,
    }

    # Load model
    model = Model(model_path, **model_params)

    # Set transcription parameters
    transcribe_params = {
        "language": language,
    }

    if prompt:
        transcribe_params["initial_prompt"] = prompt
    if max_length > 0:
        transcribe_params["max_len"] = max_length
        transcribe_params["split_on_word"] = True
    if realtime:
        transcribe_params["new_segment_callback"] = _make_realtime_callback()

    # Transcribe
    segments = model.transcribe(str(audio_path), **transcribe_params)

    # Convert to list of dicts
    result = []
    for seg in segments:
        result.append({
            "start": seg.t0 / 100.0,  # Convert centiseconds to seconds
            "end": seg.t1 / 100.0,
            "text": seg.text.strip(),
        })

    log(f"Transcribed {len(result)} segments", verbose)
    return result


def format_txt(segments: list[dict]) -> str:
    """Format segments as plain text."""
    return "\n".join(seg["text"] for seg in segments if seg["text"])


def format_srt(segments: list[dict]) -> str:
    """Format segments as SRT subtitles."""
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


def _seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_json(segments: list[dict]) -> str:
    """Format segments as JSON."""
    return json.dumps({"segments": segments}, indent=2, ensure_ascii=False)


def format_lrc(segments: list[dict]) -> str:
    """Format segments as LRC lyrics."""
    lines = []
    for seg in segments:
        if not seg["text"]:
            continue
        time_str = _seconds_to_lrc_time(seg["start"])
        lines.append(f"{time_str}{seg['text']}")
    return "\n".join(lines)


def _seconds_to_lrc_time(seconds: float) -> str:
    """Convert seconds to LRC time format [MM:SS.xx]."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"[{minutes:02d}:{secs:05.2f}]"


def format_output(segments: list[dict], fmt: str) -> str:
    """Format transcription segments to specified output format."""
    formatters = {
        "txt": format_txt,
        "srt": format_srt,
        "json": format_json,
        "lrc": format_lrc,
    }
    formatter = formatters.get(fmt, format_txt)
    return formatter(segments)


@click.command()
@click.argument("inputs", nargs=-1, required=True)
@click.option("-o", "--output", default=None, help="Output directory")
@click.option("-l", "--language", default="en", help="Language code (default: en)")
@click.option("-m", "--model", default=None, help="Model name or path (default: auto by language)")
@click.option(
    "-f",
    "--format",
    "output_format",
    type=click.Choice(["txt", "srt", "json", "lrc"]),
    default="txt",
    help="Output format (default: txt)",
)
@click.option("-c", "--chunk", is_flag=True, help="Enable chunking at silences")
@click.option("-r/-R", "--realtime/--no-realtime", default=None, help="Show realtime output (default: auto-detect TTY)")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.option("--prompt", default=None, help="Initial transcription prompt")
@click.option("--max-length", default=0, type=int, help="Max segment length in characters")
@click.version_option(version="0.1.0")
def cli(
    inputs: tuple[str, ...],
    output: str | None,
    language: str,
    model: str | None,
    output_format: str,
    chunk: bool,
    realtime: bool | None,
    verbose: bool,
    prompt: str | None,
    max_length: int,
):
    """Transcribe audio files using Whisper.

    INPUTS can be file paths or HTTP(S) URLs. Multiple inputs are concatenated.

    \b
    Examples:
        whisper.py audio.mp3
        whisper.py -l zh chinese_audio.mp3
        whisper.py -f srt -o subtitles video.mp4
        whisper.py part1.mp3 part2.mp3 part3.mp3
        whisper.py https://example.com/audio.mp3
    """
    # Auto-detect realtime mode based on TTY
    if realtime is None:
        realtime = sys.stdout.isatty()

    # Resolve model path
    model_path = resolve_model_path(model, language, verbose)

    # Determine output directory
    if output is None:
        if len(inputs) == 1 and not inputs[0].startswith(("http://", "https://")):
            # Use input filename without extension
            output = Path(inputs[0]).stem
        else:
            output = "whisper_output"
        log(f"Using output directory: {output}", verbose)

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create temp directory for processing
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        # Download URLs and collect input files
        input_files = []
        for input_path in inputs:
            local_path = download_if_url(input_path, temp_dir, verbose)
            input_files.append(local_path)

        # Convert/concatenate to single WAV
        working_wav = temp_dir / "working_audio.wav"

        if len(input_files) == 1:
            convert_to_wav(input_files[0], working_wav, verbose)
        else:
            concatenate_audio(input_files, working_wav, verbose)

        # Transcribe
        all_segments = []

        if chunk:
            # Chunk mode: split at silences and transcribe each chunk
            chunks_dir = temp_dir / "chunks"
            chunks_dir.mkdir()

            chunk_files = chunk_audio_at_silences(working_wav, chunks_dir, verbose)

            for chunk_file in chunk_files:
                segments = transcribe(
                    chunk_file, model_path, language, prompt, max_length, verbose, realtime
                )
                all_segments.extend(segments)
        else:
            # Full file transcription
            all_segments = transcribe(
                working_wav, model_path, language, prompt, max_length, verbose, realtime
            )

        # Format and save output
        formatted = format_output(all_segments, output_format)
        output_file = output_dir / f"transcription.{output_format}"
        output_file.write_text(formatted, encoding="utf-8")

        log(f"Transcription saved to: {output_file}", verbose)

        if not verbose:
            click.echo(str(output_file))


if __name__ == "__main__":
    cli()
