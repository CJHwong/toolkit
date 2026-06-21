# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "funasr==1.3.11",
#     "torch==2.12.1",
#     "torchaudio==2.11.0",
#     "click==8.3.1",
#     "pydub==0.25.1",
#     "httpx==0.28.1",
# ]
# ///
"""
FunASR CLI - Audio transcription using FunASR (Paraformer) on CPU/MPS.

USAGE:
    # Basic transcription (Chinese + English code-switching)
    uv run funasr_asr.py audio.mp3

    # Force CPU (skip Apple GPU / MPS)
    uv run funasr_asr.py -d cpu audio.mp3

    # SRT subtitles output
    uv run funasr_asr.py -f srt -o subtitles.srt video.mp4

    # From URL
    uv run funasr_asr.py https://example.com/audio.mp3

    # JSON output with timing metadata
    uv run funasr_asr.py -f json audio.mp3

MODELS:
    funasr/paraformer-zh             220M, Chinese + English, timestamps (default)
    FunAudioLLM/Fun-ASR-Nano-2512    800M LLM ASR, 31 langs, code-switching (auto -t)

DEVICES:
    auto   try MPS (Apple GPU), fall back to CPU  (default)
    cpu    PyTorch CPU
    mps    Apple Silicon GPU via Metal (partial op coverage; CPU fallback enabled)
    cuda   NVIDIA GPU

REQUIREMENTS:
    - ffmpeg: brew install ffmpeg
    - First run downloads paraformer-zh + fsmn-vad + ct-punc (~1 GB) from HuggingFace
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import click
import httpx
from pydub import AudioSegment

DEFAULT_MODEL = "funasr/paraformer-zh"
VAD_MODEL = "funasr/fsmn-vad"
PUNC_MODEL = "funasr/ct-punc"


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


def resolve_device(device_arg: str, verbose: bool = False) -> str:
    """Resolve 'auto' to mps when available, else cpu. Enable MPS CPU fallback."""
    import torch

    if device_arg == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        log(f"Auto-selected device: {device}", verbose)
    else:
        device = device_arg

    if device == "mps":
        # Many FunASR ops lack MPS kernels; let them fall back to CPU instead of crashing.
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    return device


def _build_segments(item: dict, audio_duration: float) -> list[dict]:
    """Extract sentence-level segments from a FunASR result item."""
    segments = []
    for sentence in item.get("sentence_info", []) or []:
        text = (sentence.get("text") or "").strip()
        if not text:
            continue
        segments.append({
            "start": sentence.get("start", 0) / 1000.0,
            "end": sentence.get("end", 0) / 1000.0,
            "text": text,
        })
    # Fallback: single segment from full text (no sentence timestamps available)
    if not segments and (item.get("text") or "").strip():
        segments.append({"start": 0, "end": audio_duration, "text": item["text"].strip()})
    return segments


def _make_model(model_name: str, device: str, llm: bool, verbose: bool):
    """Build an AutoModel. LLM-style models (Fun-ASR-Nano) need remote code and
    self-punctuate; Paraformer needs a separate punctuation-restoration model."""
    from funasr import AutoModel

    kwargs = dict(
        model=model_name, vad_model=VAD_MODEL, hub="hf", device=device,
        disable_update=True, disable_pbar=not verbose,
    )
    if llm:
        kwargs["trust_remote_code"] = True
        kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}
    else:
        kwargs["punc_model"] = PUNC_MODEL
    return AutoModel(**kwargs)


def _generate(model, audio_path: Path, llm: bool, language: str | None):
    kwargs = {"batch_size_s": 300}
    if llm:
        kwargs["language"] = language or "zh"
        kwargs["itn"] = True  # write "1800" not "一千八百"
    return model.generate(input=str(audio_path), **kwargs)


def transcribe(
    audio_path: Path,
    model_name: str,
    device: str,
    language: str | None = None,
    llm: bool = False,
    verbose: bool = False,
) -> dict:
    """Transcribe audio and return result with timing metadata."""
    log(f"Loading model: {model_name} (device={device}, llm={llm})", verbose)
    load_start = time.monotonic()
    model = _make_model(model_name, device, llm, verbose)
    load_elapsed = time.monotonic() - load_start
    log(f"Model loaded in {load_elapsed:.1f}s", verbose)

    log(f"Transcribing {audio_path.name}...", verbose)
    infer_start = time.monotonic()
    try:
        result = _generate(model, audio_path, llm, language)
    except (NotImplementedError, RuntimeError, TypeError) as exc:
        if device == "cpu":
            raise
        # MPS lacks float64 (Paraformer's CIF predictor needs it) and some ops have
        # no Metal kernel. Fall back to CPU rather than die.
        log(f"{device} inference failed ({exc}); retrying on cpu", level="ERROR")
        model = _make_model(model_name, "cpu", llm, verbose)
        device = "cpu"
        result = _generate(model, audio_path, llm, language)
    infer_elapsed = time.monotonic() - infer_start

    audio_duration = get_audio_duration(audio_path)
    rtf = infer_elapsed / audio_duration if audio_duration > 0 else 0

    item = result[0] if result else {"text": ""}
    segments = _build_segments(item, audio_duration)

    return {
        "text": item.get("text", "").strip(),
        "segments": segments,
        "audio_duration": audio_duration,
        "load_time": load_elapsed,
        "infer_time": infer_elapsed,
        "rtf": rtf,
        "model": model_name,
        "device": device,
    }


def format_txt(segments: list[dict]) -> str:
    return "\n".join(seg["text"] for seg in segments if seg["text"])


def format_srt(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        if not seg["text"]:
            continue
        lines.append(f"{i}")
        lines.append(f"{_seconds_to_srt_time(seg['start'])} --> {_seconds_to_srt_time(seg['end'])}")
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
        lines.append(f"{_seconds_to_lrc_time(seg['start'])}{seg['text']}")
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
@click.option("-l", "--language", default=None, help="Language code (paraformer ignores it; LLM models use it, default zh)")
@click.option("-m", "--model", default=None, help=f"Model name (default: {DEFAULT_MODEL})")
@click.option("-d", "--device", default="auto", type=click.Choice(["auto", "cpu", "mps", "cuda"]), help="Compute device (default: auto)")
@click.option("-t/-T", "--trust-remote-code/--no-trust-remote-code", "trust_remote_code", default=None, help="Load model custom code (auto-on for *nano* models)")
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
    device: str,
    trust_remote_code: bool | None,
    output_format: str,
    verbose: bool,
):
    """Transcribe audio using FunASR (Paraformer or Fun-ASR-Nano).

    INPUTS can be file paths or HTTP(S) URLs. Multiple inputs are concatenated.

    \b
    Examples:
        funasr_asr.py audio.mp3
        funasr_asr.py -d cpu audio.mp3
        funasr_asr.py -m FunAudioLLM/Fun-ASR-Nano-2512 audio.mp3
        funasr_asr.py -f srt -o subtitles.srt video.mp4
    """
    model_name = model or DEFAULT_MODEL
    # LLM-style models (Fun-ASR-Nano) ship custom code; auto-enable unless overridden.
    llm = ("nano" in model_name.lower()) if trust_remote_code is None else trust_remote_code
    resolved_device = resolve_device(device, verbose)

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

        result = transcribe(wav_path, model_name, resolved_device, language, llm, verbose)

    if output is None:
        if len(inputs) == 1 and not inputs[0].startswith(("http://", "https://")):
            output_file = Path(inputs[0]).with_suffix(f".{output_format}")
        else:
            output_file = Path(f"funasr_output.{output_format}")
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
            f"[device={result['device']}] "
            f"[model={result['model']}]",
            err=True,
        )

    click.echo(str(output_file))


if __name__ == "__main__":
    cli()
