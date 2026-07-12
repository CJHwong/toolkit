# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "dots-tts-mlx @ git+https://github.com/sb1992/dots-tts-mlx.git@v0.7.0",
#     "mlx==0.32.0",
#     "mlx-metal==0.32.0 ; sys_platform == 'darwin'",
#     "mlx-lm==0.31.3",
#     "click==8.3.1",
# ]
# ///
# NOTE: mlx-lm is pinned because dots-tts-mlx declares it bare ("mlx-lm"), and a
# loose resolve pulls mlx-lm 0.0.3 (no mlx_lm.models.cache), which breaks the
# port's import. Pin the MLX stack to keep a year-old script runnable.
"""
dots.tts CLI - 48kHz zero-shot voice cloning on Apple Silicon (pure MLX).

Wraps sb1992/dots-tts-mlx, the clean-room MLX port of rednote-hilab/dots.tts
(2B AR flow-matching TTS, 48kHz AudioVAE). No PyTorch in the inference path;
runs natively on Metal. Cloning is in-context: pass a reference clip plus its
exact transcript and the model continues in that voice.

Pre-quantized weights are auto-downloaded from shraey/dots-tts-mlx on first
run (int4 ~2.4GB by default), so you do not need to fetch or convert anything.

USAGE:
    # Run directly from GitHub (no clone needed):
    URL=https://raw.githubusercontent.com/CJHwong/toolkit/main/python/dots_tts.py

    # Clone a voice (reference audio + its exact transcript)
    uv run $URL "Hello, this is a voice cloning demo." \
        -r ref.wav -t "Exact transcript of ref.wav" -o clone.wav

    # Faster MeanFlow decoder (~2x), 4-step
    uv run $URL --variant mf-int4 "Quick clone." -r ref.wav -t "transcript"

    # Force a language tag (uppercase ISO code; no auto_detect on this port)
    uv run $URL "你好世界" -r ref.wav -t "transcript" -l ZH -o zh.wav

    # Long-form: sentence-chunked so long/multilingual text does not truncate
    uv run $URL --long "$(cat long.txt)" -r ref.wav -t "transcript"

    # Or run locally:
    uv run dots_tts.py [OPTIONS] "text to speak"

OPTIONS:
    -r, --ref-audio        Reference audio (required)
    -t, --ref-text         Exact transcript of the reference audio (required)
    -m, --model            Local converted weights dir (default: auto-download)
    --variant              int4 | int8 | mf-int4 | mf-int8 (default: int4)
    -o, --output           Output filename (default: output.wav, auto-increments)
    -l, --language         Uppercase ISO code (EN/ZH/DE/ES/FR/HI...). Default: none
    -n, --num-steps        Flow-matching steps (default: auto, 4 for mf / 10 for soar)
    -g, --guidance-scale   CFG scale, soar only (default: 1.2; ignored by mf)
    --speaker-scale        Speaker embedding scale (default: 1.5)
    --seed                 RNG seed (default: 42)
    --max-generate-length  Max total audio patch count (default: 500)
    --long                 Sentence-chunked long-form generation
    --gap-ms               Silence between chunks in --long mode (default: 80)
    --speed                Pitch-preserving tempo via ffmpeg atempo (default: 1.0)
    --no-trim-onset        Keep the raw vocoder onset (trim is on by default)
    -v, --verbose          Show progress details

VARIANTS:
    int4      soar decoder, ~2.4GB (default, best quality)
    int8      soar decoder, ~3.1GB (conservative)
    mf-int4   MeanFlow decoder, ~2.4GB (~2x faster, 4-step, CFG fused)
    mf-int8   MeanFlow decoder, ~3.1GB

NOTES:
    - Apple Silicon only (MLX is Metal-only).
    - First run downloads the chosen variant from Hugging Face into
      ~/.cache/dots-tts-mlx/<variant>. Point --model at a local dir to skip this.
    - --speed needs ffmpeg installed.
    - This port does not support x-vector-only or no-reference sampling; a
      reference clip plus its transcript is always required.
    - Supports piped input: echo "text" | uv run dots_tts.py -r ref.wav -t "..."
"""
import sys
from pathlib import Path

import click

HF_REPO = "shraey/dots-tts-mlx"
VARIANTS = ("int4", "int8", "mf-int4", "mf-int8")


def get_unique_filename(base_path: Path) -> Path:
    """Return a unique filename, adding -2, -3, etc. if the file already exists."""
    if not base_path.exists():
        return base_path

    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent

    counter = 2
    while True:
        new_path = parent / f"{stem}-{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


def resolve_output_path(output: str) -> Path:
    """Resolve the output path, auto-incrementing the default filename."""
    output_path = Path(output)
    if output == "output.wav":
        output_path = get_unique_filename(output_path)
    return output_path


def get_text_from_input(text: str | None) -> str:
    """Get text from argument or stdin."""
    if text is None:
        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()
        else:
            raise click.UsageError(
                "No text provided. Pass text as an argument or pipe it via stdin."
            )

    if not text:
        raise click.UsageError("Text cannot be empty.")

    return text


def ensure_model(model: str | None, variant: str, verbose: bool) -> Path:
    """Return a local weights dir, auto-downloading the variant if none given."""
    if model:
        path = Path(model).expanduser()
        if not path.exists():
            raise click.UsageError(f"Model directory not found: {model}")
        return path

    from huggingface_hub import snapshot_download

    cache = Path.home() / ".cache" / "dots-tts-mlx" / variant
    model_dir = cache / variant
    if (model_dir / "config.json").exists():
        if verbose:
            click.echo(f"Using cached weights: {model_dir}")
        return model_dir

    if verbose:
        click.echo(f"Downloading {variant} weights from {HF_REPO}...")
    snapshot_download(
        HF_REPO,
        allow_patterns=[f"{variant}/*"],
        local_dir=str(cache),
    )
    if not (model_dir / "config.json").exists():
        raise click.FileError(
            str(model_dir),
            f"Download did not produce {variant}/config.json under {cache}",
        )
    return model_dir


def apply_speed(wav_path: Path, speed: float) -> None:
    """Pitch-preserving time-stretch via ffmpeg atempo (chained for extremes)."""
    import subprocess

    factors, t = [], float(speed)
    while t > 2.0:
        factors.append(2.0)
        t /= 2.0
    while t < 0.5:
        factors.append(0.5)
        t /= 0.5
    factors.append(t)
    chain = ",".join(f"atempo={f:.6f}" for f in factors)
    tmp = wav_path.with_suffix(".tmp.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(wav_path), "-filter:a", chain, str(tmp)],
        check=True,
    )
    tmp.replace(wav_path)


@click.command()
@click.version_option(version="0.1.0")
@click.argument("text", required=False)
@click.option("-r", "--ref-audio", required=True, help="Path to reference audio")
@click.option("-t", "--ref-text", required=True, help="Exact transcript of the reference audio")
@click.option("-m", "--model", default=None, help="Local converted weights dir (default: auto-download)")
@click.option("--variant", type=click.Choice(VARIANTS), default="int4", help="Weights variant to auto-download")
@click.option("-o", "--output", default="output.wav", help="Output filename")
@click.option("-l", "--language", default=None, help="Uppercase ISO code (EN/ZH/...); default none")
@click.option("-n", "--num-steps", type=int, default=None, help="Flow-matching steps (auto by default)")
@click.option("-g", "--guidance-scale", type=float, default=1.2, help="CFG scale (soar only)")
@click.option("--speaker-scale", type=float, default=1.5, help="Speaker embedding scale")
@click.option("--seed", type=int, default=42, help="RNG seed")
@click.option("--max-generate-length", type=int, default=500, help="Max total audio patch count")
@click.option("--long", "long_mode", is_flag=True, help="Sentence-chunked long-form generation")
@click.option("--gap-ms", type=int, default=80, help="Silence between chunks in --long mode")
@click.option("--speed", type=float, default=1.0, help="Pitch-preserving tempo via ffmpeg atempo")
@click.option("--no-trim-onset", is_flag=True, help="Keep the raw vocoder onset (trim is on by default)")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
def cli(
    text, ref_audio, ref_text, model, variant, output, language, num_steps,
    guidance_scale, speaker_scale, seed, max_generate_length, long_mode, gap_ms,
    speed, no_trim_onset, verbose,
):
    """Clone a voice from reference audio + its transcript (48kHz, pure MLX).

    Examples:

    \b
        dots_tts.py "Hello world" -r ref.wav -t "transcript of ref.wav"
        dots_tts.py --variant mf-int4 "Faster clone." -r ref.wav -t "transcript"
        dots_tts.py "你好世界" -r ref.wav -t "transcript" -l ZH -o zh.wav
        dots_tts.py --long "$(cat long.txt)" -r ref.wav -t "transcript"
        echo "Piped text" | dots_tts.py -r ref.wav -t "transcript"
    """
    import shutil

    import mlx.core as mx
    import numpy as np
    import soundfile as sf

    from dots_tts_mlx.loader import from_pretrained

    mx.set_memory_limit(int(45 * (1 << 30)))  # memory ceiling, set before heavy alloc

    text = get_text_from_input(text)
    output_path = resolve_output_path(output)

    ref_path = Path(ref_audio)
    if not ref_path.exists():
        raise click.UsageError(f"Reference audio file not found: {ref_audio}")

    if abs(speed - 1.0) > 1e-3 and not shutil.which("ffmpeg"):
        raise click.UsageError("--speed requires ffmpeg installed")

    model_dir = ensure_model(model, variant, verbose)

    if verbose:
        click.echo(f"Loading model: {model_dir}")

    model = from_pretrained(str(model_dir), dtype=mx.bfloat16).model

    if verbose:
        click.echo(
            f"Generating: {text[:60]}{'...' if len(text) > 60 else ''}"
            f" | ref: {ref_audio} | lang: {language or 'none'}"
        )

    gen = model.generate_long if long_mode else model.generate
    extra = {"gap_ms": gap_ms} if long_mode else {}
    out = gen(
        text=text,
        prompt_audio=ref_audio,
        prompt_text=ref_text,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        speaker_scale=speaker_scale,
        language=language,
        seed=seed,
        max_generate_length=max_generate_length,
        trim_onset=not no_trim_onset,
        streaming_decode=True,
        **extra,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wav = np.asarray(out["audio"].astype(mx.float32)).ravel()
    sr = int(out["sample_rate"])
    sf.write(str(output_path), wav, sr)

    if abs(speed - 1.0) > 1e-3:
        apply_speed(output_path, speed)
        if verbose:
            click.echo(f"Applied --speed {speed}")

    if verbose:
        peak = mx.get_peak_memory() / (1 << 30)
        click.echo(f"Audio saved to: {output_path} ({wav.shape[-1] / sr:.2f}s @ {sr}Hz, MLX peak {peak:.2f}GB)")
    else:
        click.echo(str(output_path))


if __name__ == "__main__":
    cli()