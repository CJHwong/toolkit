#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "index-tts-2.5-mlx==0.1.1",
#     "mlx==0.32.0",
#     "mlx-metal==0.32.0 ; sys_platform == 'darwin'",
#     "opencc-python-reimplemented==0.1.7",
#     "click==8.3.1",
# ]
# ///
# NOTE: index-tts-2.5-mlx declares its stack loosely ("mlx>=0.24",
# "transformers>=4.45"), so the MLX pair is pinned here to keep a year-old
# script resolving to the versions it was written against.
"""
IndexTTS-2.5 CLI - 22.05kHz zero-shot voice cloning on Apple Silicon (pure MLX).

Wraps index-tts-2.5-mlx, the torch-free MLX port of index-tts/index-tts
(IndexTTS-2.5, 0.8B AR GPT + CFM + BigVGAN). No PyTorch in the inference path;
runs natively on Metal. Cloning needs the reference clip only, no transcript.

Int8-quantized weights (~5GB) are auto-downloaded from
yunfengwang/IndexTTS-2.5-mlx into the Hugging Face cache on first run.

USAGE:
    # Run directly from GitHub (no clone needed):
    URL=https://raw.githubusercontent.com/CJHwong/toolkit/main/python/index_tts.py

    # Clone a voice (reference audio only, no transcript needed)
    # Traditional Chinese is converted to Simplified first, see NOTES
    uv run $URL "大家好，這是語音複製示範。" -r ref.wav -o clone.wav

    # English text (language must be set, the default is zh)
    uv run $URL "Hello, this is a voice cloning demo." -r ref.wav -l en

    # Reproducible output
    uv run $URL "Same every time." -r ref.wav -l en --greedy --seed 42

    # Slower delivery (0.5 - 2.0, higher is slower)
    uv run $URL "Take your time." -r ref.wav -l en --duration-factor 1.2

    # Long text is segmented internally, with silence between segments
    uv run $URL "$(cat long.txt)" -r ref.wav -l en --interval-silence 300

    # Or run locally:
    uv run index_tts.py [OPTIONS] "text to speak"

OPTIONS:
    -r, --ref-audio       Reference audio to clone, 15s or shorter (required)
    -o, --output          Output filename (default: output.wav, auto-increments)
    -l, --language        zh | en | ja | yue (default: zh)
    -m, --model-dir       Local weights dir (default: auto-download)
    --no-convert          Keep Traditional Chinese as written (-l zh only)
    --no-normalization    Skip wetext text normalization
    --greedy              Greedy decoding instead of sampling
    --seed                RNG seed (default: none, so output varies per run)
    --top-k               Sampling top-k (default: 30)
    --top-p               Sampling top-p (default: 0.8)
    --temperature         Sampling temperature (default: 0.8)
    --repetition-penalty  Repetition penalty (default: 10.0)
    --max-mel-tokens      Max acoustic tokens per segment (default: 1500)
    --max-text-tokens-per-segment  Segment size for long text (default: 120)
    --interval-silence    Silence between segments in ms (default: 200)
    --duration-factor     Pace, 0.5 - 2.0, higher is slower (default: 1.0)
    --n-timesteps         CFM steps (default: 25)
    --cfg-rate            CFM guidance rate (default: 0.7)
    -v, --verbose         Show progress details

NOTES:
    - Apple Silicon only (MLX is Metal-only). macOS 13 or later.
    - The first run downloads ~5GB into the Hugging Face cache. Point
      --model-dir at a local dir to skip this. Set HF_TOKEN for a gated repo.
    - Output quality is capped by reference quality. Clean speech of 5s to 15s
      works best; noise in the clip lands in the output, and a synthetic
      reference (macOS `say`) gives synthetic-sounding speech.
    - The reference carries timbre, not accent. A 台灣腔 reference still reads
      in mainland Mandarin, and a British reference still reads American. Pick
      the reference for voice, not for regional delivery.
    - This port drops the emotion control of upstream IndexTTS-2. Inline
      special tokens such as <|Laughter|> or [laugh] are not supported and
      give unpredictable results. Express emotion through the reference clip.
    - Language is not auto-detected. Pass -l en for English text, or the
      Chinese frontend mangles it.
    - The tokenizer is Simplified-only, so Traditional Chinese input comes out
      garbled. Under -l zh the text is converted with opencc tw2sp first, which
      also maps Taiwan vocabulary to its mainland form (軟體 -> 软件). Pass
      --no-convert to send the text through untouched. Only the characters
      change; the cloned voice and the accent are unaffected.
    - An abbreviation written with a slash is misread. "HB/L No." came back as
      "HVAC call number" and "HBALF L number" across four references. Write it
      without the slash ("HBL No.") and it reads correctly. --no-normalization
      does not help, so this is the model, not wetext.
    - --duration-factor is linear: measured 0.599 / 0.799 / 1.298 / 1.598 of
      the baseline length at 0.6 / 0.8 / 1.3 / 1.6, so a target duration is one
      shot. Unlike an ffmpeg atempo stretch, the model really speaks slower, so
      pitch stays natural. A higher factor also lowers RTF.
    - Speed is hardware bound. On the same 24.7s output, warm: RTF 1.38 on an
      M1 Pro, 0.56 on an M5 Pro. The first run after boot pays Metal kernel
      compilation (30.3s vs 13.8s on the same machine), so budget for a slow
      first item in a batch.
    - Supports piped input: echo "text" | uv run index_tts.py -r ref.wav -l en
"""
import platform
import sys
import time
from pathlib import Path

import click

LANGUAGES = ("zh", "en", "ja", "yue")


def require_apple_silicon() -> None:
    """Exit cleanly when MLX cannot run here."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise click.ClickException(
            "IndexTTS-2.5 MLX runs on Apple Silicon only "
            f"(found {platform.system()}/{platform.machine()})."
        )


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


def to_simplified(text: str) -> str:
    """Convert Traditional Chinese to the Simplified form the tokenizer knows.

    tw2sp also maps Taiwan vocabulary to its mainland form, which is what the
    model was trained on. Traditional characters left as-is come out garbled.
    """
    import opencc

    return opencc.OpenCC("tw2sp").convert(text)


@click.command()
@click.version_option(version="0.1.0")
@click.argument("text", required=False)
@click.option("-r", "--ref-audio", required=True, help="Reference audio to clone (<=15s)")
@click.option("-o", "--output", default="output.wav", help="Output filename")
@click.option("-l", "--language", type=click.Choice(LANGUAGES), default="zh", help="Text language")
@click.option("-m", "--model-dir", default=None, help="Local weights dir (default: auto-download)")
@click.option("--no-convert", is_flag=True, help="Keep Traditional Chinese as written (-l zh only)")
@click.option("--no-normalization", is_flag=True, help="Skip wetext text normalization")
@click.option("--greedy", is_flag=True, help="Greedy decoding instead of sampling")
@click.option("--seed", type=int, default=None, help="RNG seed (default: none)")
@click.option("--top-k", type=int, default=30, help="Sampling top-k")
@click.option("--top-p", type=float, default=0.8, help="Sampling top-p")
@click.option("--temperature", type=float, default=0.8, help="Sampling temperature")
@click.option("--repetition-penalty", type=float, default=10.0, help="Repetition penalty")
@click.option("--max-mel-tokens", type=int, default=1500, help="Max acoustic tokens per segment")
@click.option("--max-text-tokens-per-segment", type=int, default=120, help="Segment size for long text")
@click.option("--interval-silence", type=int, default=200, help="Silence between segments (ms)")
@click.option("--duration-factor", type=float, default=1.0, help="Pace, higher is slower")
@click.option("--n-timesteps", type=int, default=25, help="CFM steps")
@click.option("--cfg-rate", type=float, default=0.7, help="CFM guidance rate")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
def cli(
    text, ref_audio, output, language, model_dir, no_convert, no_normalization, greedy, seed,
    top_k, top_p, temperature, repetition_penalty, max_mel_tokens,
    max_text_tokens_per_segment, interval_silence, duration_factor, n_timesteps,
    cfg_rate, verbose,
):
    """Clone a voice from a reference clip (22.05kHz, pure MLX).

    Examples:

    \b
        index_tts.py "大家好" -r ref.wav
        index_tts.py "Hello world" -r ref.wav -l en -o hello.wav
        index_tts.py "Reproducible." -r ref.wav -l en --greedy --seed 42
        index_tts.py "Slow down." -r ref.wav -l en --duration-factor 1.3
        echo "Piped text" | index_tts.py -r ref.wav -l en
    """
    require_apple_silicon()

    text = get_text_from_input(text)
    output_path = resolve_output_path(output)

    if language == "zh" and not no_convert:
        converted = to_simplified(text)
        if converted != text and verbose:
            click.echo(f"Converted to Simplified: {converted[:60]}{'...' if len(converted) > 60 else ''}")
        text = converted

    ref_path = Path(ref_audio).expanduser()
    if not ref_path.exists():
        raise click.UsageError(f"Reference audio file not found: {ref_audio}")

    if model_dir is not None:
        weights = Path(model_dir).expanduser()
        if not weights.exists():
            raise click.UsageError(f"Model directory not found: {model_dir}")
        model_dir = str(weights)

    from index_tts_2_5_mlx import IndexTTS

    if verbose:
        click.echo("Loading IndexTTS-2.5 (first run downloads ~5GB)...")

    tts = IndexTTS(model_dir=model_dir, use_normalization=not no_normalization)

    if verbose:
        click.echo(
            f"Generating: {text[:60]}{'...' if len(text) > 60 else ''}"
            f" | ref: {ref_audio} | lang: {language}"
        )

    started = time.perf_counter()
    pcm = tts.synthesize(
        text,
        lang=language,
        ref_audio_path=str(ref_path),
        greedy=greedy,
        seed=seed,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_mel_tokens=max_mel_tokens,
        max_text_tokens_per_segment=max_text_tokens_per_segment,
        interval_silence=interval_silence,
        duration_factor=duration_factor,
        n_timesteps=n_timesteps,
        cfg_rate=cfg_rate,
    )
    elapsed = time.perf_counter() - started

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tts.write_wav(str(output_path), pcm)

    if verbose:
        seconds = len(pcm) / tts.sample_rate
        click.echo(
            f"Audio saved to: {output_path} "
            f"({seconds:.2f}s @ {tts.sample_rate}Hz, {elapsed:.1f}s wall, "
            f"RTF {elapsed / seconds:.2f})"
        )
    else:
        click.echo(str(output_path))


if __name__ == "__main__":
    cli()
