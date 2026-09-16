# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "transformers==5.4.0",
#     "mlx-audio==0.4.2",
#     "click==8.3.1",
#     "numpy==2.4.4",
#     "soundfile==0.13.1",
#     "librosa==0.11.0",
#     "opencc-python-reimplemented==0.1.7",
# ]
# ///
"""
Qwen3-TTS CLI - Text-to-Speech for Apple Silicon using MLX.

Inspired by https://simonwillison.net/2026/Jan/22/qwen3-tts/

USAGE:
    # Run directly from GitHub (no clone needed):
    URL=https://raw.githubusercontent.com/CJHwong/toolkit/main/tts/qwen3_tts.py

    # Design a voice from description
    uv run $URL design 'I am a pirate, give me your gold!' -i 'gruff voice' -o pirate.wav

    # Clone a voice from reference audio
    uv run $URL clone 'Hello from the other side' -r voice.wav -t 'original transcript'

    # Use a preset speaker
    uv run $URL speak 'Good morning everyone!' -s Ryan

    # Use a preset speaker with style instruction
    uv run $URL speak 'I am so excited!' -s Aiden -i 'very happy and energetic'

    # Use faster 0.6B model
    uv run $URL speak --small 'Quick test' -s Vivian

    # Or run locally:
    uv run qwen3_tts.py <command> [options] "text to speak"

COMMANDS:
    clone   - Clone a voice from reference audio
    design  - Create a voice from text description
    speak   - Use a preset speaker voice

EXAMPLES:
    # Voice cloning (requires reference audio + transcript)
    uv run qwen3_tts.py clone -r voice.wav -t "transcript" "New text to speak"
    uv run qwen3_tts.py clone -r voice.wav -t "$(cat transcript.txt)" "Hello"

    # Voice design (describe the voice you want)
    uv run qwen3_tts.py design -i "warm female voice" "Hello world"
    uv run qwen3_tts.py design -i "deep male voice, slow pace" "Welcome"

    # Preset speakers (Chinese: Vivian, Serena, Uncle_Fu, Dylan, Eric | English: Ryan, Aiden)
    uv run qwen3_tts.py speak "Hello world"
    uv run qwen3_tts.py speak -s Vivian -l Chinese "你好"
    uv run qwen3_tts.py speak -i "speak angrily" -s Ryan "Stop doing that!"
    uv run qwen3_tts.py speak -i "whisper" -s Serena "This is a secret"
    uv run qwen3_tts.py speak --small "Quick test with 0.6B model"

OPTIONS:
    -o, --output    Output filename (default: output.wav, auto-increments)
    -l, --language  Language: Auto, English, Chinese, Japanese, Korean, etc. (default: Auto)
    -v, --verbose   Show progress details
    --small         Use faster 0.6B model (clone/speak only)
    --no-clean      Keep chat shortcodes, emoji and Markdown markers as written
    --no-convert    Keep Traditional Chinese as written (clone/speak only)
    --interval-silence  Silence between split segments in ms (default: 200)

SPEAKERS:
    Chinese: Vivian, Serena, Uncle_Fu, Dylan, Eric
    English: Ryan, Aiden

NOTES:
    - First run downloads models (~3GB for 1.7B, ~1GB for 0.6B)
    - Reference audio is automatically converted to mono 24kHz
    - Supports piped input: echo "text" | uv run qwen3_tts.py design
    - Traditional Chinese is mispronounced. One reference clip, one model,
      one variable: 颱風假 came out 圈封夾, 綜合 came out 沖派, 續聘 came out
      助聘. Converted with opencc tw2s, all three read correctly. clone and
      speak convert by default; pass --no-convert to send the text untouched.
      tw2s changes characters only, so Taiwan vocabulary survives (軟體 stays
      軟體, 批次 stays 批次). All three TTS engines use tw2s.
    - Chat shortcodes, emoji and Markdown markers are stripped before
      synthesis because none of them can be spoken. Punctuation that carries
      meaning is kept: 45,000~55,000 reads as 四万五千到五万五千 and 35% as
      百分之三十五. Pass --no-clean to keep the raw text.
    - Reference audio must be a format libsndfile reads. An .m4a (AAC) fails
      with "Format not recognised"; convert to WAV first.
    - Long text is segmented internally. generate() splits on newlines and
      yields one result per segment, and every segment is joined with
      --interval-silence between them. A single call over 1,294 characters
      used to truncate, losing the closing line entirely; paragraph breaks in
      the input now become segment boundaries, so keep them.
"""
import re
import sys
from pathlib import Path

import click
import numpy as np
import soundfile as sf


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


def resolve_output_path(output: str, auto_increment: bool = True) -> Path:
    """Resolve the output path, optionally auto-incrementing for default filenames."""
    output_path = Path(output)
    if auto_increment and output == "output.wav":
        output_path = get_unique_filename(output_path)
    return output_path


def save_audio(audio, sample_rate: int, output_path: Path, verbose: bool = False):
    """Save audio array to file."""
    sf.write(str(output_path), np.array(audio), sample_rate)
    if verbose:
        click.echo(f"Audio saved to: {output_path}")


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


CHINESE_LANGUAGES = {"auto", "chinese", "zh", "zh-cn", "zh-tw"}


def to_simplified(text: str) -> str:
    """Convert Traditional Chinese to the Simplified form the tokenizer knows.

    tw2s changes characters only, so Taiwan vocabulary survives as written
    (軟體 stays 軟體, 批次 stays 批次). Traditional input left as-is is
    mispronounced. Measured against one reference clip, same model, one
    variable: 颱風假 came out 圈封夾, 綜合 came out 沖派, 續聘 came out 助聘.
    All three read correctly once the text was converted.
    """
    import opencc

    return opencc.OpenCC("tw2s").convert(text)


SHORTCODE_PATTERN = re.compile(r":[A-Za-z0-9_+\-]{1,32}:")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # pictographs, emoticons, transport, supplemental
    "\U00002600-\U000027BF"  # miscellaneous symbols and dingbats
    "\U00002B00-\U00002BFF"  # miscellaneous symbols and arrows
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"             # zero width joiner
    "]"
)
MARKDOWN_PATTERN = re.compile(r"[*_`#>]+")


def clean_for_speech(text: str) -> str:
    """Drop markup that carries no speech, so the model cannot misread it.

    Removes chat shortcodes (":pray:"), emoji, and Markdown markers.
    Keeps the punctuation that carries prosody and meaning: 。，、：；！？「」（）
    plus digits, thousands separators, "~" and "%". Those read correctly as
    measured on a real run (45,000~55,000 spoke as 四万五千到五万五千, 35% as
    百分之三十五), so stripping them deletes meaning rather than noise.
    """
    text = SHORTCODE_PATTERN.sub(" ", text)
    text = EMOJI_PATTERN.sub("", text)
    text = MARKDOWN_PATTERN.sub(" ", text)
    text = text.replace("​", "").replace("﻿", "")
    text = "\n".join(line.strip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def prepare_text(
    text: str, language: str, no_clean: bool, no_convert: bool, verbose: bool
) -> str:
    """Clean markup, then convert script, before synthesis.

    Conversion is gated on language because tw2s rewrites Japanese kanji too,
    and on Han characters being present, without which it is a no-op.
    """
    if not no_clean:
        cleaned = clean_for_speech(text)
        if cleaned != text and verbose:
            preview = cleaned[:60] + ("..." if len(cleaned) > 60 else "")
            click.echo(f"Cleaned for speech: {preview}")
        text = cleaned

    if no_convert or language.strip().lower() not in CHINESE_LANGUAGES:
        return text
    if not any("㐀" <= c <= "䶿" or "一" <= c <= "鿿" for c in text):
        return text

    converted = to_simplified(text)
    if converted != text and verbose:
        preview = converted[:60] + ("..." if len(converted) > 60 else "")
        click.echo(f"Converted to Simplified: {preview}")
    return converted


def concatenate_segments(results: list, sample_rate: int, gap_ms: int):
    """Join the audio of every generated segment, with silence between them.

    `generate()` splits the text on `split_pattern` and yields one result per
    segment. Keeping only the first silently drops every later paragraph.
    """
    clips = [np.asarray(result.audio) for result in results]
    gap = np.zeros(int(sample_rate * gap_ms / 1000), dtype=clips[0].dtype)

    parts = []
    for index, clip in enumerate(clips):
        if index:
            parts.append(gap)
        parts.append(clip)
    return np.concatenate(parts)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Qwen3-TTS CLI - Text-to-Speech with voice design and cloning."""
    pass


@cli.command("design")
@click.argument("text", required=False)
@click.option("-o", "--output", default="output.wav", help="Output filename")
@click.option("-l", "--language", default="Auto", help="Language (Auto, English, Chinese, etc.)")
@click.option(
    "-i",
    "--instruct",
    default="",
    help="Voice instruction (e.g., 'warm female voice', 'deep male voice')",
)
@click.option("--interval-silence", type=int, default=200, help="Silence between split segments (ms)")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
def design_command(
    text: str | None,
    output: str,
    language: str,
    instruct: str,
    interval_silence: int,
    verbose: bool,
):
    """Generate speech using voice design (natural language voice description).

    Examples:

    \b
        qwen3_tts.py design "Hello world"
        qwen3_tts.py design -i "warm female voice" "Welcome to the show"
        qwen3_tts.py design -l Chinese "你好世界"
        echo "Piped text" | qwen3_tts.py design
    """
    from mlx_audio.tts.utils import load_model

    text = get_text_from_input(text)
    output_path = resolve_output_path(output)

    if verbose:
        click.echo("Loading VoiceDesign model...")

    model = load_model("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")

    if verbose:
        click.echo(f"Generating audio for: {text[:50]}{'...' if len(text) > 50 else ''}")
        if instruct:
            click.echo(f"Voice instruction: {instruct}")

    results = list(
        model.generate_voice_design(
            text=text,
            language=language,
            instruct=instruct,
            verbose=verbose,
        )
    )

    if verbose and len(results) > 1:
        click.echo(f"Joined {len(results)} segments with {interval_silence}ms silence")

    audio = concatenate_segments(results, model.sample_rate, interval_silence)
    save_audio(audio, model.sample_rate, output_path, verbose)

    if not verbose:
        click.echo(str(output_path))


@cli.command("clone")
@click.argument("text", required=False)
@click.option("-o", "--output", default="output.wav", help="Output filename")
@click.option("-l", "--language", default="Auto", help="Language (Auto, English, Chinese, etc.)")
@click.option(
    "-r",
    "--ref-audio",
    required=True,
    help="Path to reference audio for voice cloning",
)
@click.option(
    "-t",
    "--ref-text",
    required=True,
    help="Transcript of the reference audio",
)
@click.option("--small", is_flag=True, help="Use faster 0.6B model instead of 1.7B")
@click.option("--no-clean", is_flag=True, help="Keep shortcodes, emoji and Markdown markers")
@click.option("--no-convert", is_flag=True, help="Keep Traditional Chinese as written")
@click.option("--interval-silence", type=int, default=200, help="Silence between split segments (ms)")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
def clone_command(
    text: str | None,
    output: str,
    language: str,
    ref_audio: str,
    ref_text: str,
    small: bool,
    no_clean: bool,
    no_convert: bool,
    interval_silence: int,
    verbose: bool,
):
    """Clone a voice from reference audio and generate new speech.

    Examples:

    \b
        qwen3_tts.py clone -r voice.wav -t "Hello there" "New text to speak"
        qwen3_tts.py clone -r ref.wav -t "$(cat transcript.txt)" "Clone this"
        qwen3_tts.py clone --small -r voice.wav -t "Hi" "Faster with 0.6B model"
    """
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model

    text = prepare_text(
        get_text_from_input(text), language, no_clean, no_convert, verbose
    )
    output_path = resolve_output_path(output)

    # Validate reference audio
    ref_path = Path(ref_audio)
    if not ref_path.exists():
        raise click.UsageError(f"Reference audio file not found: {ref_audio}")

    model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-Base" if small else "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    if verbose:
        click.echo(f"Loading {'0.6B' if small else '1.7B'} Base model for voice cloning...")

    model = load_model(model_name)

    if verbose:
        click.echo(f"Reference audio: {ref_audio}")
        click.echo(f"Reference text: {ref_text}")
        click.echo(f"Generating audio for: {text[:50]}{'...' if len(text) > 50 else ''}")

    # Load reference audio
    import soundfile as sf_read
    ref_audio_data, ref_sr = sf_read.read(ref_audio)

    # Convert to mono if multi-channel
    if ref_audio_data.ndim > 1:
        ref_audio_data = ref_audio_data.mean(axis=1)
        if verbose:
            click.echo(f"Converted multi-channel audio to mono")

    if ref_sr != 24000:
        # Resample to 24kHz if needed
        import librosa
        ref_audio_data = librosa.resample(ref_audio_data, orig_sr=ref_sr, target_sr=24000)
        if verbose:
            click.echo(f"Resampled from {ref_sr}Hz to 24000Hz")

    ref_audio_mx = mx.array(ref_audio_data.astype(np.float32))

    # Use generate() which handles ICL voice cloning. It splits the text on
    # newlines and yields one result per segment, so every segment must be kept.
    results = list(
        model.generate(
            text=text,
            lang_code=language,
            ref_audio=ref_audio_mx,
            ref_text=ref_text,
            verbose=verbose,
        )
    )

    if verbose and len(results) > 1:
        click.echo(f"Joined {len(results)} segments with {interval_silence}ms silence")

    audio = concatenate_segments(results, model.sample_rate, interval_silence)
    save_audio(audio, model.sample_rate, output_path, verbose)

    if not verbose:
        click.echo(str(output_path))


@cli.command("speak")
@click.argument("text", required=False)
@click.option("-o", "--output", default="output.wav", help="Output filename")
@click.option("-l", "--language", default="Auto", help="Language (Auto, English, Chinese, etc.)")
@click.option(
    "-s",
    "--speaker",
    default="Ryan",
    help="Speaker: Chinese (Vivian, Serena, Uncle_Fu, Dylan, Eric) | English (Ryan, Aiden)",
)
@click.option(
    "-i",
    "--instruct",
    default="",
    help="Style instruction (e.g., 'speak angrily', 'very happy', 'slow and soft')",
)
@click.option("--small", is_flag=True, help="Use faster 0.6B model instead of 1.7B")
@click.option("--no-clean", is_flag=True, help="Keep shortcodes, emoji and Markdown markers")
@click.option("--no-convert", is_flag=True, help="Keep Traditional Chinese as written")
@click.option("--interval-silence", type=int, default=200, help="Silence between split segments (ms)")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
def speak_command(
    text: str | None,
    output: str,
    language: str,
    speaker: str,
    instruct: str,
    small: bool,
    no_clean: bool,
    no_convert: bool,
    interval_silence: int,
    verbose: bool,
):
    """Generate speech using a preset custom voice with optional style control.

    Available speakers:
        Chinese: Vivian, Serena, Uncle_Fu, Dylan, Eric
        English: Ryan, Aiden

    Examples:

    \b
        qwen3_tts.py speak "Hello world"
        qwen3_tts.py speak -s Aiden "Good morning everyone"
        qwen3_tts.py speak -s Vivian -l Chinese "你好"
        qwen3_tts.py speak -i "speak angrily" "I told you not to do that!"
        qwen3_tts.py speak -i "whisper softly" -s Serena "This is a secret"
        qwen3_tts.py speak --small "Quick test with 0.6B model"
    """
    from mlx_audio.tts.utils import load_model

    text = prepare_text(
        get_text_from_input(text), language, no_clean, no_convert, verbose
    )
    output_path = resolve_output_path(output)

    model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice" if small else "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    if verbose:
        click.echo(f"Loading {'0.6B' if small else '1.7B'} CustomVoice model...")

    model = load_model(model_name)

    if verbose:
        click.echo(f"Speaker: {speaker}")
        if instruct:
            click.echo(f"Style instruction: {instruct}")
        click.echo(f"Generating audio for: {text[:50]}{'...' if len(text) > 50 else ''}")

    results = list(
        model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            verbose=verbose,
        )
    )

    if verbose and len(results) > 1:
        click.echo(f"Joined {len(results)} segments with {interval_silence}ms silence")

    audio = concatenate_segments(results, model.sample_rate, interval_silence)
    save_audio(audio, model.sample_rate, output_path, verbose)

    if not verbose:
        click.echo(str(output_path))


if __name__ == "__main__":
    cli()
