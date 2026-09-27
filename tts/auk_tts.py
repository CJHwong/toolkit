# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "auk @ git+https://github.com/Tencent-Hunyuan/AuK.git@6943a1e967409e8c73139a7a345f2a611cfb3dd6",
#     "mlx==0.32.2",
#     "mlx-metal==0.32.2 ; sys_platform == 'darwin'",
#     # upstream leaves transformers at >=4.52,<5 and validated the MLX port on 4.57
#     "transformers==4.57.6",
#     "huggingface-hub==0.35.3",
#     "click==8.3.1",
#     "numpy==2.2.6",
#     "soundfile==0.13.1",
#     "soxr==1.1.0",
#     # qwen-omni-utils imports audioread but declares only a bare librosa;
#     # librosa 1.0 dropped audioread, so a loose resolve fails on any audio input
#     "qwen-omni-utils==0.0.9",
#     "librosa==0.11.0",
#     "av==17.1.0",
#     "opencc-python-reimplemented==0.1.7",
# ]
# ///
"""
AuK CLI - speech generation and editing for Apple Silicon using MLX.

AuK (Tencent Hunyuan, MIT) is one model for three jobs: clone a voice, design
a voice from a description, and edit existing audio by instruction (emotion,
pitch, speed, words, accent, denoise, separation).

USAGE:
    # Run directly from GitHub (no clone needed):
    URL=https://raw.githubusercontent.com/CJHwong/toolkit/main/tts/auk_tts.py

    # Clone a voice (reference audio only, no transcript needed)
    uv run $URL clone "大家好，這是語音複製示範。" -r ref.wav -o clone.wav

    # Design a voice from a description, no reference
    uv run $URL design "Welcome to the evening news." -i "a calm male news anchor in his forties"

    # Edit existing audio by instruction
    uv run $URL edit -r clone.wav -i "将情感转变为愤怒。" -o angry.wav
    uv run $URL edit -r take.wav -i "Replace 'Staging' with 'Production'." -o fixed.wav
    uv run $URL edit -r noisy.wav -i "Remove only the background noise, preserve everything else, and output audio of the same length."

    # Higher quality, about 4.4 times slower
    uv run $URL clone "Hello there." -r ref.wav --base

    # Or run locally:
    uv run auk_tts.py <command> [options]

COMMANDS:
    clone   - Speak new text in the voice of a reference clip
    design  - Speak text in a voice built from a written description
    edit    - Change existing audio by a natural-language instruction

OPTIONS:
    -o, --output          Output filename (default: output.wav, auto-increments)
    -r, --ref-audio       Reference clip (clone) or input audio (edit)
    -i, --instruct        Voice description (design) or edit instruction (edit)
    -s, --seconds         Output length. Default: estimated from the text
                          (clone, design) or the input length (edit)
    --base                Base model, 32 steps (default: Flash, 4 steps)
    --seed                RNG seed (default: none, so output varies per run)
    --no-convert          Keep Traditional Chinese as written
    --interval-silence    Silence between chunks in ms (default: 200)
    -v, --verbose         Show progress details

EDIT INSTRUCTIONS (the upstream cookbook templates; Chinese forms also work):
    Emotion   Change the emotion to {happy/angry/sad/fearful/surprised/disgusted/calm/excited}.
              将情感转变为{开心/愤怒/悲伤/恐惧/惊讶/厌恶/平静/兴奋}。
    Words     Replace '{original}' with '{new}'. | Add '{text}' before '{anchor}'. | Remove '{text}'.
    Pitch     Raise the pitch by {1/2/3} semitones. | Lower the pitch by {1/2/3} semitones.
    Speed     Adjust the speech speed to {0.5/0.75/1.25/1.5/2.0}x.   (set -s to input / speed)
    Volume    Increase the volume by {5/10/15} dB. | Decrease the volume by {5/10/15} dB.
    Timbre    Keep the spoken content unchanged and change the timbre to: "{description}".
    Accent    Remove the regional accent while preserving the speaker's voice and content.
    Whisper   Convert this speech into a soft whisper while preserving the speaker and content.
              Convert this whispered speech into a normal speaking voice while preserving the speaker and content.
    Sounds    Remove all {breaths/laughs/coughs} from the audio. | Add a {sound} at the {beginning/end} of the speech.
    Clean up  Remove only the background noise, preserve everything else, and output audio of the same length.
              Remove only the room reverberation, preserve everything else, and output audio of the same length.
    Separate  Keep only the {first/second} speaker to start talking and remove all other speakers.
              Keep only the speaker who says "{content}" and remove all other speakers.
              Keep only the singing voice and remove everything else.

NOTES:
    - Apple Silicon only. First run downloads about 6.6 GB of 8-bit MLX
      weights (8.3 GB if you use both --base and Flash). Resident memory is
      about 9 GB.
    - The weights are smcleod/AuK-MLX-8bit, a third-party conversion. They
      were checked against a local 8-bit conversion of the official tencent/AuK
      and Qwen2.5-Omni-3B release before this script used them.
    - The model does not choose the output length. clone and design estimate
      it from the text: 5.3 Han characters per second (measured) and 2.7
      English words per second. A wrong estimate stretches or squeezes the
      speech, so pass -s when you know the length.
    - Long text is split at sentence ends into chunks of up to 25 seconds and
      joined with --interval-silence. Every clone chunk uses the original
      reference. design builds the voice on the first chunk, then clones the
      rest from it, so the voice does not change between chunks.
    - Traditional Chinese is mispronounced: 颱風假 came back as 肥豐假. With
      opencc tw2s first it reads correctly, so clone and design convert by
      default. tw2s changes characters only, so Taiwan vocabulary survives.
      edit instructions are passed as written.
    - Emotion on a cloned voice takes two passes: clone, then edit. The edit
      keeps the words and costs some timbre (speaker similarity 0.91 -> 0.85
      measured with Flash).
    - Output is 24 kHz mono.
"""
import re
import sys
import tempfile
from pathlib import Path

import click
import numpy as np
import soundfile as sf

WEIGHTS_REPO = "smcleod/AuK-MLX-8bit"
WEIGHTS_REVISION = "abffd38adf7d9f0a94822e7c153e0c4ef70ef213"
BITS = 8

HAN_PER_SECOND = 5.3
WORDS_PER_SECOND = 2.7
CHUNK_SECONDS = 25.0

HAN_PATTERN = re.compile(r"[㐀-䶿一-鿿]")
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['.][A-Za-z0-9]+)*")
SENTENCE_END = re.compile(r"(?<=[。！？!?；;])|(?<=\.)\s+|\n+")
CLAUSE_END = re.compile(r"(?<=[，、,：:])")


def get_unique_filename(base_path: Path) -> Path:
    """Return a unique filename, adding -2, -3, etc. if the file already exists."""
    if not base_path.exists():
        return base_path
    counter = 2
    while True:
        new_path = base_path.parent / f"{base_path.stem}-{counter}{base_path.suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


def resolve_output_path(output: str) -> Path:
    """Auto-increment only the default name, so an explicit -o overwrites."""
    output_path = Path(output)
    if output == "output.wav":
        output_path = get_unique_filename(output_path)
    return output_path


def get_text_from_input(text: str | None) -> str:
    """Get text from argument or stdin."""
    if text is None:
        if sys.stdin.isatty():
            raise click.UsageError("No text provided. Pass text as an argument or pipe it via stdin.")
        text = sys.stdin.read()
    text = text.strip()
    if not text:
        raise click.UsageError("Text cannot be empty.")
    return text


def require_apple_silicon() -> None:
    import platform

    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise click.ClickException("AuK MLX runs on Apple Silicon only (darwin arm64).")


def to_simplified(text: str) -> str:
    """Convert Traditional Chinese to Simplified, characters only.

    Traditional left as-is is mispronounced: 颱風假 came back as 肥豐假 (Flash)
    and "safe 假" (base). Converted with tw2s, both read 6 of 6 hard words.
    """
    if not HAN_PATTERN.search(text):
        return text
    import opencc

    return opencc.OpenCC("tw2s").convert(text)


def estimate_seconds(text: str) -> float:
    han = len(HAN_PATTERN.findall(text))
    words = len(WORD_PATTERN.findall(text))
    return max(1.0, han / HAN_PER_SECOND + words / WORDS_PER_SECOND)


def split_long(piece: str) -> list[str]:
    """Split one over-long sentence at clause marks, as a fallback."""
    if estimate_seconds(piece) <= CHUNK_SECONDS:
        return [piece]
    return [part for part in CLAUSE_END.split(piece) if part.strip()]


def chunk_text(text: str) -> list[str]:
    """Group sentences into chunks of about CHUNK_SECONDS of speech each."""
    pieces = [part for sentence in SENTENCE_END.split(text) if sentence.strip()
              for part in split_long(sentence.strip())]
    chunks: list[str] = []
    for piece in pieces:
        if chunks and estimate_seconds(chunks[-1] + piece) <= CHUNK_SECONDS:
            chunks[-1] = chunks[-1] + " " + piece if WORD_PATTERN.match(piece) else chunks[-1] + piece
        else:
            chunks.append(piece)
    return chunks


def clone_instruction(text: str) -> str:
    return f'Say the following with the same voice: "{text}"'


def design_instruction(description: str, text: str) -> str:
    if HAN_PATTERN.search(text):
        return f'请基于下面的描述: "{description}",生成语音内容"{text}".'
    return f'Generate speech based on the following description: "{description}". The content to speak is: "{text}".'


def load_engine(base: bool, verbose: bool):
    """Download the variant's 8-bit weights once, then build a resident engine."""
    require_apple_silicon()
    from huggingface_hub import snapshot_download

    variant = "base" if base else "flash"
    if verbose:
        click.echo(f"Fetching {WEIGHTS_REPO} ({variant}, {BITS}-bit)...")
    weights = Path(snapshot_download(
        WEIGHTS_REPO,
        revision=WEIGHTS_REVISION,
        allow_patterns=[
            f"dit_{variant}.q{BITS}.safetensors", f"fusion_{variant}.safetensors",
            f"config_{variant}.yaml", "vae.safetensors", "thinker/*", "qwen/*",
        ],
    ))

    from auk_mlx.infer import AukMLX

    if verbose:
        click.echo("Loading model...")
    return AukMLX(
        str(weights), str(weights / f"config_{variant}.yaml"), str(weights / "qwen"),
        bits=BITS, sequential=False,
    )


def generate(engine, instruction: str, audio_path: str | None, seconds: float | None, seed: int | None):
    from auk_mlx.infer import GenerateOptions

    try:
        audio, _ = engine.generate(
            instruction, audio_path=audio_path, opts=GenerateOptions(gen_seconds=seconds, seed=seed)
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    return np.asarray(audio, dtype=np.float32)


def join_clips(clips: list[np.ndarray], sample_rate: int, gap_ms: int) -> np.ndarray:
    gap = np.zeros(int(sample_rate * gap_ms / 1000), dtype=np.float32)
    parts = []
    for index, clip in enumerate(clips):
        if index:
            parts.append(gap)
        parts.append(clip)
    return np.concatenate(parts)


def save_and_report(audio: np.ndarray, sample_rate: int, output: str, verbose: bool) -> None:
    output_path = resolve_output_path(output)
    sf.write(str(output_path), audio, sample_rate)
    if verbose:
        click.echo(f"Saved {output_path} ({len(audio) / sample_rate:.2f}s @ {sample_rate} Hz)")
    else:
        click.echo(str(output_path))


def speak_chunks(engine, chunks: list[str], ref_audio: str | None, description: str | None,
                 seconds: float | None, seed: int | None, verbose: bool) -> list[np.ndarray]:
    """Generate each chunk. Without a reference, chunk 1 becomes the reference.

    design has no reference, so a voice invented per chunk would change
    between chunks. Chunk 1 is designed from the description, then every
    later chunk is cloned from it.
    """
    clips = []
    with tempfile.TemporaryDirectory() as scratch:
        for index, chunk in enumerate(chunks, 1):
            length = seconds or estimate_seconds(chunk)
            if verbose:
                click.echo(f"Chunk {index}/{len(chunks)} ({length:.1f}s): {chunk[:40]}")
            if ref_audio is None:
                clip = generate(engine, design_instruction(description, chunk), None, length, seed)
                ref_audio = str(Path(scratch) / "voice.wav")
                sf.write(ref_audio, clip, engine.sample_rate)
            else:
                clip = generate(engine, clone_instruction(chunk), ref_audio, length, seed)
            clips.append(clip)
    return clips


def check_file(path: str) -> None:
    if not Path(path).is_file():
        raise click.UsageError(f"Audio file not found: {path}")


def check_seconds(seconds: float | None, chunks: list[str]) -> None:
    if seconds is not None and len(chunks) > 1:
        raise click.UsageError(
            f"-s sets the length of one call, but the text splits into {len(chunks)} chunks. "
            "Shorten the text or drop -s."
        )


common_options = [
    click.option("-o", "--output", default="output.wav", help="Output filename"),
    click.option("-s", "--seconds", type=float, default=None, help="Output length in seconds"),
    click.option("--base", is_flag=True, help="Base model, 32 steps (default: Flash, 4 steps)"),
    click.option("--seed", type=int, default=None, help="RNG seed"),
    click.option("-v", "--verbose", is_flag=True, help="Show progress details"),
]


def with_common_options(command):
    for option in reversed(common_options):
        command = option(command)
    return command


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """AuK CLI - clone, design and edit speech on Apple Silicon."""


@cli.command("clone")
@click.argument("text", required=False)
@click.option("-r", "--ref-audio", required=True, help="Reference clip to clone")
@click.option("--no-convert", is_flag=True, help="Keep Traditional Chinese as written")
@click.option("--interval-silence", type=int, default=200, help="Silence between chunks (ms)")
@with_common_options
def clone_command(text, ref_audio, no_convert, interval_silence, output, seconds, base, seed, verbose):
    """Speak new text in the voice of a reference clip."""
    check_file(ref_audio)
    text = get_text_from_input(text)
    if not no_convert:
        text = to_simplified(text)
    chunks = chunk_text(text)
    check_seconds(seconds, chunks)

    engine = load_engine(base, verbose)
    clips = speak_chunks(engine, chunks, ref_audio, None, seconds, seed, verbose)
    save_and_report(join_clips(clips, engine.sample_rate, interval_silence), engine.sample_rate, output, verbose)


@cli.command("design")
@click.argument("text", required=False)
@click.option("-i", "--instruct", required=True, help="Voice description")
@click.option("--no-convert", is_flag=True, help="Keep Traditional Chinese as written")
@click.option("--interval-silence", type=int, default=200, help="Silence between chunks (ms)")
@with_common_options
def design_command(text, instruct, no_convert, interval_silence, output, seconds, base, seed, verbose):
    """Speak text in a voice built from a written description."""
    text = get_text_from_input(text)
    if not no_convert:
        text = to_simplified(text)
    chunks = chunk_text(text)
    check_seconds(seconds, chunks)

    engine = load_engine(base, verbose)
    clips = speak_chunks(engine, chunks, None, instruct, seconds, seed, verbose)
    save_and_report(join_clips(clips, engine.sample_rate, interval_silence), engine.sample_rate, output, verbose)


@cli.command("edit")
@click.option("-r", "--ref-audio", required=True, help="Input audio to edit")
@click.option("-i", "--instruct", required=True, help="Edit instruction (see --help of the script)")
@with_common_options
def edit_command(ref_audio, instruct, output, seconds, base, seed, verbose):
    """Change existing audio by a natural-language instruction."""
    check_file(ref_audio)
    engine = load_engine(base, verbose)
    audio = generate(engine, instruct, ref_audio, seconds, seed)
    save_and_report(audio, engine.sample_rate, output, verbose)


if __name__ == "__main__":
    cli()
