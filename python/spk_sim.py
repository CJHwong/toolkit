# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "speechbrain==1.1.0",
#     "click==8.3.1",
# ]
# ///
# NOTE: torch, torchaudio, numpy, huggingface-hub are NOT pinned here on
# purpose. speechbrain is brittle against transitive versions (it calls
# torchaudio.list_audio_backends and hf_hub_download(use_auth_token=...),
# both removed in newer releases). Letting speechbrain 1.1.0 drive its own
# compatible torch/hub stack keeps the script runnable; pinning those heavy
# deps to latest would break the import. Direct deps are pinned per toolkit
# convention; transitives float within speechbrain's declared ranges.
"""
spk_sim CLI - speaker cosine similarity between a reference clip and others.

Scores how closely each clip matches a reference voice, for TTS clone evaluation.
Uses speechbrain ECAPA-TDNN (VoxCeleb) speaker embeddings: encode both clips,
cosine similarity in [-1, 1]. Higher = closer to the reference voice.

USAGE:
    # Run directly from GitHub (no clone needed):
    URL=https://raw.githubusercontent.com/CJHwong/toolkit/main/python/spk_sim.py

    # Score one or more clones against a reference
    uv run $URL ref.wav clone_a.wav clone_b.wav

    # JSON output (for piping into an eval harness)
    uv run $URL ref.wav clone.wav --json

    # Score every wav in a dir against a reference
    uv run $URL ref.wav clones/*.wav

    # Or run locally:
    uv run spk_sim.py [OPTIONS] REF.wav CLONE.wav [CLONE.wav ...]

OPTIONS:
    -m, --model      speechbrain speaker-verification HF repo (default: ECAPA VoxCeleb)
    --device         cpu | mps | cuda (default: cpu; embed is small, cpu is fine)
    --threshold      same-speaker cosine cutoff for the verdict (default: 0.25)
    -j, --json       emit JSON instead of a table
    -v, --verbose    show progress

NOTES:
    - First run downloads the ECAPA model (~80MB) from Hugging Face.
    - Inputs are resampled to 16kHz mono via ffmpeg (the ECAPA training sample rate).
    - ECAPA is trained on VoxCeleb (largely English); it still embeds Mandarin/
      code-switched voices usefully for relative similarity, but treat absolute
      numbers as comparative, not calibrated for non-English speakers.
    - A typical same-speaker threshold is ~0.25; same clip ~1.0, unrelated ~0.
"""
import json
import tempfile
from pathlib import Path

import click


def resample_16k_mono(src: Path, tmpdir: Path, verbose: bool) -> Path:
    """Resample a clip to 16kHz mono wav via ffmpeg (ECAPA's training rate)."""
    import hashlib
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        raise click.UsageError("ffmpeg not found (needed to resample to 16kHz mono)")

    # Hash the absolute path so ref and a same-stemmed clone (e.g. ref hoss.wav
    # vs clone .../hoss.wav) get distinct temp files instead of clobbering each
    # other, which would make verify_files compare a file to itself (sim=1.0).
    digest = hashlib.md5(str(src.resolve()).encode()).hexdigest()[:10]
    dst = tmpdir / f"{digest}_{src.stem}__16k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src),
         "-ar", "16000", "-ac", "1", str(dst)],
        check=True,
    )
    if verbose:
        click.echo(f"resampled {src.name} -> 16kHz mono", err=True)
    return dst


@click.command()
@click.version_option(version="0.1.0")
@click.argument("ref", type=click.Path(exists=True, path_type=Path))
@click.argument("clones", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("-m", "--model", default="speechbrain/spkrec-ecapa-voxceleb", help="speechbrain HF repo")
@click.option("--device", default="cpu", help="cpu | mps | cuda")
@click.option("--threshold", type=float, default=0.25, help="same-speaker cosine cutoff")
@click.option("-j", "--json", "as_json", is_flag=True, help="emit JSON")
@click.option("-v", "--verbose", is_flag=True, help="verbose")
def cli(ref, clones, model, device, threshold, as_json, verbose):
    """Score speaker cosine similarity of each CLONE against REF (ECAPA, 16kHz).

    Examples:

    \b
        spk_sim.py ref.wav clone_a.wav clone_b.wav
        spk_sim.py ref.wav clone.wav --json
        spk_sim.py ref.wav clones/*.wav
    """
    import torchaudio

    # speechbrain 1.0.2 calls torchaudio.list_audio_backends() at import for an
    # advisory warning check; torchaudio 2.10 removed it. Shim so speechbrain
    # imports on any torchaudio version. The check is advisory only (this script
    # never switches the global audio backend; we resample with ffmpeg instead).
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["ffmpeg"]

    from speechbrain.inference.speaker import SpeakerRecognition

    if verbose:
        click.echo(f"Loading speaker model: {model} (device={device})", err=True)

    verifier = SpeakerRecognition.from_hparams(
        source=model,
        savedir=str(Path.home() / ".cache" / "spk_sim" / model.split("/")[-1]),
        run_opts={"device": device},
    )

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        ref16 = resample_16k_mono(ref, td_path, verbose)

        rows = []
        for clone in clones:
            clone16 = resample_16k_mono(clone, td_path, verbose)
            score, prediction = verifier.verify_files(str(ref16), str(clone16))
            sim = float(score.item())
            verdict = "same" if sim >= threshold else "diff"
            rows.append({
                "clone": str(clone),
                "similarity": sim,
                "verdict": verdict,
            })
            if verbose:
                click.echo(f"{clone.name}: sim={sim:.4f} ({verdict})", err=True)

    if as_json:
        click.echo(json.dumps({"ref": str(ref), "threshold": threshold, "results": rows},
                              indent=2, ensure_ascii=False))
        return

    header = f"{'clone':<40} {'sim':>8}  {'verdict':>7}"
    click.echo(header)
    click.echo("-" * len(header))
    for r in rows:
        click.echo(f"{Path(r['clone']).name:<40} {r['similarity']:>8.4f}  {r['verdict']:>7}")
    click.echo(f"\nref: {ref}  threshold: {threshold}  (sim >= threshold => same speaker)")


if __name__ == "__main__":
    cli()