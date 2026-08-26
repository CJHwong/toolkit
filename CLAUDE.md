# toolkit

A flat collection of standalone CLI scripts. Each script in `python/` is a
single self-contained file, run directly with `uv run`. No shared package, no
`requirements.txt`, no virtualenv to manage.

## PEP 723 scripts

Every script carries its own dependencies inline as a PEP 723 metadata block,
so `uv run script.py` resolves and caches an isolated environment on first run
and nothing leaks between scripts.

A script header looks like this:

```python
#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "click==8.3.1",
# ]
# ///
"""
One-line summary.

USAGE:
    uv run script.py ...
"""
```

Conventions to match when adding or editing a script:

- Keep it to one file. If it needs a second module, it does not belong here.
- Put the `USAGE` block and any non-obvious model/platform notes in the module
  docstring, not in a separate README.
- Do heavy imports (torch, mlx, transformers) inside `main()`, after argument
  parsing, so `--help` stays instant.
- Guard platform assumptions early (e.g. exit cleanly when a model is Apple
  Silicon only) and check external tools (`ffmpeg`) before doing real work.
- Scripts are meant to also run straight from a raw GitHub URL, so no relative
  imports and no local paths.

## Show the GitHub URL in USAGE

Because a script runs directly from its raw GitHub source, the `USAGE` examples
use that URL, not a local filename. Define it once as a shell variable, then
reuse it:

```
USAGE:
    # Run directly from GitHub (no clone needed):
    URL=https://raw.githubusercontent.com/CJHwong/toolkit/main/python/your_script.py

    uv run $URL ...
```

This only works when every dependency resolves from PyPI. A script that needs a
`[tool.uv.sources]` git dep can't be fetched and run from the URL alone (uv
ignores `[tool.uv.sources]` in remote scripts), so document the `--with` form
for those instead.

## Pin every dependency

Every entry in a `dependencies` list is pinned to an exact version with `==`.
No bare names, no `>=`, no `<` ranges. A `uv run` of a year-old script must
pull the same versions it was written against, not whatever happens to be
latest that day.

To pin a new or loosened dependency, resolve what uv actually picks and write
that version into the header:

```
uv export --script python/your_script.py --no-hashes
```

Copy the resolved `name==version` for each dependency you declared back into
the header. Metadata resolution only, no full downloads, so this is cheap.

Git sources in `[tool.uv.sources]` are the one exception to `==` (they are not
PyPI versions); pin them by `rev` to a commit when the upstream is stable
enough to.

## Pin what the upstream left loose

A wrapped package often declares its own stack with `>=`, which puts the
script back at the mercy of whatever resolves that day. When a script wraps a
third-party port, pin that port's core stack in your own header too, and say in
a comment why. `dots_tts.py` pins `mlx-lm` because a loose resolve picks a
version whose module layout the port cannot import; `index_tts.py` pins the
MLX pair for the same reason.

## Verify an audio script with a real run

A TTS or ASR script is not verified by a `--help` that exits 0. Generate real
audio and measure it. Two axes, both scripted, and both are needed:

- **Intelligibility**: transcribe the output with `whisper.py` and diff the
  round-trip against the input text. Text-frontend bugs produce fluent,
  confident, wrong speech that a casual listen does not catch.
- **Voice fidelity**: `spk_sim.py` for ECAPA cosine similarity between the
  reference clip and the clone.

Read both. A clone can score high similarity while the words are destroyed, so
either number alone will call a broken run a pass. Control for the instrument
before blaming the model: transcribe the reference clip itself, and if the ASR
mangles that too, the ASR is the problem.

Report generation cost with every run: audio length, wall clock, and RTF. RTF
is hardware bound and varies several fold across Apple Silicon generations, so
a number without its machine is meaningless.

What each wrapped model actually does, as opposed to what its docs claim, lives
in `python/CLAUDE.md`. Read it before promising a capability.
