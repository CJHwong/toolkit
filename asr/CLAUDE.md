# asr/

Speech recognition. Each script is standalone, runs with `uv run`, and can be
fetched straight from its published URL.

Root `CLAUDE.md` holds the conventions for writing these scripts, including
"Verify an audio script with a real run". Read that before measuring anything
here.

**`whisper.py` lives in `python/`, not here.** `auto_dub.py` executes it from
its raw URL at runtime (`WHISPER_SCRIPT`, line 69), so moving the file would
break dubbing, and it doubles as a general transcription tool. Its findings
belong with the other ASR engines, so they are recorded here.

## Which engine

| Need | Engine | Why |
|---|---|---|
| Default transcription and TTS round-trip verification | **`whisper.py`** | Renders Traditional for this audio, keeps digits as digits, no heavy dependency |
| A second opinion on the same audio | **`funasr_asr.py`** | Different architecture, so it fails differently. That is the point of it |
| Simplified Chinese output | `funasr_asr.py` | Paraformer is Simplified-trained and returns Simplified |
| Nothing to install beyond Python | `whisper.py` | `funasr_asr.py` installs 83 packages including torch on first run |
| `qwen3_asr.py` | untested | No run recorded yet, so no claim is made |

## Use two engines, and read their disagreements

A single ASR cannot tell you whether a wrong word came from the speech or from
the transcription. Two engines can, and the rule is simple:

- **Both engines wrong the same way** means the audio is wrong. `颱風假` came
  back as `繳封甲` from whisper and `找冯甲` from FunASR. Two unrelated models
  do not invent the same wrong word.
- **The engines disagree** means the audio is fine and the instruments are
  guessing among homophones. `天泉` came back as `天權` from one and `天全` from
  the other, and `續聘` as `續聘` from one and `徐聘` from the other. Both
  readings are *quán* and *xù*. The audio was correct.

This rule is what turned "the recording sounds wrong" into a specific,
fixable defect, and what cleared `天泉`, `中元普渡` and `開區權會` of blame.

## whisper.py

1. **It writes its transcript to `<input>.txt` by default, silently
   overwriting whatever sits there.** Running it on `gn.wav` replaced the
   reference transcript `gn.txt` with its own output. The words happened to
   match, so nothing was lost, but the file was clobbered. **Always pass
   `-o` to a path you chose.**
2. **It renders Traditional Chinese** for this audio, while FunASR returns
   Simplified. Any comparison across the two needs a script conversion first,
   or every character counts as a mismatch.
3. **It leaves digits as digits.** `48000` transcribes as `48000`, which keeps
   a string diff honest against a source that also has digits.

## funasr_asr.py

1. **It rewrites digits as Chinese words.** `48000` becomes `四万八千`, `35%`
   becomes `百分之三十五`, `108` becomes `一百零八`. Convenient for reading,
   and it **deflates any string-diff similarity score**. On audio identical to
   what whisper scored 0.9755, FunASR scored 0.8975, and the gap was mostly
   this rewriting rather than mishearing.
2. **It returns Simplified**, so its Traditional handling is invisible. Feed it
   Traditional text and it will still answer in Simplified, which hides whether
   the model read the characters correctly.
3. **First run installs 83 packages including torch**, roughly 90 MB. Budget
   for that before a timing measurement that includes it.

## Not built

- No ASR script converts script on output, so a Traditional-versus-Simplified
  comparison needs opencc applied by hand.
- `qwen3_asr.py` has no recorded run. Measure it the same way before trusting
  it for anything.
