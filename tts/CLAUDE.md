# tts/

Voice cloning and speech synthesis. Each script is standalone, runs with
`uv run`, and can be fetched straight from its published URL.

Root `CLAUDE.md` holds the conventions for writing these scripts, including
"Verify an audio script with a real run". Read that before measuring anything
here. This file holds two things: which engine to reach for, and what each
engine actually does.

## Which engine

| Need | Engine | Why |
|---|---|---|
| Chinese clone from a reference clip | **`index_tts.py`** | Takes Traditional text directly, segments long text, controls duration, and seeds the run |
| Fastest iteration | **`qwen3_tts.py --small`** | RTF 0.84 measured warm, against 1.37 for the 1.7B |
| Reproducible output | `index_tts.py`, `dots_tts.py` | Both take `--seed`. qwen3 is unseeded by design |
| Exact target length | `index_tts.py` | `--duration-factor` is linear to 0.002, so one shot lands it |
| A voice from a written description | `qwen3_tts.py design` | The only engine that builds a voice from a description, so it clones nobody |
| Emotion with no reference | `qwen3_tts.py speak` | The only engine that honors `instruct`, over 9 preset speakers |
| 48 kHz output | `dots_tts.py` | The only 48 kHz engine |
| **Cloned timbre plus emotion** | **nothing** | See "Getting a cloned voice with emotion" |

## The default

For Chinese cloning, reach for **`index_tts.py`**. Not because its timbre
measured better, which the noise floor does not support, but because no other
engine has all of this at once:

1. It accepts Traditional text with no conversion step.
2. It segments long text internally, with `--interval-silence` between segments.
3. It controls duration natively, and the control is linear.
4. It takes `--seed`, so a run repeats. `qwen3_tts.py` has no seed.

For the fastest loop, `qwen3_tts.py --small` is the quickest thing measured.

## Capability matrix

| | clone from ref | emotion instruct | duration control |
|---|---|---|---|
| `qwen3_tts.py` Base | yes | kwarg accepted, ignored | no |
| `qwen3_tts.py` CustomVoice | no, 9 presets | yes | no |
| `dots_tts.py` | yes | no | post-hoc ffmpeg atempo only |
| `index_tts.py` | yes | dropped by the MLX port | yes, native and linear |

**The standing gap:** no shipped MLX path combines a cloned timbre with an
emotion instruct in one call. Upstream IndexTTS-2 has it (`emo_audio_prompt`,
`emo_alpha`); the MLX port drops it. Do not promise emotion control on a
cloned voice without checking this table first.

## Measured comparison

Same reference clip, same 146-character passage, one run each, Apple M5 Pro
over ssh. Lower RTF is faster; 1.0 is real time.

| Engine | Text | Rate | Duration | Gen | RTF | whisper sim | `spk_sim` | Hard words |
|---|---|---|---|---|---|---|---|---|
| `index_tts.py` | Traditional | 22.05k | 30.73s | 29s | **0.94** | **1.0000** | 0.8357 | 6/6 |
| `qwen3_tts.py` 1.7B | Simplified | 24k | 28.38s | 39s | 1.37 | **1.0000** | 0.7238 | 6/6 |
| `qwen3_tts.py` 0.6B | Simplified | 24k | 33.57s | 25s warm | **0.84** | 0.9847 | 0.7582 | 5/6 |
| `dots_tts.py` | Simplified | 48k | 26.08s | 341s | 13.1 | 0.9924 | 0.8046 | 6/6 |
| `dots_tts.py` | Traditional | 48k | 27.84s | 256s | 9.2 | 0.9208 | 0.7401 | 4/6 |

**Read the speed and the intelligibility columns. Do not read the `spk_sim`
column as a ranking.** Both qwen3 sizes are unseeded, and one engine measured
0.8238 on one text and 0.7238 on another, a spread of 0.1 that is the same size
as the entire range across engines above. Ranking on those numbers would be
reading sampling noise. N replicates per condition is the only thing that
separates the two.

Notes on the same table:

- `dots_tts.py` warm is roughly ten times slower than the alternatives, with no
  intelligibility gain on this text. Reach for it only when 48 kHz is required.
- `index_tts.py` outputs 22.05 kHz. For delivery over a channel that re-encodes
  to 32 kbps anyway, such as a LINE voice message, that difference disappears.
- The 0.6B produced the same 131 characters as the 1.7B, so it loses no content.
  It speaks about 18% slower, which is pacing, not filler.

## index_tts.py (IndexTTS-2.5 via MLX)

1. **Traditional Chinese is silently garbled.** The tokenizer
   (`multilingual_zh_ja_yue_char_del`) is Simplified-only. Same sentence, same
   ref, same seed: 繁體 gave "JQBuck 微評修好了", 简体 round-tripped exactly.
   Timbre unaffected either way (0.832 vs 0.846), so speaker similarity alone
   calls this a pass. The script converts with opencc under `-l zh`, and on
   2026-09-16 it moved to `tw2s`, which changes characters only and keeps
   Taiwan vocabulary. It used `tw2sp` before that, which also mapped vocabulary
   (軟體 -> 软件, 批次 -> 批量, 佇列 -> 队列). **That move is a behaviour
   change: index output now keeps Taiwan wording where it used to replace it.**
2. **Accent does not transfer, timbre does.** A strong 台灣腔 reference still
   reads in mainland Mandarin. ECAPA cannot see this; it scores identity, not
   region. "Clone my voice" means timbre only.
3. **Reference quality is the ceiling.** macOS `say` references produce robotic
   output and are also slower to synthesize from (RTF 1.28-1.32) than real
   human references (1.11-1.17). Never evaluate a cloner with synthetic refs.
4. **Slash abbreviations break.** "HB/L No." read as "HVAC call number",
   "HBALF L number", "Hbaffle number", "HBAC BAL number" across four
   references. Deterministic and voice-independent. "HBL No." is clean, and
   `--no-normalization` does not help, so it is the model, not wetext. Untested:
   whether `/` also fails in dates and fractions, and how `&`, `%`, `#`,
   container numbers and currency read.
5. **`--duration-factor` is linear** to within 0.002 across 0.6 to 1.6, so a
   target length lands in one shot. It is generative, not a time-stretch, so
   pitch stays natural. A higher factor also lowers RTF.
6. **Speed is hardware bound.** Same 24.7s output, warm: M1 Pro RTF 1.38, M5
   Pro 0.56. The port advertises 0.47, which is roughly honest on current
   hardware. Cold Metal kernel compilation costs more than the hardware gap:
   30.3s vs 13.8s on the same M5 Pro. Run batches on the faster box over ssh.
7. **The reference is meant to be 15s or shorter.** `gn.wav` at 16.5s ran and
   returned 0, so this reads as a guideline rather than a hard limit.

**Which checkpoint:** unknowable. The 2.5 technical report describes a GRPO
post-trained variant (WER 6.75->6.00, speaker similarity 73.18->73.63), but
IndexTeam publishes one HF repo with one `gpt.pth` and no `-RL` variant, and
neither the model card nor the paper says which was released.

## qwen3_tts.py (Qwen3-TTS via mlx-audio)

**Base ignores `instruct`.** `Qwen3-TTS-12Hz-1.7B-Base` clones from ref audio
plus ref text and accepts an `instruct` kwarg in mlx-audio's `generate()`, but
the weights do not honor it. Proven by ear and by variance: unseeded
no-instruct runs varied 4.0 dB RMS and 4.4 dB peak on their own, which swamps
every instruct delta. An `-i/--instruct` flag was wired, tested, and reverted;
shipping it would imply control the model does not deliver.

**Emotion lives in the other two variants.** CustomVoice
(`generate_custom_voice(text, speaker, instruct)`) honors instruct but only
over 9 preset speakers, 1.7B only, no ref audio. VoiceDesign
(`generate_voice_design(text, instruct)`) builds a voice from a description,
so it clones nobody.

**qwen3-tts is unseeded.** Any single A/B is confounded by sampling variance.
To test an effect, run N replicates per condition and compare distributions.
One-versus-one will show you a signal that is not there.

**Traditional Chinese is mispronounced, and the defect is size-independent.**
One reference clip, one model, one variable. At 1.7B: 颱風假 came out 圈封夾,
綜合 came out 沖派, 續聘 came out 助聘. At 0.6B: 颱風假 came out 牙封甲, 社區
came out 市區, 調整颱風天 came out 正牙封甲. So this is the text frontend, not
the weights, and the fix is required at both sizes.

Converted with opencc `tw2s`, all of those read correctly and both ASR engines
transcribed them. `clone` and `speak` convert by default, gated on a Chinese
language and on Han characters being present; `--no-convert` sends the text
untouched. `tw2s` changes characters only, so Taiwan vocabulary survives
(軟體 stays 軟體, 批次 stays 批次). All three TTS engines converged on `tw2s`
on 2026-09-16, so they now agree on vocabulary handling.

**Numbers, `~` and `%` already read correctly.** `45,000~55,000` spoke as
四万五千到五万五千 and `35%` as 百分之三十五, both confirmed by whisper and
FunASR. Do not write a blanket symbol strip here; the punctuation carries the
meaning, and stripping `~` and `%` deletes real words. `clean_for_speech()`
therefore drops shortcodes, emoji and Markdown markers, and keeps everything
else. `--no-clean` bypasses it.

**One call over long text truncates.** 1,294 characters in a single call gave
217.7s of audio where 246s was expected. Both ASRs found 謝謝大家 missing from
the end and a phrase dropped from the opening, and ECAPA similarity fell to
0.5727 against 0.8238 for the same text chunked at paragraph level. Chunk it.

**Chunking can loop.** One 67-character chunk produced 20.1s of audio instead
of the expected 12.8s, and both ASRs read that whole paragraph as
`與與與與與劉劉劉`. The model is unseeded, so a retry of identical text fixed
it. Diagnose by characters per second before blaming the text: healthy output
runs 4.3 to 5.9, the looping chunk ran 3.33.

**The reference must be a format libsndfile reads.** An `.m4a` (AAC) fails with
`LibsndfileError: Format not recognised`. Convert to WAV before the model loads,
or the run dies before it starts.

**Long text is segmented internally and every segment is kept.**
`Model.generate()` takes `split_pattern` (default `"\n"`) and `max_tokens`
(default 4096), splits the text, and yields a `GenerationResult` per segment
carrying `segment_idx`. All three commands join every segment, with
`--interval-silence` (default 200ms) between them. Verified on a real run:
three paragraphs produced 5.26s of audio and transcribed back with all three
present. Until this was fixed the script kept `results[0]` and dropped the
rest, which is most of why a single call over 1,294 characters truncated.

Verified mlx-audio 0.4.2 `Model` signatures:
- `generate(text, voice, instruct, temperature, speed, lang_code, ref_audio, ref_text, split_pattern, max_tokens, ...)`
- `generate_custom_voice(text, speaker, language, instruct, ...)`
- `generate_voice_design(text, instruct, language, ...)`
- `generate_voice_clone` -> None, not ported, so no precomputed-prompt reuse

## dots_tts.py (dots.tts via dots-tts-mlx)

**Traditional Chinese is mispronounced, the same way qwen3 is.** Same engine,
same reference, same text, one variable. Simplified scored 0.9924 with 6 of 6
hard words; Traditional scored 0.9208 with 4 of 6, reading 勞基法 as
**`logifa`**, 出勤 as 出城, and 爭執 as 掙扎. Latin characters coming back for
Chinese input is a tokenizer failure, not a homophone slip. The script converts
with opencc `tw2s` as of 2026-09-16; pass `--no-convert` to send the text
untouched.

**No emotion control at all.** `DotsTtsModel.generate()` has no instruct or
emotion parameter. Voice shaping is limited to `guidance_scale`,
`speaker_scale`, `num_steps`, and post-hoc `--speed` (ffmpeg atempo). You
cannot tell it to say something angrily.

**Slow.** Warm RTF 9.2 on an M5 Pro, against 0.84 for `qwen3_tts.py --small`
and 0.94 for `index_tts.py`. The cost buys 48 kHz output and nothing else
measured here.

**Unwired perf win:** `generate()` and `generate_long()` accept
`profile: SpeakerProfile | None`. Build a profile from a reference once and
reuse it instead of re-encoding `prompt_audio` plus `prompt_text` every call.
`dots_tts.py` does not expose this yet.

## Getting a cloned voice with emotion

Cheapest first.

1. **Put the emotion in the reference.** Both cloners are in-context and
   inherit delivery from the ref clip. An angry clone means recording an angry
   reference. Works today, costs one reference per emotion. This is how
   in-context cloning is meant to be used and it gets most of the way there.
2. **VoiceDesign**, when "a voice like mine" is acceptable:
   `qwen3_tts.py design -i "warm male voice, mid-30s, speak angrily"`. Emotion
   lands because the variant is instruct-trained, but timbre fidelity is worse
   than cloning.
3. **LoRA fine-tune Qwen3-TTS Base.** Real but expensive, and it leaves this
   stack: PyTorch plus HF Trainer on a rented GPU, not MLX. Two documented
   traps. Official full SFT has a double label-shift that accelerates audio
   each epoch, a missing `text_projection` call that crashes 0.6B, and a
   default LR of 2e-5 that is too high (use 2e-6). And fine-tuning flattens
   emotion, which is the opposite of the goal; the suggested mitigation of
   tuning Base for timbre then applying instruct via CustomVoice is not
   verified end to end. Community recipe: 10-30 min of clean 24kHz
   single-speaker audio, LoRA rank 16 / alpha 32, LR 2e-6, ~10 epochs,
   inference LoRA scale 0.3 (1.0 over-steers), batch 4 on a 24GB GPU.
   - https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning
   - https://github.com/instavar/qwen3-tts-lora-finetuning
   - https://qwenlm-qwen3-tts.mintlify.app/guides/custom-voice

## Not built

- `index_tts.py`: chat-shortcode, emoji and Markdown cleaning. The other two
  engines strip those; index does not, so `:pray:` still reaches the model
  there. Add `clean_for_speech()` for consistency.
- `index_tts.py`: symbol-cleaning pass, pending the probe in finding 4. Do not
  write a blanket slash strip; `2026/08/27` may well be fine. The
  shortcode and emoji cleaner above is a different thing, and safe.
- `dots_tts.py --profile`, to expose `SpeakerProfile` reuse.
- Emotion-reference picker: a clone helper that takes a directory of
  emotion-tagged references and selects one per run.
