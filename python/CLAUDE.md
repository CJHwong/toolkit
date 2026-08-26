# Model knowledge, per engine

What the wrapped models actually do, as opposed to what their docs claim. Every
claim here came from a real run: generate audio, transcribe it back, measure it.
Root `CLAUDE.md` holds the conventions for writing these scripts; this file
holds what the engines get wrong.

Verification method for anything below is in root `CLAUDE.md` under "Verify an
audio script with a real run". Read both axes. A clone can score high speaker
similarity while the words are destroyed.

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

## index_tts.py (IndexTTS-2.5 via MLX)

1. **Traditional Chinese is silently garbled.** The tokenizer
   (`multilingual_zh_ja_yue_char_del`) is Simplified-only. Same sentence, same
   ref, same seed: 繁體 gave "JQBuck 微評修好了", 简体 round-tripped exactly.
   Timbre unaffected either way (0.832 vs 0.846), so speaker similarity alone
   calls this a pass. The script converts with opencc `tw2sp` under `-l zh`.
   `tw2sp` also maps Taiwan vocabulary (批次->批量, 佇列->队列); `tw2s`
   converts characters only and keeps the Taiwan words.
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

Verified mlx-audio 0.4.2 `Model` signatures:
- `generate(text, voice, instruct, temperature, speed, lang_code, ref_audio, ref_text, ...)`
- `generate_custom_voice(text, speaker, language, instruct, ...)`
- `generate_voice_design(text, instruct, language, ...)`
- `generate_voice_clone` -> None, not ported, so no precomputed-prompt reuse

## dots_tts.py (dots.tts via dots-tts-mlx)

**No emotion control at all.** `DotsTtsModel.generate()` has no instruct or
emotion parameter. Voice shaping is limited to `guidance_scale`,
`speaker_scale`, `num_steps`, and post-hoc `--speed` (ffmpeg atempo). You
cannot tell it to say something angrily.

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

- Symbol-cleaning pass for `index_tts.py`, pending the probe in finding 4.
  Do not write a blanket slash strip; `2026/08/27` may well be fine.
- `tw2sp` versus `tw2s` switch, if Taiwan vocabulary should win over matching
  the model's training distribution.
- `dots_tts.py --profile`, to expose `SpeakerProfile` reuse.
- Emotion-reference picker: a clone helper that takes a directory of
  emotion-tagged references and selects one per run.
