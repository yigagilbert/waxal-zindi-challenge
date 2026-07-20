# Whisper large-v3 Plan (fallback experiment)

The high-ceiling **acoustic** bet. The leaderboard top (~0.91–0.93, WER ~0.10) is an acoustic
gap, not a decode gap — our XLS-R+LM is decode-optimized but capped by the acoustic model. Whisper
large-v3 is the untried architecture that could close it. **Run only after the champion-continuation
result is known**, unless a second GPU is idle.

Config: [whisper_large_v3_clean_audio_v3_lora.yaml](../configs/whisper_large_v3_clean_audio_v3_lora.yaml).

## Approach

- Base: `openai/whisper-large-v3` (seq2seq, open weights).
- Train on `clean_audio_v3` clean bucket (90,301); validate on in-domain WAXAL validation.
- **LoRA first** (r=16, q_proj/v_proj) — fail cheap before any full fine-tune. Full FT is a
  separate, later decision only if LoRA shows real promise.
- Trainer: `scripts/train_whisper.py` (has `lora.enabled`, `predict_with_generate`).

## Why LoRA before full fine-tune

- Full large-v3 FT is many GPU-days and easy to overfit 90k clips of low-resource speech.
- LoRA answers the only question that matters first: *does whisper large-v3 have a materially
  higher acoustic ceiling on these languages than XLS-R?* If LoRA can't approach the champion,
  full FT is unlikely to be worth the spend.

## What to watch (Whisper-specific failure modes)

- **Hallucination** — fluent text unrelated to audio (Whisper's classic low-resource failure).
- **Long-output / looping** — repeated phrases; watch generation length distribution.
- **Punctuation/casing** — WAXAL targets are `language_safe`-normalized; Whisper emits punctuation.
  Apply the same normalization at eval; mismatch inflates WER artificially.
- **Language drift** — large-v3 may decode Lingala/Shona toward a higher-resource neighbor.

## Early-stopping gate (kill cheap)

- If by ~step 1000–1500 the validation WER/combined is not on a trajectory to approach the
  champion (~0.135 combined equivalent), **stop**. Do not spend GPU-days without evidence.
- If hallucination/looping counts are high and not falling, stop — LoRA won't fix a base-model
  behavior mismatch.
- Compare on the **same raw WAXAL validation** as the champion, normalization matched.

## Comparison target

| Model | Validation combined (raw WAXAL val) | Notes |
|---|---|---|
| Champion XLS-R + expanded LM | ~0.135 | the number to beat |
| Whisper large-v3 LoRA | _tbd_ | greedy/generate; then + LM if CTC-style rescoring is added |

## Commands

### Smoke
```bash
python scripts/train_whisper.py \
  --config configs/whisper_large_v3_clean_audio_v3_lora.yaml \
  --max-train-samples 12 --max-eval-samples 6 --max-steps 2 \
  --output-dir checkpoints/whisper_large_v3_clean_audio_v3_lora_smoke
```

### LoRA run (tmux; only after champion continuation, or on a second GPU)
```bash
export HF_DATASETS_CACHE=/dev/shm/hf_datasets_cache
python scripts/train_whisper.py \
  --config configs/whisper_large_v3_clean_audio_v3_lora.yaml
```

## Phase-2 note

Whisper decodes without needing language prefixes, so it is Phase-2 friendly, but a Whisper
champion must still be wired into the no-metadata LID + routing flow — see
[PHASE2_NO_METADATA_READINESS.md](PHASE2_NO_METADATA_READINESS.md).
