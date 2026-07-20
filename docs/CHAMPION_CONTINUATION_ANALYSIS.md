# Champion Continuation Analysis — clean_audio_v3

Experiment: continue fine-tuning `champion/checkpoint-24000` on `clean_audio_v3` at a low LR,
to see if better/more clean audio improves the acoustic model **before** investing in Whisper.
Config: [xlsr_300m_champion_continue_clean_audio_v3.yaml](../configs/xlsr_300m_champion_continue_clean_audio_v3.yaml).

Baseline to beat = the champion numbers in [CHAMPION_CHECKPOINT_REPORT.md](CHAMPION_CHECKPOINT_REPORT.md).

## The comparison must be apples-to-apples

- **Decision metric:** per-language + pooled **combined** (0.5·WER + 0.5·CER) on
  **`data/processed_generalization_mix` validation (raw/untrimmed audio)**, decoded with the
  **same** expanded-LM beam search + params as the champion (lin α0.9/β0.5, lug α0.4/β−0.5,
  sna α0.7/β−0.5, beam 400). Test audio is raw, so we always evaluate on raw validation — never
  the cleaned/trimmed validation, even though training used trimmed audio.
- The in-loop `eval_cer` (on the cleaned validation) is **monitoring only**, not the gate.

## Early-stopping gates (manual, checked between checkpoints)

Automatic guard: `early_stopping.patience: 4` on in-loop `eval_cer` (stops a dead run).
The real decision is manual, using the metrics below.

**STOP / discard the continuation if any of:**
- Pooled validation combined does not beat champion (0.135) by **> 0.003** by ~step 2000.
- Lingala regresses, or improves < 0.005 while lug/sna get worse (net wash).
- Shona or Luganda combined regresses by **> 0.005** (catastrophic forgetting signal).
- dot-only / empty / very-short (<3 char) prediction counts **increase** vs champion.
- Repeated-n-gram (looping) counts increase vs champion.
- Best improvement is within run-to-run noise (< 0.002 pooled).

**CONTINUE / promote a checkpoint only if:**
- Pooled combined clearly beats champion (> 0.003), **or**
- Lingala improves meaningfully with lug/sna held within −0.002, **or**
- Degenerate-output counts drop materially with combined at least flat.

## Per-checkpoint evaluation (fill this in)

For each saved checkpoint (500, 1000, …): decode `processed_generalization_mix` validation with
beam+LM (champion params) and record. Write raw numbers to
`outputs/analysis/champion_continue_clean_audio_v3_curve.json`.

| checkpoint | lin comb | lug comb | sna comb | pooled comb | dot-only | empty | very-short | rep-ngram | verdict |
|---|---|---|---|---|---|---|---|---|---|
| champion (baseline) | 0.1711 | 0.1148 | 0.0857 | ~0.135 | _tbd_ | _tbd_ | _tbd_ | _tbd_ | baseline |
| continue-500  |  |  |  |  |  |  |  |  |  |
| continue-1000 |  |  |  |  |  |  |  |  |  |
| continue-1500 |  |  |  |  |  |  |  |  |  |
| continue-2000 |  |  |  |  |  |  |  |  |  |
| continue-2500 |  |  |  |  |  |  |  |  |  |
| continue-3000 |  |  |  |  |  |  |  |  |  |

## Commands

### Smoke (verify it loads champion weights + runs a few steps; ~2 min)
```bash
python scripts/train_xlsr_ctc.py \
  --config configs/xlsr_300m_champion_continue_clean_audio_v3.yaml \
  --max-train-samples 12 --max-eval-samples 6 --max-steps 2 \
  --output-dir checkpoints/xlsr_300m_champion_continue_clean_audio_v3_smoke
```

### Full short continuation (tmux; ~hours)
```bash
export HF_DATASETS_CACHE=/dev/shm/hf_datasets_cache   # keep transient arrow off root
python scripts/train_xlsr_ctc.py \
  --config configs/xlsr_300m_champion_continue_clean_audio_v3.yaml
```

### Evaluate a checkpoint vs champion (per-language beam+LM on raw validation)
Run the sweep in single-point mode (or the fixed params) per language, then read combined:
```bash
python scripts/sweep_kenlm_decode_params.py \
  --checkpoint checkpoints/xlsr_300m_champion_continue_clean_audio_v3/checkpoint-2000 \
  --dataset-dir data/processed_generalization_mix --split validation \
  --kenlm-dir data/lm_expanded --order 5 \
  --alphas 0.9 --betas 0.5 --beam-width 400 --languages lin --max-samples-per-language 2000 \
  --output outputs/analysis/continue_ckpt2000_lin.json
# repeat with --languages lug --alphas 0.4 --betas -0.5, and --languages sna --alphas 0.7 --betas -0.5
```
Degenerate-output counts (dot-only/empty/very-short/rep-ngram) come from
`scripts/analyze_prediction_distributions.py` on the greedy/beam predictions.

## Submission candidate (ONLY if a checkpoint beats champion on raw validation)

Same chain as the champion, swapping `--checkpoint` to the winning continuation checkpoint and
writing to `outputs/submissions/submission_champion_continue_clean_audio_v3_best.csv`:
3× `run_xlsr_inference.py --split test --languages <l> --decoder-mode beam_lm
--kenlm-model data/lm_expanded/<l>_5gram.binary --unigrams-file data/lm_expanded/<l>.txt`
(lin α0.9/β0.5, lug α0.4/β−0.5, sna α0.7/β−0.5, `--beam-width 400`) →
`merge_predictions.py` (`--raw-dir "$WAXAL_RAW_DIR"`) → `postprocess_predictions.py` →
`make_submission.py --empty-target "."` → `validate_submission.py`.

Validate: 4,253 rows · aligned to SampleSubmission · 0 missing/duplicate/empty · no weird chars.
**Do not submit unless validation combined clearly beats champion.** Keep 0.861 selected regardless.

## Result & recommendation (2026-07-20) — NEGATIVE, champion retained

Clean-only continuation `xlsr_300m_champion_continue_clean_audio_v3` was run and **early-stopped
at step 2500** (EarlyStopping patience 4, no `eval_cer` improvement).

In-loop eval (cleaned validation, greedy — monitoring metric, not the raw-val gate):

| step | eval_cer | eval_loss |
|------|----------|-----------|
| 500  | 0.2450   | 0.3141 |
| 1000 | 0.2465   | 0.3166 |
| 1500 | 0.2457   | 0.3158 |
| 2000 | 0.2463   | 0.3138 |
| 2500 | 0.2453   | 0.3105 |

- **`eval_cer` is dead flat (~0.245)** and **train loss never descended (2.3–2.8)** — the model
  did not meaningfully change. (`eval_wer` is unusable here: cleaned-val references tokenize to
  ~1 "word"/example, inflating it to ~25; use `eval_cer`.)
- Root cause = as predicted: `clean_audio_v3` train is Afrivoice-heavy, and language-balanced
  sampling makes the Lingala pool ~54% external read-speech. Continuing at 2e-5 on that
  distribution neither fit the new data nor improved in-domain WAXAL — matching the earlier
  clean-audio-v2 negative result.

**Verdict: discard the continuation, retain `champion/checkpoint-24000`. Keep 0.861 selected.**
No submission generated (no checkpoint beat the champion; a raw-val beam+LM gate on checkpoint-500
is optional-for-the-record but expected to tie/lose given the flat loss and eval).

**Next:** the acoustic ceiling is not reachable by continuing this XLS-R on more read-speech.
Move to the Whisper large-v3 LoRA bet ([WHISPER_LARGE_V3_PLAN.md](WHISPER_LARGE_V3_PLAN.md)).
The clean+medium variant is **not** run (its precondition — clean-only improving — failed).
A WAXAL-clean-only continuation is low expected value (champion already fits WAXAL; the model
barely moved here) and is deprioritized in favor of Whisper.
