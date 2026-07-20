# Champion-Recipe Retrain on clean_audio_v3 — Analysis

Question: does the **same proven champion recipe**, trained from base XLS-R on the **cleaner
clean_audio_v3** data, beat the current champion? Config:
[xlsr_300m_champion_recipe_clean_audio_v3.yaml](../configs/xlsr_300m_champion_recipe_clean_audio_v3.yaml).

Baseline to beat = [CHAMPION_CHECKPOINT_REPORT.md](CHAMPION_CHECKPOINT_REPORT.md).

## Prior (be honest before spending ~9–10 GPU-hours)

- **clean-audio-v2** (from-scratch on cleaned + external audio): worse on every language incl Lingala.
- **clean_audio_v3 continuation** (low-LR): flat, no in-domain gain.
- clean_audio_v3 train is ~63% external read-speech (Afrivoice-heavy for `lin`), OOD vs WAXAL.

clean_audio_v3 is more balanced than v2, so not a guaranteed repeat — hence a gated controlled test.

## Fair comparison (identical to the champion)

- **Decision metric:** per-language + pooled **combined** (0.5·WER + 0.5·CER) on
  **`data/processed_generalization_mix` validation (raw/untrimmed audio)**, decoded with beam+LM.
  Test audio is raw, so validation is always raw — never the cleaned/trimmed validation, even
  though this model trains on trimmed audio.
- **Decode params must be re-tuned for the new model** (its logit distribution differs from the
  champion's). Run the alpha/beta sweep on the new checkpoint before comparing — do NOT assume the
  champion's (lin 0.9/0.5, lug 0.4/−0.5, sna 0.7/−0.5) are optimal for it.
- The in-loop `eval_cer` (cleaned validation, greedy) is **monitoring only**, not the gate, and is
  not comparable to the champion's numbers.

## Gates

Automatic: `early_stopping.patience 6` on in-loop `eval_cer`.

Manual raw-val gates at **step 8000, 16000, 24000** (a from-scratch model isn't converged early, so
don't judge before ~8000). **Kill if:**
- By ~step 12000–16000 the raw-val pooled combined is not descending toward the champion's 0.135
  and is clearly plateauing above it.
- Lingala regresses vs champion while others barely move (net wash) — the expected v2 failure mode.
- dot-only / empty / very-short / repeated-n-gram counts rise vs champion.

**Promote only if** the best checkpoint's raw-val pooled combined (with its own tuned decode)
**clearly beats the champion's 0.135**, `lin` not regressed, degenerate-output counts not worse.

## Metric tracking (per checkpoint)

Write to `outputs/analysis/champion_recipe_clean_v3_curve.json`:

| checkpoint | lin comb | lug comb | sna comb | pooled comb | dot-only | empty | very-short | rep-ngram | verdict |
|---|---|---|---|---|---|---|---|---|---|
| champion (baseline) | 0.1711 | 0.1148 | 0.0857 | ~0.135 | _tbd_ | _tbd_ | _tbd_ | _tbd_ | baseline |
| recipe-v3-8000  |  |  |  |  |  |  |  |  |  |
| recipe-v3-16000 |  |  |  |  |  |  |  |  |  |
| recipe-v3-24000 |  |  |  |  |  |  |  |  |  |

- Per-language WER/CER/combined: from the beam+LM sweep on raw validation, per language.
- dot-only / empty / very-short / repeated-n-gram: `scripts/analyze_prediction_distributions.py`
  on the checkpoint's greedy + beam predictions.

## Commands

### Smoke (~2 min — confirms base loads + runs on clean_audio_v3)
```bash
python scripts/train_xlsr_ctc.py \
  --config configs/xlsr_300m_champion_recipe_clean_audio_v3.yaml \
  --max-train-samples 12 --max-eval-samples 6 --max-steps 2 \
  --output-dir checkpoints/xlsr_300m_champion_recipe_clean_audio_v3_smoke
```

### Full recipe run (tmux; ~9–10 h). Put checkpoints on a disk with room (~20GB for 5).
```bash
export HF_DATASETS_CACHE=/dev/shm/hf_datasets_cache
export WAXAL_RAW_DIR=$PWD/google-waxal-asr-challenge20260630-10570-elxebu
# optional: keep checkpoints off root if tight ->
#   mkdir -p /mnt/waxal/ckpt_recipe_v3 && ln -s /mnt/waxal/ckpt_recipe_v3 checkpoints/xlsr_300m_champion_recipe_clean_audio_v3
python scripts/train_xlsr_ctc.py \
  --config configs/xlsr_300m_champion_recipe_clean_audio_v3.yaml
```

### Re-tune decode + gate a checkpoint on raw validation
```bash
python scripts/sweep_kenlm_decode_params.py \
  --checkpoint checkpoints/xlsr_300m_champion_recipe_clean_audio_v3/checkpoint-16000 \
  --dataset-dir data/processed_generalization_mix --split validation \
  --kenlm-dir data/lm_expanded --order 5 \
  --alphas 0.4 0.5 0.6 0.7 0.8 0.9 --betas -0.5 0.0 0.25 0.5 --beam-width 400 \
  --max-samples-per-language 2000 \
  --output outputs/analysis/recipe_v3_ckpt16000_sweep.json
# degenerate-output counts:
# scripts/analyze_prediction_distributions.py on the checkpoint's predictions
```

## Result & recommendation

_Fill after the run: best checkpoint (or "none — champion retained"), the table above, its tuned
decode params, raw-val comparison to champion, and whether it becomes the new champion._
