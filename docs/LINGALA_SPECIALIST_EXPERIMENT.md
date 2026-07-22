# Lingala Specialist Experiment (Step 3 of the top-3 plan)

The headline targeted bet: continue the **champion** on the Lingala-only slice of its own mix.
Config: [xlsr_300m_lin_specialist_from_champion.yaml](../configs/xlsr_300m_lin_specialist_from_champion.yaml).
Rationale in [TOP3_GAP_ANALYSIS.md](TOP3_GAP_ANALYSIS.md) — the entire top-3 gap is lin.

## Design invariants

- **Start:** `champion/checkpoint-24000` with `load_processor_from_checkpoint: true`
  (112-token vocab + trained lm_head preserved — verified pattern from the continuation run).
- **Data:** `data/processed_generalization_mix` filtered `languages: [lin]` → WAXAL-lin 14,400 +
  FLEURS-lin ~4,037, **raw audio** (exactly what the champion saw). **Zero Afrivoice** — the
  read-speech trap is excluded by construction.
- **Deployment: routing-only.** lug/sna forgetting is acceptable; the routed pipeline sends only
  LID-predicted-lin clips (LID 98.3–98.6% on real data) to this model → Phase-2-safe.
- LR 5e-5 (between the dead 2e-5 continuation and the full 3e-4), 4,000 steps ≈ 3.5 epochs,
  bf16, eval/save 500, ES patience 8 (relaxed — let the decay phase finish).
- In-loop eval here IS meaningful (lin slice of the **raw** validation), unlike the cleaned-val
  runs — but the promotion gate is still the beam+LM sweep below.

## Commands

```bash
# smoke (~2 min)
python scripts/train_xlsr_ctc.py \
  --config configs/xlsr_300m_lin_specialist_from_champion.yaml \
  --max-train-samples 12 --max-eval-samples 6 --max-steps 2 \
  --output-dir checkpoints/xlsr_300m_lin_specialist_smoke
# success: "Loaded processor/vocab from champion/checkpoint-24000: 112 tokens", no lm_head MISMATCH

# full (~3-4 h, tmux)
export HF_DATASETS_CACHE=/dev/shm/hf_datasets_cache
python scripts/train_xlsr_ctc.py --config configs/xlsr_300m_lin_specialist_from_champion.yaml
```

### Gate: raw-val lin beam+LM sweep on the best checkpoint

```bash
python scripts/sweep_kenlm_decode_params.py \
  --checkpoint checkpoints/xlsr_300m_lin_specialist_from_champion/checkpoint-<BEST> \
  --dataset-dir data/processed_generalization_mix --split validation --languages lin \
  --kenlm-dir data/lm_expanded --order 5 \
  --alphas 0.6 0.7 0.8 0.9 1.0 --betas 0.0 0.25 0.5 0.75 --beam-width 400 \
  --max-samples-per-language 2000 \
  --output outputs/analysis/lin_specialist_sweep.json
```
(Use `data/lm_expanded_v2`/order 6 instead if Step 1 produced a better lin LM.)

## Decision thresholds (champion lin beam+LM = 0.1711)

| specialist lin combined | verdict |
|---|---|
| ≥ 0.171 | fail — discard; proceed to Step 4 decision |
| < 0.166 | pass — route lin to specialist, build submission candidate |
| < 0.155 | strong — submit same day |
| < 0.140 | major — likely top-5 move on its own |

Also check vs champion: dot-only / very-short / repeated-ngram counts must not rise, and greedy
lin combined should improve too (greedy 0.2647 baseline) — if only beam+LM improves, the gain is
fragile.

## Validation curve (fill in per checkpoint)

| checkpoint | in-loop eval_cer (lin raw val, greedy) | beam+LM combined (sweep) |
|---|---|---|
| champion baseline | ~0.159 (greedy CER) | 0.1711 |
| 500 | | |
| 1000 | | |
| 1500 | | |
| 2000 | | |
| 2500 | | |
| 3000 | | |
| 3500 | | |
| 4000 | | |

## If it passes: routed submission build

```bash
# lin test decode from the SPECIALIST (params from its own sweep)
python scripts/run_xlsr_inference.py --checkpoint checkpoints/xlsr_300m_lin_specialist_from_champion/checkpoint-<BEST> \
  --dataset-dir data/processed_generalization_mix --split test --languages lin \
  --decoder-mode beam_lm --kenlm-model data/lm_expanded/lin_5gram.binary \
  --unigrams-file data/lm_expanded/lin.txt --alpha <A> --beta <B> --beam-width 400 \
  --output outputs/predictions/lin_specialist_test.csv

# lug/sna decodes: reuse outputs/predictions/champ_exp2_{lug,sna}_test.csv (champion, unchanged)
python scripts/merge_predictions.py \
  --predictions outputs/predictions/lin_specialist_test.csv \
  --predictions outputs/predictions/champ_exp2_lug_test.csv \
  --predictions outputs/predictions/champ_exp2_sna_test.csv \
  --raw-dir "$WAXAL_RAW_DIR" --output outputs/predictions/spec_routed_test.csv
python scripts/postprocess_predictions.py --predictions outputs/predictions/spec_routed_test.csv \
  --output outputs/predictions/spec_routed_test_pp.csv
python scripts/make_submission.py --predictions outputs/predictions/spec_routed_test_pp.csv \
  --model-name lin_specialist_routed --empty-target "." \
  --output outputs/submissions/lin_specialist_routed.csv
python scripts/validate_submission.py --submission outputs/submissions/lin_specialist_routed.csv
```

Phase-2 note: for the Phase-2 test the same specialist plugs into `run_no_metadata_pipeline.py`
routing (LID-predicted lin → specialist logits). Champion + 0.861 remain selected until this
candidate's public score beats them.

## Result

_Fill after the run: best checkpoint, gate table, routed-submission decision._
