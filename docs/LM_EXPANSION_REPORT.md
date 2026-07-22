# Lingala LM Expansion Report (Step 1 of the top-3 plan)

Goal: shrink lin's beam+LM combined below the champion's **0.1711** using only license-safe
text. lin is 43.9% of the test and the entire top-3 gap ([TOP3_GAP_ANALYSIS.md](TOP3_GAP_ANALYSIS.md)).

## Text sources (all disclosed; no Phase-1 test labels anywhere)

| source | license | lines (fill) | notes |
|---|---|---|---|
| existing `data/lm_expanded/lin.txt` | mixed, already disclosed | 54,055 | WAXAL train (2×) + FLEURS + Afrivoice manifests |
| Wikipedia Lingala (`wikimedia/wikipedia`, `20231101.ln`) | CC-BY-SA-4.0 | _tbd_ | sentence-split articles; general lexical coverage |

## Build commands (box; CPU-only; existing LMs in data/lm_expanded are NOT touched)

```bash
git pull
# order 5 with wiki added (merges the existing corpus; harvests only the wiki source for lin)
python scripts/collect_lm_text.py --languages lin \
  --merge-corpus-dir data/lm_expanded \
  --skip-source google/fleurs --skip-source Sunbird/salt \
  --skip-source yigagilbert/luganda-speech-cv-yogera-filtered \
  --skip-source DigitalUmuganda/Afrivoice \
  --max-lines-per-source 60000 \
  --output-dir data/lm_expanded_v2 --order 5 --overwrite

# order 6 variant from the same corpus
python scripts/collect_lm_text.py --languages lin \
  --merge-corpus-dir data/lm_expanded_v2 --skip-hf \
  --output-dir data/lm_expanded_v2_o6 --order 6 --overwrite
```

(`--skip-source` entries prevent re-harvesting text already inside the merged corpus;
only Wikipedia is newly pulled. lmplz needs to be on PATH — same box setup as before.)

## Sweep (lin only, vs champion setting)

```bash
# order 5, expanded corpus
python scripts/sweep_kenlm_decode_params.py \
  --checkpoint champion/checkpoint-24000 \
  --dataset-dir data/processed_generalization_mix --split validation --languages lin \
  --kenlm-dir data/lm_expanded_v2 --order 5 \
  --alphas 0.7 0.8 0.9 1.0 --betas 0.0 0.25 0.5 0.75 --beam-width 400 \
  --max-samples-per-language 2000 \
  --output outputs/analysis/lm_v2_lin_sweep_o5.json

# order 6
python scripts/sweep_kenlm_decode_params.py \
  --checkpoint champion/checkpoint-24000 \
  --dataset-dir data/processed_generalization_mix --split validation --languages lin \
  --kenlm-dir data/lm_expanded_v2_o6 --order 6 \
  --alphas 0.7 0.8 0.9 1.0 --betas 0.0 0.25 0.5 0.75 --beam-width 400 \
  --max-samples-per-language 2000 \
  --output outputs/analysis/lm_v2_lin_sweep_o6.json
```

## Results (fill in)

Baseline: champion + `data/lm_expanded` 5-gram, lin best (α0.9, β0.5, beam400) =
**WER 0.2136 / CER 0.1285 / combined 0.1711**.

| LM | best (α,β) | WER | CER | combined | Δ vs 0.1711 |
|---|---|---|---|---|---|
| expanded_v2 5-gram | | | | | |
| expanded_v2 6-gram | | | | | |

Degenerate-output check on the winning decode (`analyze_prediction_distributions.py` or the
router script's flags): dot-only ___ / very-short ___ / repeated-ngram ___ (champion baseline:
fill from the same tool for a fair count). CER must not get worse (unigram list grows with wiki
vocabulary; watch for over-eager lexicon substitutions).

## Gate

- combined < **0.1690** (≥0.002 real gain): adopt for lin; redecode lin test with the new LM +
  params, rebuild merged submission (lug/sna decodes unchanged), validate, submit as
  `champ_lmv2` candidate.
- 0.1690–0.1711: keep as decode option; do not spend a submission slot on it alone (bundle with
  the specialist if that lands).
- worse: discard; the lexical-coverage hypothesis is exhausted — specialist only.
