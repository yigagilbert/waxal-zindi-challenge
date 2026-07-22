# Model-Zoo Oracle Analysis

Bounds the routing/ensemble upside **before** any more training: the per-sample oracle
(best candidate per sample by CER) is the ceiling of any router built from those candidates.
Tool: [scripts/build_prediction_router.py](../scripts/build_prediction_router.py).

_2026-07-20._

## Honest inventory: what candidates actually exist

The historical zoo — Alvin-Lingala, noirlab Whisper-lin, xlsr-v2-ckpt6000, XLS-R 1B — **is
gone**: checkpoints and prediction CSVs lived on the old Vast box and `outputs/` was never
git-tracked. The lost teacher-cleaned Lingala manifest (`clean_train_alvin_lingala_v1.csv`) is
part of the same loss. Regenerable candidates today:

| candidate | how | status |
|---|---|---|
| champion routed beam+LM | already on disk | `outputs/predictions/nometa_validation_expanded.csv` |
| champion greedy | 1 GPU pass (~15 min) | `run_xlsr_inference.py --split validation --decoder-mode greedy` |
| recipe-v3 ckpt-6500 (worse overall, possibly complementary per-sample) | kept checkpoint; 1 GPU pass | optional |
| lin specialist (if trained) | this plan's experiment #3 | future |

## Commands

```bash
# 1. generate the greedy candidate (validation)
python scripts/run_xlsr_inference.py --checkpoint champion/checkpoint-24000 \
  --dataset-dir data/processed_generalization_mix --split validation \
  --decoder-mode greedy --batch-size 8 \
  --output outputs/predictions/champ_greedy_validation.csv

# 2. oracle
python scripts/build_prediction_router.py --mode oracle \
  --predictions champion_beam=outputs/predictions/nometa_validation_expanded.csv \
  --predictions champion_greedy=outputs/predictions/champ_greedy_validation.csv \
  --references data/processed_generalization_mix/validation.csv \
  --output outputs/analysis/model_zoo_oracle_validation.json

# 3. (only if oracle_upside >= 0.005) tune + evaluate the rules router with CV
python scripts/build_prediction_router.py --mode route \
  --predictions champion_beam=outputs/predictions/nometa_validation_expanded.csv \
  --predictions champion_greedy=outputs/predictions/champ_greedy_validation.csv \
  --references data/processed_generalization_mix/validation.csv \
  --kenlm-dir data/lm_expanded \
  --output outputs/analysis/router_validation_report.json
```

## Decision gates

- `overall.oracle_upside` (best single − oracle) **≥ 0.005** → routing has real headroom; tune
  the rules router; apply to test only if the CV `gain` ≥ 0.003.
- `oracle_upside < 0.005` → routing cannot move the score; the remaining gap is acoustic
  (consistent with TOP3_GAP_ANALYSIS: it's Lingala) — the lin specialist matters more.
- When the **lin specialist** exists, rerun the oracle with it as a candidate: that number is
  the honest preview of the specialist-routed submission before spending a Zindi slot.

## Results

_Fill after running:_

| candidate set | best single | oracle | upside | router CV gain | verdict |
|---|---|---|---|---|---|
| beam+LM vs greedy | | | | | |
| + recipe-6500 | | | | | |
| + lin specialist | | | | | |
