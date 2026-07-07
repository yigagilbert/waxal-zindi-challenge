# Next Training Run

Date: 2026-07-05

## Status

Superseded for the current objective.

This runbook describes a WAXAL-clean-only model. The current objective is a single-stage model that generalizes across broader audio conditions for Phase 2. Use:

```text
docs/GENERALIZATION_MIX_RUN.md
configs/xlsr_300m_generalization_mix.yaml
```

instead.

## Objective

Train one Phase 2-ready multilingual XLS-R model using official WAXAL data only, with:

- Sunbird-assisted Luganda cleaning,
- Alvin-assisted Lingala filtering,
- existing clean Shona bucket,
- no external data yet.

This run is the direct successor to:

```text
configs/xlsr_300m_balanced_sunbird_lug_all_v2.yaml
```

## Why This Run

The latest public result showed that the current XLS-R v2 model generalizes reasonably, but Lingala remains the main weakness.

Alvin validation on WAXAL Lingala was meaningfully better than checkpoint-6000:

| Model | WER | CER | Combined Error | Zindi-Style Score |
|---|---:|---:|---:|---:|
| `xlsr_v2_ckpt6000` | 0.413917 | 0.179602 | 0.296760 | 0.703240 |
| `alvin_xlsr_lingala` | 0.360817 | 0.140890 | 0.250853 | 0.749147 |

Alvin also reduced Lingala dot-only outputs on validation from `96` to `0`.

We are using Alvin as a teacher/filter for Lingala, not as a routed final model and not as an automatic label replacement system.

## Training Data

Primary training manifest:

```text
data/quality/clean_train_alvin_lingala_v1.csv
```

Manifest summary from the GPU run:

| Item | Count |
|---|---:|
| Base clean rows before Alvin Lingala filtering | 31,624 |
| Base non-Lingala rows kept | 18,570 |
| Lingala diagnostic rows | 14,400 |
| Clean Lingala rows kept | 11,481 |
| Lingala manual-review rows withheld | 1,451 |
| Lingala suspicious rows withheld | 1,468 |
| Final combined manifest rows | 30,051 |

Label policy:

- keep original WAXAL labels for clean Lingala,
- do not use Alvin labels yet,
- do not include manual-review Lingala rows,
- do not include suspicious Lingala rows,
- keep previous Sunbird-assisted Luganda strategy,
- keep clean Shona strategy.

## External Data Decision

Do not use `KasuleTrevor/Lingala_100hrs` in this next training run.

The sampled audit found:

- no dataset-level license in dataset info,
- no source-level license confirmation,
- sources include `Afrivoice`, `LRSC`, and `fleurs`,
- 243 possible text overlaps with WAXAL Lingala train,
- no Phase 1 test ID overlap in the sampled audit.

This dataset remains research-only until license and leakage checks are complete.

## Config

Use:

```text
configs/xlsr_300m_balanced_alvin_lingala_all.yaml
```

Key settings:

| Setting | Value |
|---|---|
| Base model | `facebook/wav2vec2-xls-r-300m` |
| Train manifest | `data/quality/clean_train_alvin_lingala_v1.csv` |
| Languages | all |
| Sampling | language-balanced |
| Normalization | `language_safe` |
| Feature encoder | frozen |
| Precision | fp16 |
| Gradient checkpointing | true |
| Max steps | 6,000 |
| Eval/save interval | 500 |
| Best-model metric | `eval_cer` |

## Commands

Run smoke first:

```bash
export WAXAL_NO_SYNC=1
WAXAL_NO_SYNC=1 make train-xlsr-alvin-lingala-all-smoke
```

If smoke passes, run full training:

```bash
WAXAL_NO_SYNC=1 make train-xlsr-alvin-lingala-all
```

Run validation inference:

```bash
WAXAL_NO_SYNC=1 make xlsr-alvin-lingala-all-validation
```

Evaluate and compare against v2:

```bash
WAXAL_NO_SYNC=1 make eval-xlsr-alvin-lingala-all
WAXAL_NO_SYNC=1 make analyze-xlsr-alvin-lingala-all-validation
```

Only after validation is stronger, run test inference:

```bash
WAXAL_NO_SYNC=1 make xlsr-alvin-lingala-all-test
WAXAL_NO_SYNC=1 make analyze-xlsr-alvin-lingala-all-test
```

Create a submission only after the validation and test sanity reports are acceptable:

```bash
WAXAL_NO_SYNC=1 make submission-xlsr-alvin-lingala-all
```

## Success Criteria

The run is a win if it improves or preserves global validation while clearly improving Lingala:

- overall combined error improves vs current v2,
- Lingala WER/CER improves,
- Lingala dot-only/very-short outputs decrease,
- Luganda does not regress meaningfully,
- Shona does not regress meaningfully,
- test prediction distribution stays sane.

Minimum useful outcome:

- Lingala improves and global validation is not worse by more than a very small margin.

Strong outcome:

- global validation improves,
- Lingala improves materially,
- public score improves after one carefully chosen submission.

## Early Stop Criteria

Stop or avoid a long continuation if by step 1500-2500:

- eval CER is flat or worse than v2 trend,
- Lingala remains dot/short-heavy,
- one language collapses while others improve,
- decoded samples show severe repetition or blanking.

## Artifacts To Back Up

Before destroying the instance, save:

```text
checkpoints/xlsr_300m_balanced_alvin_lingala_all/
configs/xlsr_300m_balanced_alvin_lingala_all.yaml
outputs/experiments/
outputs/predictions/
outputs/analysis/
outputs/submissions/
data/quality/clean_train_alvin_lingala_v1.csv
outputs/lingala_teacher/
outputs/lingala_models/
```
