# Lingala Recovery Plan

Date: 2026-07-05

## Current Diagnosis

The XLS-R v2 public result is not a failure. Zindi's public score of about `0.7660` corresponds to a combined error of about `0.234`, close to local validation combined error around `0.227`.

The main weakness is language-specific:

- Lingala has many dot-only predictions.
- Lingala has many very-short predictions.
- Repeated-ngram issues are concentrated in Lingala.
- Luganda looks healthier after Sunbird-assisted cleaning.
- Shona is acceptable but should still be monitored.

## New Priority

Evaluate:

```text
Alvin-Nahabwe/wav2vec2-xls-r-300m-Fleurs_AMMI_AFRIVOICE_LRSC-ln-109hrs-v2
```

Why:

- It is a Lingala ASR model.
- It uses the same XLS-R 300M family as our current best WAXAL model.
- It is Apache-2.0 per the user-provided model details.
- It reports strong Lingala metrics.
- It is likely more relevant to the current bottleneck than generic Whisper.

## Execution Order

1. Back up current artifacts.
2. Confirm Alvin gated-model access.
3. Run Alvin on WAXAL Lingala validation.
4. Compare Alvin vs checkpoint-6000 on WAXAL Lingala validation.
5. Run Alvin on WAXAL Lingala train for teacher diagnostics.
6. Build Alvin-assisted Lingala manifest.
7. Run Alvin on WAXAL Lingala test for sanity comparison.
8. Decide between routing, fine-tuning, filtering, or external-data warm-start.

Current decision after the Alvin train-teacher pass:

- do not route separate models for the final strategy,
- do not use Alvin labels as replacements yet,
- do not use `KasuleTrevor/Lingala_100hrs` yet,
- train one multilingual XLS-R 300M model using `data/quality/clean_train_alvin_lingala_v1.csv`.

Next config:

```text
configs/xlsr_300m_balanced_alvin_lingala_all.yaml
```

Runbook:

```text
docs/NEXT_TRAINING_RUN.md
```

## Commands

```bash
export WAXAL_NO_SYNC=1
uv run --no-sync hf auth login
WAXAL_NO_SYNC=1 make alvin-lingala-access
WAXAL_NO_SYNC=1 make xlsr-sunbird-lug-v2-validation
WAXAL_NO_SYNC=1 make alvin-lingala-validation
WAXAL_NO_SYNC=1 make compare-alvin-lingala-validation
WAXAL_NO_SYNC=1 make alvin-lingala-train-teacher
WAXAL_NO_SYNC=1 make lingala-alvin-diagnostics
WAXAL_NO_SYNC=1 make build-lingala-alvin-manifest
WAXAL_NO_SYNC=1 make xlsr-v2-test-all
WAXAL_NO_SYNC=1 make alvin-lingala-test
WAXAL_NO_SYNC=1 make compare-alvin-lingala-test
```

## Decision Options

### A. Direct Lingala Routing

Use Alvin for Lingala test predictions and current XLS-R v2 for Luganda/Shona.

Use if:

- Alvin beats checkpoint-6000 on WAXAL Lingala validation, and
- Alvin has far fewer dot-only/very-short test outputs, and
- output length distribution is sane.

Risk:

- Domain mismatch or normalization mismatch can hurt hidden score.

### B. Alvin-Initialized WAXAL Lingala Fine-Tune

Start from Alvin and fine-tune on WAXAL Lingala clean data.

Config:

```text
configs/xlsr_300m_alvin_lingala_waxal_finetune.yaml
```

Use if:

- Alvin is strong but not perfectly matched to WAXAL style,
- teacher diagnostics show WAXAL Lingala labels are mostly usable,
- direct routing is promising but imperfect.

### C. Alvin-Assisted Filtering Only

Use Alvin to remove or review suspicious Lingala rows, then retrain multilingual XLS-R from generic base.

Use if:

- Alvin finds many suspicious labels,
- Alvin direct predictions are not good enough for routing,
- model disagreement appears to diagnose training noise.

This is the selected next run. It is more Phase 2-safe than a routed submission because it produces one multilingual model.

### D. External Warm-Start

Use safe parts of `KasuleTrevor/Lingala_100hrs` or source datasets only after licensing and leakage checks pass.

Use if:

- Alvin confirms external Lingala data improves WAXAL validation,
- source licenses are final-solution safe,
- no Phase 1 leakage risk is found.

### E. Mixed Low-Ratio External Data

Mix WAXAL with external Lingala at low ratio.

Use only after warm-start shows benefit.

## Early Stop Criteria

Stop a Lingala fine-tune if by checkpoint 1000-1500:

- Lingala validation WER/CER are worse than checkpoint-6000,
- dot-only/very-short counts do not improve,
- output length collapses,
- CER improves but WER worsens heavily,
- Luganda/Shona routing plan becomes more complicated than the gain justifies.

## Submission Criteria

Spend a leaderboard submission only if:

- validation improves or sanity improves strongly,
- test Lingala dot-only/very-short count drops materially,
- no new weird-character or repetition issue appears,
- submission file validates exactly against sample order.
