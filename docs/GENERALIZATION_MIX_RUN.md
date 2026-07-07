# Generalization Mix Training Run

Date: 2026-07-05

## Objective

Train one single-stage multilingual ASR model that generalizes beyond WAXAL Phase 1 audio conditions while still preserving WAXAL validation performance.

This replaces the WAXAL-only next-run plan as the preferred direction when Phase 2 audio conditions are unknown.

## Model

```text
facebook/wav2vec2-xls-r-300m
```

Config:

```text
configs/xlsr_300m_generalization_mix.yaml
```

## Training Data Philosophy

Use a single training stage, but mix sources deliberately:

- WAXAL clean official data anchors the model to challenge text style.
- FLEURS adds clean multilingual read speech across Luganda, Lingala, and Shona.
- Sunbird SALT adds extra Luganda speech diversity.
- `KasuleTrevor/Lingala_100hrs` can add Lingala diversity, but only with an explicit risk flag because its dataset-level license is missing and overlap checks are not complete.

## Safe Mix

This is the recommended first mixed-data run:

```text
WAXAL clean data
+ FLEURS Luganda/Lingala/Shona
+ Sunbird SALT Luganda
```

Prepare:

```bash
export WAXAL_NO_SYNC=1
WAXAL_NO_SYNC=1 make prepare-generalization-mix-safe
```

Output:

```text
data/processed_generalization_mix/hf_dataset
data/processed_generalization_mix/generalization_mix_report.json
```

## All-Source Mix

This includes `KasuleTrevor/Lingala_100hrs`:

```bash
export WAXAL_NO_SYNC=1
WAXAL_NO_SYNC=1 make prepare-generalization-mix-all
```

This command intentionally passes:

```text
--allow-unverified-lingala-100hrs
```

Use it only if you explicitly accept the unresolved risk:

- dataset-level license missing from Hugging Face metadata,
- source-level licenses not fully verified,
- sampled audit found possible text overlap with WAXAL Lingala train,
- Phase 1 leakage risk needs deeper review.

## Optional Source Caps

To prevent external data from dominating the single-stage run:

```bash
FLEURS_MAX_PER_LANGUAGE=3000 \
SALT_MAX=5000 \
LINGALA_100HRS_MAX=10000 \
WAXAL_NO_SYNC=1 make prepare-generalization-mix-all
```

For the safe mix:

```bash
FLEURS_MAX_PER_LANGUAGE=3000 \
SALT_MAX=5000 \
WAXAL_NO_SYNC=1 make prepare-generalization-mix-safe
```

## Training

Smoke:

```bash
WAXAL_NO_SYNC=1 make train-xlsr-generalization-mix-smoke
```

Full:

```bash
WAXAL_NO_SYNC=1 make train-xlsr-generalization-mix
```

## Evaluation

Validation:

```bash
WAXAL_NO_SYNC=1 make xlsr-generalization-mix-validation
WAXAL_NO_SYNC=1 make eval-xlsr-generalization-mix
WAXAL_NO_SYNC=1 make analyze-xlsr-generalization-mix-validation
```

Test sanity only after validation is acceptable:

```bash
WAXAL_NO_SYNC=1 make xlsr-generalization-mix-test
WAXAL_NO_SYNC=1 make analyze-xlsr-generalization-mix-test
```

Submission only after validation and sanity pass:

```bash
WAXAL_NO_SYNC=1 make submission-xlsr-generalization-mix
```

## Success Criteria

The model should be judged on both WAXAL accuracy and robustness:

- overall validation combined error vs XLS-R v2,
- per-language WER/CER,
- Lingala dot-only count,
- very-short output count,
- repeated-ngram count,
- prediction length distribution,
- noisy/low-energy subset performance,
- long-audio subset performance.

The run is successful if it improves robustness while preserving or improving WAXAL validation.

## Important Rule Notes

No public WAXAL Phase 1 test labels may be used.

External datasets must be:

- publicly accessible,
- license-compatible,
- documented in final solution,
- free of challenge-test label leakage.

FLEURS and SALT are more straightforward. `Lingala_100hrs` is promising but remains a deliberate risk until license/source overlap checks are signed off.
