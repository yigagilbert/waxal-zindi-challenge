# Lingala Model Comparison

This report is generated from local WAXAL Lingala validation/test predictions.

## Status

Pending execution on the GPU instance.

## Required Commands

Run Alvin first:

```bash
export WAXAL_NO_SYNC=1
WAXAL_NO_SYNC=1 make alvin-lingala-access
WAXAL_NO_SYNC=1 make xlsr-sunbird-lug-v2-validation
WAXAL_NO_SYNC=1 make alvin-lingala-validation
WAXAL_NO_SYNC=1 make compare-alvin-lingala-validation
```

If Alvin passes validation sanity, run train and test diagnostics:

```bash
WAXAL_NO_SYNC=1 make alvin-lingala-train-teacher
WAXAL_NO_SYNC=1 make lingala-alvin-diagnostics
WAXAL_NO_SYNC=1 make build-lingala-alvin-manifest
WAXAL_NO_SYNC=1 make xlsr-v2-test-all
WAXAL_NO_SYNC=1 make alvin-lingala-test
WAXAL_NO_SYNC=1 make compare-alvin-lingala-test
```

## Metrics To Compare

For validation:

- WER,
- CER,
- combined error,
- Zindi-style score: `1 - combined_error`,
- dot-only count,
- blank/empty count,
- very-short output count,
- repeated-ngram count,
- mean/median prediction length,
- prediction/reference length ratio,
- examples where Alvin strongly improves over checkpoint-6000,
- examples where checkpoint-6000 strongly beats Alvin.

For test sanity:

- dot-only count,
- very-short count,
- repeated-ngram count,
- output length distribution,
- unusual characters,
- CTC blank ratio if token stats are available.

## Decision Rules

Use Alvin for direct Lingala routing only if:

- WAXAL Lingala validation WER/CER are better than checkpoint-6000, or
- validation is similar but dot-only/very-short outputs are substantially lower, and
- Phase 1 test Lingala output distribution looks sane.

Use Alvin as a teacher only if:

- validation text quality is plausible,
- teacher outputs are not blank/dot-heavy,
- disagreement flags identify plausible WAXAL label/audio problems.

Fine-tune from Alvin only if:

- Alvin beats or clearly complements checkpoint-6000 on validation,
- teacher diagnostics do not show systematic normalization mismatch,
- the clean Alvin Lingala manifest retains enough examples.

Do not submit an Alvin-routed leaderboard file until this document has real validation/test numbers.
