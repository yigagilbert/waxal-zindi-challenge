# Checkpoint Generalization Analysis

Date: 2026-07-05

## Executive Summary

The apparent local-validation vs public-leaderboard gap was mostly a metric interpretation issue, not a catastrophic generalization failure.

Zindi displayed this public result for `submission_xlsr_v2_ckpt6000_20260705T071924Z.csv`:

| Metric | Value |
|---|---:|
| Public score | 0.766027346 |
| Public CER | 0.119741275 |
| Public WER | 0.348204032 |

The public score is consistent with:

```text
1 - 0.5 * (WER + CER)
= 1 - 0.5 * (0.348204032 + 0.119741275)
= 0.7660273465
```

Our internal `combined` metric is an error metric:

```text
combined_error = 0.5 * (WER + CER)
```

So the public submission corresponds to:

```text
public_combined_error = 1 - 0.766027346 = 0.233972654
```

That is close to the final local validation error:

| Split | WER | CER | Combined Error | Zindi-Style Score |
|---|---:|---:|---:|---:|
| Local validation, checkpoint-6000 | 0.346210 | 0.107953 | 0.227082 | 0.772918 |
| Public leaderboard, checkpoint-6000 | 0.348204 | 0.119741 | 0.233973 | 0.766027 |

Conclusion: checkpoint-6000 generalized reasonably. There is no evidence of a massive validation/public gap.

## Current Public Submission

Submitted file:

```text
submission_xlsr_v2_ckpt6000_20260705T071924Z.csv
```

Rank at time of screenshot:

```text
21
```

Important interpretation:

- Higher Zindi score is better.
- Our local `combined` is lower-is-better error.
- Compare `1 - local_combined` against Zindi public score.

## Checkpoint Comparison Plan

Even though checkpoint-6000 generalized reasonably, checkpoint comparison is still useful because validation gains after step 5000 were tiny:

| Checkpoint | Local Training-Log Combined Error | Notes |
|---|---:|---|
| checkpoint-5000 | about 0.22714 | Strong; likely close to final |
| checkpoint-5500 | about 0.22709 | Slightly better |
| checkpoint-6000 | about 0.22708 | Best local validation; submitted |

Because the differences are tiny, lower checkpoints should be evaluated for prediction sanity before spending more submissions.

## Required Commands

Generate validation predictions for all v2 checkpoints:

```bash
WAXAL_NO_SYNC=1 make xlsr-v2-validation-all
```

Analyze validation metrics and prediction distributions:

```bash
WAXAL_NO_SYNC=1 make analyze-xlsr-v2-validation-all
```

Generate test predictions for all v2 checkpoints:

```bash
WAXAL_NO_SYNC=1 make xlsr-v2-test-all
```

Analyze test prediction distributions:

```bash
WAXAL_NO_SYNC=1 make analyze-xlsr-v2-test-all
```

Create deterministic submission files:

```bash
WAXAL_NO_SYNC=1 make submissions-xlsr-v2-all
```

Validate any candidate submission before upload:

```bash
uv run --no-sync scripts/validate_submission.py \
  --submission outputs/submissions/submission_xlsr_v2_ckpt5000.csv \
  --raw-dir "$WAXAL_RAW_DIR"
```

Repeat for `ckpt5500` and `ckpt6000`.

## Test Distribution Sanity For Submitted Checkpoint-6000

Local analysis of the downloaded submitted CSV:

| Statistic | Value |
|---|---:|
| Rows | 4,253 |
| Empty predictions | 0 |
| Overall mean chars | 170.83 |
| Overall median chars | 168 |
| Overall mean words | 24.94 |
| Repeated ngram count | 15 |
| Repeated run >= 3 count | 6 |
| Unusual character count | 0 |
| Very short outputs | 166 |
| Very long outputs | 1 |

By language:

| Language | Rows | Mean Chars | Median Chars | Very Short | Repeated Ngram |
|---|---:|---:|---:|---:|---:|
| lin | 1,866 | 144.36 | 147 | 128 | 15 |
| lug | 638 | 201.18 | 188 | 0 | 0 |
| sna | 1,749 | 188.00 | 183 | 38 | 0 |

Main weakness found:

- Lingala has many `.` or near-empty predictions.
- All repeated-ngram cases are also concentrated in Lingala.
- Luganda looks much healthier after Sunbird-assisted cleaning.

This suggests the next improvement path is probably Lingala-specific, not more Luganda cleaning.

## Recommendation Before Another Submission

Do not blindly submit every checkpoint.

First compare these files:

```text
outputs/analysis/xlsr_v2_validation_checkpoint_analysis.json
outputs/analysis/xlsr_v2_test_checkpoint_analysis.json
```

Submit checkpoint-5000 or checkpoint-5500 only if it clearly reduces one of:

- Lingala `.` / very-short outputs,
- repeated Lingala phrases,
- suspicious test-length distribution,
- validation per-language degradation.

If checkpoint-5000 has materially fewer very-short Lingala predictions while validation score is nearly identical, it is a plausible next submission. Otherwise checkpoint-6000 remains the best-known model.

## Downloaded Test Submission Comparison

The downloaded checkpoint submissions were compared on 2026-07-05:

```text
/Users/sunbird/Downloads/submission_xlsr_v2_ckpt5000.csv
/Users/sunbird/Downloads/submission_xlsr_v2_ckpt5500.csv
/Users/sunbird/Downloads/submission_xlsr_v2_ckpt6000.csv
```

All three files are structurally valid:

| Checkpoint | Rows | Duplicate IDs | Missing IDs | Extra IDs | Sample Order |
|---|---:|---:|---:|---:|---|
| checkpoint-5000 | 4,253 | 0 | 0 | 0 | yes |
| checkpoint-5500 | 4,253 | 0 | 0 | 0 | yes |
| checkpoint-6000 | 4,253 | 0 | 0 | 0 | yes |

Overall test prediction sanity:

| Checkpoint | Dot-Only | Very Short | Repeated Ngram | Repeat Run >= 3 | Mean Chars | Mean Words |
|---|---:|---:|---:|---:|---:|---:|
| checkpoint-5000 | 124 | 166 | 16 | 4 | 171.26 | 25.20 |
| checkpoint-5500 | 130 | 166 | 12 | 6 | 171.04 | 25.01 |
| checkpoint-6000 | 124 | 166 | 15 | 6 | 170.83 | 24.94 |

By-language sanity:

| Checkpoint | Lang | Dot-Only | Very Short | Repeated Ngram | Repeat Run >= 3 | Mean Chars |
|---|---|---:|---:|---:|---:|---:|
| 5000 | lin | 111 | 128 | 16 | 2 | 144.51 |
| 5500 | lin | 118 | 128 | 12 | 2 | 144.49 |
| 6000 | lin | 111 | 128 | 15 | 3 | 144.36 |
| 5000 | lug | 0 | 0 | 0 | 2 | 201.64 |
| 5500 | lug | 0 | 0 | 0 | 4 | 201.54 |
| 6000 | lug | 0 | 0 | 0 | 3 | 201.18 |
| 5000 | sna | 13 | 38 | 0 | 0 | 188.71 |
| 5500 | sna | 12 | 38 | 0 | 0 | 188.23 |
| 6000 | sna | 13 | 38 | 0 | 0 | 188.00 |

Pairwise changes are common because later checkpoints adjust wording, but severe distribution changes are rare:

| Pair | Changed lin/lug/sna | Strong Diff lin/lug/sna | Interpretation |
|---|---|---|---|
| 5000 -> 5500 | 1446 / 486 / 1334 | 3 / 0 / 0 | Many text changes, only 3 severe length shifts |
| 5500 -> 6000 | 1205 / 389 / 1083 | 0 / 0 / 1 | Mostly stable |
| 5000 -> 6000 | 1417 / 491 / 1347 | 3 / 0 / 0 | Mostly stable |

No checkpoint clearly fixes the Lingala short-output issue. Lingala has exactly 128 very-short outputs for all three checkpoints, and checkpoint-5000 and checkpoint-6000 tie on Lingala dot-only outputs.

## Candidate Hybrid Submission

A conservative per-language route was created and validated:

```text
outputs/submissions/submission_xlsr_v2_route_lin6000_lug5000_sna5500.csv
```

Routing:

| Language | Source Checkpoint | Reason |
|---|---|---|
| lin | checkpoint-6000 | Best known public score and tied-best Lingala dot-only count |
| lug | checkpoint-5000 | Slightly fewer repeat-run issues than later checkpoints |
| sna | checkpoint-5500 | One fewer Shona dot-only output than checkpoint-5000/6000 |

The hybrid file passed local submission validation:

| Check | Result |
|---|---|
| Rows | 4,253 |
| Missing IDs | 0 |
| Extra IDs | 0 |
| Duplicate IDs | 0 |
| Empty targets | 0 |
| Aligned to sample | yes |

Hybrid sanity:

| Lang | Very Short | Repeated Ngram | Repeat Run >= 3 | Mean Chars |
|---|---:|---:|---:|---:|
| lin | 128 | 15 | 3 | 144.36 |
| lug | 0 | 0 | 2 | 201.64 |
| sna | 38 | 0 | 0 | 188.23 |
| overall | 166 | 15 | 5 | 171.00 |

Recommendation: submit the hybrid only if spending one leaderboard submission is acceptable. It is a low-risk, small-expected-gain candidate. If submissions are scarce, keep checkpoint-6000 as the best proven checkpoint and spend effort on Lingala diagnostics instead.

## Next Training Direction

The XLS-R v2 model is competitive but still has a Lingala weakness. Recommended next experiments:

1. Lingala-focused diagnostics:
   - inspect short Lingala validation/test predictions,
   - compare duration/audio quality for Lingala blank predictions,
   - run a Lingala teacher if competition/license-safe.
2. Per-language routing:
   - keep checkpoint-6000 for Luganda and Shona,
   - test checkpoint-5000/5500 for Lingala.
3. Train a Lingala-specific or Lingala-upweighted XLS-R run only if checkpoint analysis confirms Lingala is the main bottleneck.

Do not run another expensive full multilingual training run until the checkpoint comparison and Lingala error analysis are complete.
