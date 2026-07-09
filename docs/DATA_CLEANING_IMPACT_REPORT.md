# Data Cleaning Impact Report

Status: TEMPLATE — populate from `outputs/data_quality/cleaning_impact_report.json`
after running `make clean-trim-audio` (the JSON is the source of truth; this doc
is the narrative summary for code review).

Preprocessing version: `clean_audio_v1`
Git commit: (from JSON `git_commit`)

## Before / after

| Metric | Before | After (clean bucket) |
|---|---:|---:|
| Train rows | (JSON train.rows_before) | (JSON train.bucket_counts.clean) |
| Train audio hours | (JSON train.original_hours) | (JSON train.cleaned_hours) |
| Edge silence removed (hours) | — | (JSON train.silence_removed_hours) |

## Bucket counts

| Bucket | Rows | lin | lug | sna |
|---|---:|---:|---:|---:|
| clean | | | | |
| medium | | | | |
| noisy | | | | |
| excluded | | | | |

(from JSON `train.bucket_counts` / `train.bucket_counts_by_language`)

## Exclusion reasons

(from JSON `train.excluded_reasons` — e.g. short_audio_long_transcript,
likely_empty_audio, chars_per_second_unrealistic, audio_transcript_mismatch,
too_short_after_trim, empty_transcript, failed_decode)

## Validation

Full validation kept in the DatasetDict. Excluded-for-analysis rows:
(JSON `validation.excluded`, by language `validation.excluded_by_language`).
Expectation: heavily lin-skewed, matching the dot-only autopsy — these are the
source-side 1.0 s stub clips.

## Impact on known failure IDs

The dot-only/very-short validation IDs from
`outputs/analysis/bad_prediction_autopsy_validation.csv` should appear almost
entirely in the excluded bucket. Verify with:

```bash
uv run --no-sync python -c "
import csv
bad = {r['ID'] for r in csv.DictReader(open('outputs/analysis/bad_prediction_autopsy_validation.csv'))}
exc = {r['id'] for r in csv.DictReader(open('data/final_combined_clean_audio_dataset/excluded_metadata.csv'))}
print(f'{len(bad & exc)}/{len(bad)} known-bad validation IDs land in excluded')
"
```

## Expected training effect

Removing audio/transcript-mismatch rows deletes the gradient signal that taught
the model 'long transcript can equal near-silent audio' — the direct cause of
dot-only collapse. Watch for: fewer dot-only validation outputs, lower Lingala
CER, and no regression on lug/sna from the (small) loss of training mass.
