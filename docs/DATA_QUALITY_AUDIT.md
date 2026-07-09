# Data Quality Audit

Date started: 2026-07-09
Preprocessing version: `clean_audio_v1`

## Why

The dot-only autopsy (2026-07-08) found that clips whose predictions collapse to
`.` are dominated by audio that is exactly 1.0 s long while carrying full-sentence
transcripts (validation flagged set: median 20 reference words, max 43). Our prep
([scripts/prepare_dataset.py](../scripts/prepare_dataset.py)) is a pass-through from
`google/WaxalNLP` with only a 16 kHz resample, so the truncation is source-side.
Training on such rows teaches the model that long transcripts can map to
near-empty audio — a direct cause of dot-only/very-short outputs.

## Tools

- `scripts/audit_audio_text_consistency.py` — read-only audit over prepared datasets
- `scripts/clean_and_trim_audio_dataset.py` — trims edge silence, buckets, saves cleaned audio

## Audit metrics (per example)

id, source id/dataset, language, split, transcript, transcript chars/words,
duration, sample rate, RMS, peak, silence ratio, clipping ratio,
leading/trailing silence, chars/sec, words/sec, quality flags.

## Flag rules (defaults; all CLI-overridable)

| Flag | Rule |
|---|---|
| `empty_transcript` | normalized transcript empty |
| `punctuation_only_transcript` | transcript is only punctuation |
| `empty_audio` | duration <= 0.05 s or zero RMS |
| `likely_silent` | RMS < 0.005 or silence ratio > 0.95 |
| `short_audio_long_transcript` | duration < 1.5 s and > 8 words |
| `short_audio_many_chars` | duration < 2.0 s and > 20 chars |
| `chars_per_second_too_high` | > 35 cps (real speech is ~5–25) |
| `chars_per_second_too_low` | < 2 cps with duration > 4 s |
| `heavy_clipping` | > 5% samples at |x| > 0.99 |
| `audio_transcript_mismatch` | silent-but-long-transcript, or short-audio/high-cps combinations |

## Bucket policy (training rows)

- **excluded**: empty transcript, failed decode, likely-empty audio, too short
  after trim, short-audio-long-transcript, cps > 35, or any mismatch flag.
  Never trained on; kept in `excluded_for_review` split with full metadata.
- **noisy**: likely-silent (but transcript short), heavy clipping, cps 25–35,
  or short-audio-many-chars. Kept in `noisy_for_review`, not default training.
- **medium**: cps 22–25 or > 50% of the original duration was edge silence.
  Usable via `extra_train_splits: [medium_train]`.
- **clean**: everything else. Default training split.

Validation is bucketed for analysis but the DatasetDict keeps the FULL
validation split so evaluation stays comparable across model generations.

## Silence trimming

`waxal.audio.trim_edges`: frame RMS (30 ms frames); speech frames are those
within `top_db` (default 40 dB) of the loudest frame and above an absolute
floor (1e-4). Only leading/trailing non-speech regions are removed; internal
pauses are untouched; 200 ms padding is kept on each side. If trimming would
leave < 0.3 s, the original audio is kept and the row is flagged
`too_short_after_trim`. Rows with no speech frame at all are flagged
`likely_empty_audio` and left untrimmed.

## Commands

```bash
# audit (read-only)
WAXAL_NO_SYNC=1 make audit-audio-text

# clean + trim + bucket + build final dataset
WAXAL_NO_SYNC=1 make clean-trim-audio

# push (private HF repo)
WAXAL_NO_SYNC=1 make push-clean-dataset HF_CLEAN_REPO=<user>/waxal-combined-clean-audio-asr-private
```

Fill in results from `outputs/data_quality/full_audio_text_audit_summary.json`
and `outputs/data_quality/cleaning_impact_report.json` after the run.
