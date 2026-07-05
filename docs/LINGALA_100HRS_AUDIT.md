# Lingala 100hrs Audit

Dataset under review:

```text
KasuleTrevor/Lingala_100hrs
```

URL:

```text
https://huggingface.co/datasets/KasuleTrevor/Lingala_100hrs
```

## Current Public Card Snapshot

The Hugging Face page shows:

- modalities: audio and text,
- format: parquet,
- approximate size class: 10K-100K rows,
- default config with about 23.5K rows,
- splits:
  - train: about 21.2K rows,
  - validation: about 1.05K rows,
  - test: about 1.34K rows,
- columns visible in viewer:
  - `audio`,
  - `text`,
  - `source`,
- source examples include `Afrivoice`.

The page visible from this environment does not provide enough license/source-level documentation to approve training use immediately.

## Required Audit

Run:

```bash
export WAXAL_NO_SYNC=1
WAXAL_NO_SYNC=1 make audit-lingala-100hrs
```

Fast audit defaults to streaming and samples 500 rows per split:

```text
outputs/lingala_external/lingala_100hrs_audit.json
outputs/lingala_external/lingala_100hrs_source_stats.csv
```

For a full audit, run without the row cap:

```bash
uv run --no-sync scripts/audit_lingala_100hrs.py \
  --metadata data/processed/train.csv \
  --test-ids data/processed/test.csv \
  --output outputs/lingala_external/lingala_100hrs_audit_full.json \
  --source-stats-output outputs/lingala_external/lingala_100hrs_source_stats_full.csv
```

## What The Audit Checks

- row count by split,
- source distribution,
- audio duration distribution,
- transcript length distribution,
- empty transcripts,
- duplicate transcripts,
- unusual characters,
- possible text overlap with WAXAL Lingala train,
- possible ID overlap with WAXAL test,
- dataset-info license field,
- missing source/license metadata.

## Decision

Do not train on `KasuleTrevor/Lingala_100hrs` until:

1. dataset-level license is confirmed,
2. source-level licenses are confirmed for FLEURS, AMMI, AFRIVOICE, and LRSC or any other source present,
3. no Phase 1 test-label leakage risk is found,
4. its validation/test split is not accidentally derived from WAXAL Phase 1 hidden labels,
5. source domains are understood well enough to set a safe sampling ratio.

## Likely Use If Approved

Preferred order:

1. Use as a Lingala warm-start dataset.
2. Fine-tune back on official WAXAL Lingala.
3. Only then consider low-ratio multilingual mixing.

Avoid direct high-ratio mixing until we know whether the dataset is image-caption style, Bible/read speech, conversational, or WAXAL-like.
