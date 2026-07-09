# Clean Audio Dataset — Reproducibility

Preprocessing version: `clean_audio_v1`
Git commit: recorded automatically in `outputs/data_quality/cleaning_impact_report.json`

## Inputs (never modified)

| Data | Location on GPU box |
|---|---|
| Raw Zindi files | `$WAXAL_RAW_DIR` (Train.csv / Test.csv / SampleSubmission.csv) |
| Prepared official WAXAL | `data/processed/hf_dataset` |
| Generalization mix (WAXAL clean + FLEURS + SALT) | `data/processed_generalization_mix/hf_dataset` |

## Outputs

| Artifact | Location |
|---|---|
| Cleaned FLAC audio | `data/audio_cleaned/<split>/<language>/<ID>.flac` |
| Final DatasetDict (audio embedded on push) | `data/final_combined_clean_audio_dataset/hf_dataset` |
| Metadata CSVs | `data/final_combined_clean_audio_dataset/*_metadata.csv` |
| Quality buckets | `data/quality/final_{clean,medium,noisy,excluded}_train.csv`, `final_{clean,excluded}_validation.csv`, `{lin,lug,sna}_clean_train.csv` |
| Audit | `outputs/data_quality/full_audio_text_audit.{csv,summary.json}` |
| Cleaning reports | `outputs/data_quality/audio_cleaning_report.csv`, `cleaning_impact_report.json` |
| Private HF repo | `<HF_USERNAME>/waxal-combined-clean-audio-asr-private` |

## Exact commands (in order)

```bash
export WAXAL_NO_SYNC=1   # torch was manually repaired for Blackwell; never plain `uv sync`

# 0. smoke everything on a small sample first
uv run --no-sync scripts/clean_and_trim_audio_dataset.py --max-samples-per-split 200 \
  --output-audio-dir data/audio_cleaned_smoke --output-dataset data/final_clean_smoke \
  --quality-dir data/quality_smoke --reports-dir outputs/data_quality_smoke

# 1. full audit (read-only, CPU)
WAXAL_NO_SYNC=1 make audit-audio-text

# 2. clean + trim + bucket + build DatasetDict (CPU, hours; safe alongside GPU training)
WAXAL_NO_SYNC=1 make clean-trim-audio

# 3. push to private HF repo (needs HF write token in env)
WAXAL_NO_SYNC=1 make push-clean-dataset HF_CLEAN_REPO=<HF_USERNAME>/waxal-combined-clean-audio-asr-private
```

## Method + thresholds

All thresholds live as CLI defaults in `scripts/clean_and_trim_audio_dataset.py`
and are echoed into `cleaning_impact_report.json` on every run. Trimming:
`waxal.audio.trim_edges` (top_db=40, pad_ms=200, 30 ms frames, absolute RMS
floor 1e-4, min post-trim duration 0.3 s). Bucket rules:
`docs/DATA_QUALITY_AUDIT.md`.

## Package versions

`uv.lock` pins everything except torch, which is manually repaired to
`torch 2.11.0+cu128` on Blackwell GPUs (`docs/RESTART_RUNBOOK.md`). Record
`uv run --no-sync python -c "import torch, datasets, soundfile, transformers as t; print(torch.__version__, datasets.__version__, t.__version__)"`
alongside any regenerated dataset.

## Reloading for training

Local: configs point at `data/final_combined_clean_audio_dataset` —
`configs/xlsr_{300m,1b}_clean_audio_{only,plus_medium}_v1.yaml`.
From the Hub (fresh box):

```python
from datasets import load_dataset
ds = load_dataset("<HF_USERNAME>/waxal-combined-clean-audio-asr-private")
ds.save_to_disk("data/final_combined_clean_audio_dataset/hf_dataset")
```

## Row counts / hours

Do not hand-edit numbers here: read them from
`outputs/data_quality/cleaning_impact_report.json` (bucket counts per language
and source, hours before/after, silence removed, exclusion reasons).

## Known limitations

- Source-side 1.0 s stub clips (WaxalNLP defect) are excluded, not repaired.
- Validation split stays FULL (all buckets) for comparable evaluation; the
  excluded-validation list is analysis-only.
- If models are trained on trimmed audio, apply `waxal.audio.trim_edges` with
  identical parameters at inference time.
