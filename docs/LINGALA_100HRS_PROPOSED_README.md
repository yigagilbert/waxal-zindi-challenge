# Proposed README for KasuleTrevor/Lingala_100hrs (PR draft)

Everything below the line is the proposed `README.md` for the dataset repo.
Remaining `TODO(owner)` items are facts only the compiler can confirm; the
`TODO(auto)` counts are filled from `outputs/lingala_external/lingala_100hrs_full_source_stats.csv`
after running the full audit (command at the bottom of this file).

---

---
license: cc-by-4.0
language:
  - ln
task_categories:
  - automatic-speech-recognition
pretty_name: Lingala 100hrs ASR
---

# Lingala 100hrs

Approximately 100 hours of Lingala speech with transcriptions, aggregated from
three publicly available CC-BY-4.0 corpora for ASR research.

## Composition

| Source | Upstream location | Rows | Hours | Splits used |
|---|---|---:|---:|---|
| AfriVoice (Lingala) | https://huggingface.co/datasets/DigitalUmuganda/AfriVoice | TODO(auto) | TODO(auto) | TODO(auto) |
| LRSC (Lingala Read Speech Corpus) | https://data.mendeley.com/datasets/28x8tc9n9k/1 | TODO(auto) | TODO(auto) | TODO(auto) |
| FLEURS (`google/fleurs`, config `ln_cd`) | https://huggingface.co/datasets/google/fleurs | TODO(auto) | TODO(auto) | TODO(auto) |

<!-- TODO(owner): confirm the auto-computed source mapping matches how you
     actually built the dataset (the per-row source field is the ground truth). -->

## Licensing

All upstream sources are CC-BY-4.0, so this aggregation is distributed under
CC-BY-4.0 with the attributions below.

| Source | License | Verified against upstream on |
|---|---|---|
| AfriVoice (Lingala) | CC-BY-4.0 (upstream repo is gated: users must accept its access terms) | 2026-07-09 |
| LRSC | CC BY 4.0 (Mendeley Data, DOI 10.17632/28x8tc9n9k.1) | 2026-07-09 |
| FLEURS | CC-BY-4.0 | 2026-07-09 |

Note: the AfriVoice upstream requires accepting access terms on Hugging Face.
Users of this aggregation should be aware the underlying data was obtained
under those terms.

## Processing

<!-- TODO(owner): describe what was done when building this repo: -->

- Audio resampled to: TODO(owner)
- Transcript normalization: TODO(owner)
- Split construction (train/validation/test): TODO(owner)
- Deduplication across sources: TODO(owner)

## Known issues

- An independent sampled audit (2026-07) found possible textual overlap between
  rows of this dataset and transcripts in `google/WaxalNLP` (Lingala). Users
  training models that will be evaluated on WaxalNLP-derived benchmarks should
  deduplicate against their evaluation sets before training.
  <!-- TODO(owner): if you know which upstream source causes this overlap
       (e.g. shared provenance with WaxalNLP), please document it. -->
- TODO(owner): any known transcription quality issues.

## Dataset structure

<!-- TODO(owner): adjust to actual features/column names. -->

- `audio`: audio + sampling rate
- `text` / `transcription`: Lingala transcript
- `source`: upstream corpus for this row

## Citations

LRSC:

> Kimanuka, U., wa Maina, C., & Büyük, O. (2023). Speech Recognition Datasets
> for Low-resource Congolese Languages. AfricaNLP workshop at ICLR 2023.
> Data: Mendeley Data, V1, doi:10.17632/28x8tc9n9k.1

FLEURS:

> Conneau, A., et al. (2022). FLEURS: Few-shot Learning Evaluation of
> Universal Representations of Speech. IEEE SLT 2022.

AfriVoice:

> Digital Umuganda. AfriVoice: multilingual African speech dataset
> (Shona, Lingala, Fulani, Malagasy, Wolof, Somali).
> https://huggingface.co/datasets/DigitalUmuganda/AfriVoice

---

## Filling in the TODO(auto) counts (run on the GPU box)

```bash
# full unsampled audit: exact rows/hours per source per split, plus the FULL
# text-overlap count vs WAXAL train and ID-overlap vs Phase 1 test
WAXAL_NO_SYNC=1 uv run --no-sync scripts/audit_lingala_100hrs.py \
  --output outputs/lingala_external/lingala_100hrs_full_audit.json \
  --source-stats-output outputs/lingala_external/lingala_100hrs_full_source_stats.csv

# print everything needed for the card, paste-ready
uv run --no-sync python -c "
import csv, json
from collections import defaultdict
agg = defaultdict(lambda: {'rows': 0, 'seconds': 0.0, 'splits': set()})
for r in csv.DictReader(open('outputs/lingala_external/lingala_100hrs_full_source_stats.csv')):
    a = agg[r['source']]
    rows = int(r['rows']); a['rows'] += rows; a['splits'].add(r['split'])
    if r['mean_duration']: a['seconds'] += float(r['mean_duration']) * rows
print('== Composition table rows ==')
for src, a in sorted(agg.items()):
    print(f\"| {src} | ... | {a['rows']} | {a['seconds']/3600:.1f} | {', '.join(sorted(a['splits']))} |\")
audit = json.load(open('outputs/lingala_external/lingala_100hrs_full_audit.json'))
print('== Observed properties (for the Processing/Structure sections) ==')
for split, rep in audit['splits'].items():
    print(f\"{split}: rows={rep['rows_seen']} hours={rep.get('total_hours')} sampling_rates={rep.get('observed_sampling_rates')} columns={rep.get('observed_columns')}\")
print('== Leakage / dedup gate (for OUR decision, not the card) ==')
print('text overlap with WAXAL lin train:', audit['overall']['possible_text_overlap_with_waxal_lingala_train'])
print('ID overlap with Phase 1 test:', audit['overall']['possible_id_overlap_with_waxal_test'])
print('duplicate texts overall:', audit['overall']['duplicate_text_count'])
print('texts shared across sources:', audit['overall'].get('texts_shared_across_sources'))
"
```
