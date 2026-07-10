# Proposed README for KasuleTrevor/Lingala_100hrs (PR draft)

Everything below the line is the proposed `README.md` for the dataset repo.
Counts/hours/schema were computed from a full-pass audit of the dataset on
2026-07-09 (`outputs/lingala_external/lingala_100hrs_full_audit.json`).
Remaining `TODO(owner)` items are facts only the compiler can confirm.

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

110.7 hours (23,539 rows) of Lingala speech with transcriptions, aggregated
from three publicly available CC-BY-4.0 corpora for ASR research.

## Composition

Counts from a full-pass audit on 2026-07-09:

| Source | Upstream location | Rows | Splits |
|---|---|---:|---|
| AfriVoice (Lingala) | https://huggingface.co/datasets/DigitalUmuganda/AfriVoice | 17,544 | train (16,144), validation (915), test (485) |
| LRSC (Lingala Read Speech Corpus) | https://data.mendeley.com/datasets/28x8tc9n9k/1 | 2,937 | train (2,489), validation (65), test (383) |
| FLEURS (`google/fleurs`, config `ln_cd`) | https://huggingface.co/datasets/google/fleurs | 2,995 | train (2,526), validation (65), test (404) |
| `lingala_tts` | TODO(owner): provenance and license unknown — see Known issues | 63 | test only |

## Splits

| Split | Rows | Hours | Mean duration | Mean words |
|---|---:|---:|---:|---:|
| train | 21,159 | 100.0 | 17.0 s | 24.1 |
| validation | 1,045 | 5.1 | 17.6 s | 24.9 |
| test | 1,335 | 5.6 | 15.1 s | 19.9 |

## Licensing

The three documented sources are all CC-BY-4.0, so this aggregation is
distributed under CC-BY-4.0 with the attributions below. The 63 `lingala_tts`
rows are excluded from this license claim until their provenance is
documented (see Known issues).

| Source | License | Verified against upstream on |
|---|---|---|
| AfriVoice (Lingala) | CC-BY-4.0 (upstream repo is gated: users must accept its access terms) | 2026-07-09 |
| LRSC | CC BY 4.0 (Mendeley Data, DOI 10.17632/28x8tc9n9k.1) | 2026-07-09 |
| FLEURS | CC-BY-4.0 | 2026-07-09 |
| `lingala_tts` | TODO(owner): unknown | — |

## Processing

Observed properties (from the audit): all audio is stored mono at 16 kHz;
columns are `audio`, `source`, `text`; no empty transcripts; no unusual
characters.

<!-- TODO(owner): describe what you did when building this repo: -->

- Original sample rates and resampling method: TODO(owner)
- Transcript normalization applied: TODO(owner)
- Split construction (train/validation/test): TODO(owner)
- Deduplication across sources: none needed per audit (0 texts shared across
  sources); within-source duplicate texts exist (1,690 rows repeat another
  row's text, mostly in train) — TODO(owner): confirm whether these are
  intentional (e.g. same prompt read by multiple speakers)

## Known issues

- **Undocumented source:** the test split contains 63 rows with
  `source=lingala_tts`, which is not one of the three documented corpora.
  TODO(owner): document its provenance and license, or remove these rows.
  Until then, users should exclude them.
- **Textual overlap with WaxalNLP:** a full-pass audit (2026-07-09) found
  3,090 rows (~13%) whose normalized transcript exactly matches a transcript
  in `google/WaxalNLP` (Lingala). This is expected where sources share prompt
  sets, but users training models that will be evaluated on WaxalNLP-derived
  benchmarks must deduplicate against their evaluation data before training.
  TODO(owner): if you know which source causes this overlap, please document it.
- 1,690 within-source duplicate transcripts (see Processing).

## Dataset structure

- `audio`: audio (mono, 16 kHz) + sampling rate
- `text`: Lingala transcript
- `source`: upstream corpus for the row (`Afrivoice`, `LRSC`, `fleurs`, `lingala_tts`)

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
