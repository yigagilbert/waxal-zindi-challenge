# External Data Expansion Audit

Date: 2026-07-13

Status: **NO-GO for merge or training**. This audit was completed before changing
the ingestion or cleaning pipeline. No external dataset has been merged into the
generalization mix.

## Decision summary

Two independent gates apply:

1. `docs/RULES_AND_DATA_USE.md` still classifies every external ASR dataset as
   research-only until Zindi resolves the contradiction between the Info and Data
   tabs. An open data license does not override that competition-rule gate.
2. Each source must independently have usable terms, public reproducibility, and
   a compatible/reproducible format.

| Source | License finding | Competition/reproducibility finding | Decision |
|---|---|---|---|
| `DigitalUmuganda/Afrivoice` Shona + Lingala | Dataset card: CC BY 4.0 | Public but click-through gated; 168.60 GiB compressed audio for the two languages | License-clear, competition-blocked, local full run storage-blocked |
| `google/fleurs` `ln_cd`, `lg_ug`, `sn_zw` | CC BY 4.0 | Public; external-data competition ruling still missing | Research-only until ruling |
| `Sunbird/salt` `multispeaker-lug` | CC BY-SA 4.0 | Public/click-through gated; external-data competition ruling still missing | Research-only until ruling |
| Private Luganda `lug_commonvoice` parquet | Common Voice audio/transcripts: CC0 1.0 | Repo owner matches authenticated user `yigagilbert`, but a private repo is not reproducible by an unaffiliated top-10 reviewer | License-clear; make public/pin commit before use |
| Private Luganda `lug_makerereradio` parquet | **CC BY-NC-ND 4.0**, not CC BY-SA 4.0 | Dataset card misidentifies this as Yogera; IDs and size identify Makerere Radio | **Exclude** from competition mix and cleaned redistribution |

## Mixed-license Luganda repository

Repository inspected at commit:

```text
yigagilbert/luganda-speech-cv-yogera-filtered
538ceee89ca2ad4d52c47db67eede021edad85ea
```

The authenticated Hugging Face identity is `yigagilbert`, and the repository owner
is also `yigagilbert`. It is therefore ours, not a third-party private repository
for which we only have read access.

The repository is nevertheless private. Ownership removes the third-party-access
risk but not the code-review reproducibility risk. Before use, publish an immutable
revision (or include the exact files and hashes in the review artifact), pin that
revision in code, and verify that every reviewer can fetch it without account-specific
authorization.

### Common Voice subset

Actual file:

```text
lug_commonvoice/train-00000-of-00001.parquet
```

- 4,071 packed rows
- 30.25261 hours from the supplied duration column
- columns: `id`, `audio`, `text`, `duration`, `language`
- no empty transcripts and no exact normalized transcript duplicates in this package
- observed mono Ogg/Vorbis at 16 kHz and mono Ogg/Opus reported at 48 kHz
- data license: CC0 1.0 (the Common Voice platform code is MPL 2.0, but the released
  speech dataset is CC0)

This subset is license-clear but needs an adapter (`id` -> `ID`, `text` ->
`transcription`), 16 kHz resampling, a pinned public revision, and provenance that
identifies the upstream Common Voice release/version.

### Makerere subset: card/provenance error

Actual file:

```text
lug_makerereradio/train-00000-of-00001.parquet
```

- 2,545 packed rows
- 18.55142 hours from the supplied duration column
- IDs begin with `makerereradio_lug_`
- columns: `id`, `audio`, `text`, `duration`, `language`
- observed mono Ogg/Vorbis at 16 kHz and mono Ogg/Opus reported at 48 kHz

The repository card calls this Yogera and assigns CC BY-SA 4.0. The actual package
identifies Makerere Radio, whose authoritative Zenodo record assigns CC BY-NC-ND
4.0. That license is unsuitable for this prize-competition mix without explicit
permission, and NoDerivatives is incompatible with publishing trimmed/normalized
audio as a cleaned derivative. This subset is not clear to use and must be excluded.

Authoritative record:
https://zenodo.org/records/5855017

## Format and source audit

### Afrivoice

The repository is a raw file dump, not a loadable Hugging Face DatasetDict:

- newline-delimited JSON manifests
- `audio_<n>.tar.xz` archives containing WAV files
- no upstream train/dev/test split
- manifest field `transcription` is often null/blank
- manifest paths must be resolved to `shard_id`

Manifest-only totals:

| Language | All rows/hours | Labeled rows/hours | Empty/unlabeled rows | Normalized duplicate transcript rows | Labeled shard coverage | Compressed audio |
|---|---:|---:|---:|---:|---:|---:|
| Shona | 97,240 / 571.79 h | 16,628 / 97.96 h | 80,612 | 3 | 88/88 | 92.96 GiB |
| Lingala | 95,508 / 515.06 h | 19,035 / 101.12 h | 76,473 | 7 | 84/84 | 75.64 GiB |

Every audio shard is needed to reach all labeled rows. The source card says WAV but
does not establish a sample rate. A shard must be decoded and audited before assuming
16 kHz. Required adapter steps are: extract one shard at a time, join manifest records
by `shard_id` + `audio_filepath`, drop blank transcripts, inspect/resample to mono
16 kHz, and create a deterministic speaker-grouped split. The current machine has
about 22 GiB free, so the full source cannot be staged by the existing pipeline.

### FLEURS

- CC BY 4.0
- native schema is already recognized: `id`, `audio`, `transcription`,
  `raw_transcription`, `num_samples`, etc.
- audio feature is 16 kHz
- upstream splits: train/validation/test (raw files call validation `dev`)

Manifest hours:

| Config | Train | Dev/validation | Test | Total |
|---|---:|---:|---:|---:|
| `ln_cd` | 18.2369 h | 1.0654 h | 2.5839 h | 21.8863 h |
| `lg_ug` | 12.6350 h | 1.3931 h | 3.4027 h | 17.4308 h |
| `sn_zw` | 9.9674 h | 1.5456 h | 3.8008 h | 15.3138 h |

FLEURS intentionally has multiple speakers reading the same sentence, so repeated
transcript text alone must not be treated as a duplicate. Deduplicate exact audio
hashes and WAXAL train/validation transcript overlaps instead. Use only the upstream
train split in the training mix; do not silently consume its validation/test splits.

### SALT `multispeaker-lug`

- CC BY-SA 4.0
- columns: `id`, `text`, `audio`, `audio_language`, `is_studio`, `speaker_id`
- mono Ogg/Opus, reported at 48 kHz; resample to 16 kHz
- upstream splits are train/dev/test, not train/validation/test
- train: 5,002 rows, 7.8986 h
- dev: 103 rows, 0.1600 h
- test: 99 rows, 0.1622 h

Use `multispeaker-lug` only for ASR. The current default also requests `studio-lug`,
which was not requested and is TTS-oriented. Use only upstream train for the training
mix and preserve dev/test.

## Current pipeline incompatibilities

The requested run cannot be represented faithfully by the current code:

1. `scripts/clean_and_trim_audio_dataset.py` computes hashes but does not perform
   deduplication.
2. `scripts/prepare_generalization_mix.py` folds FLEURS train/validation/test into
   training by default.
3. SALT `dev` is silently missed because the loader asks for `validation`.
4. SALT defaults to both `multispeaker-lug` and `studio-lug`.
5. There is no Afrivoice raw-shard adapter.
6. There is no adapter for the private Luganda parquet schemas.
7. The required local `data/processed/hf_dataset` and clean WAXAL manifests are
   absent, so an actual WAXAL-anchored merge cannot be built from this checkout.
8. The local disk cannot stage Afrivoice with the existing all-at-once design.

Consequently, no honest post-cleaning retained-hour/drop-reason report or merged
per-language total can be produced yet. Pre-cleaning source hours are not substitutes
for the requested cleaning report.

## Required go-live sequence

1. Obtain and archive Zindi's written ruling allowing external training data.
2. Exclude Makerere Radio unless its owners grant explicit competition and derivative
   permission.
3. Publish/pin the Common Voice package and document its upstream version/hashes.
4. Restore/build the WAXAL `hf_dataset` and clean manifests.
5. Add streaming Afrivoice ingestion, train-only external ingestion, SALT split
   mapping, private-parquet adapters, and real audio/text-overlap deduplication.
6. Run the cleaner to a volume with sufficient space and emit per-source as well as
   per-language drop reasons/hours.
7. Balance with a language-aware sampler (one-third Lingala, Luganda, Shona per epoch)
   and source-aware sampling inside Luganda, rather than deleting useful speech to
   force identical physical hours.
8. Validate the 300M model on untouched WAXAL validation before starting XLS-R 1B.

## Training-level placement

This work is a **Level 1 prerequisite (data legality, provenance, and quality)**.
The balanced 300M generalization experiment is Level 2. XLS-R 1B is Level 3 and
should wait until the Level 1 gates are closed and Level 2 demonstrates a gain. The
current audit therefore strengthens, rather than changes, the recommendation to
delay the 1B run.
