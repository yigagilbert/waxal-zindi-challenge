# Rules and Data Use

This project is being run as a competition solution for the Zindi Google WAXAL ASR Challenge. The operating rule for Phase 1 is conservative: use only the challenge-specified WAXAL data for training, and use external datasets only as research references unless Zindi explicitly confirms otherwise.

Source checked: https://zindi.africa/competitions/google-waxal-asr-challenge on 2026-07-03.

## Exact Rule Quotes

External datasets:

> "You may use only the datasets specified for this challenge."

Pretrained models:

> "You may use pretrained models as long as they are openly available to everyone."

The same rules section also requires publicly available open-source packages and disallows AutoML tools. The top-10 code review requirements make reproducibility important: every dataset, model, checkpoint, and preprocessing decision must be logged.

## Working Interpretation

Allowed:

- Official Zindi challenge files: `Train.csv`, `Test.csv`, `SampleSubmission.csv`, challenge starter resources.
- Challenge-specified WAXAL audio examples corresponding to the official Zindi IDs and splits.
- Openly available pretrained models, as long as their licenses are compatible with competition use and code review.
- Teacher-model diagnostics on official WAXAL train/validation audio, provided labels are not replaced automatically and all outputs are logged.

Not allowed in this repo until explicitly approved:

- Training directly on FLEURS, Common Voice, SALT, BibleTTS, Makerere Radio Luganda, OpenSLR, or other public speech datasets.
- Using Phase 1 public-test labels or any labels for official test IDs.
- Private datasets, private model weights, paid-only APIs, AutoML tools, or anything unavailable to other competitors.

Unclear and treated as unsafe for direct training:

- Public external ASR datasets that are open licensed but not specified on the challenge page.
- Non-commercial model checkpoints in a prize competition setting.
- Teacher-generated replacement labels for WAXAL train data. Phase 1 only flags suspicious examples.

## Dataset Decisions

| Resource | Status | Reason | Current Use |
|---|---|---|---|
| Official Zindi WAXAL train/validation/test IDs | Safe | Challenge-specified data | Training, validation, submission |
| `google/WaxalNLP` audio for matching Zindi IDs | Safe when restricted to official IDs/splits | This is the WAXAL source used to fetch challenge audio | Audio loading only; do not use hidden test labels |
| FLEURS | Research-only | External dataset not explicitly specified for this challenge | Model/data research only |
| Mozilla Common Voice | Research-only | External dataset not explicitly specified for this challenge | Model/data research only |
| Sunbird/SALT dataset | Research-only | External dataset not explicitly specified for this challenge | Use model card/context only; no direct training |
| BibleTTS | Research-only | External dataset not explicitly specified for this challenge | Model/data research only |
| Makerere Radio Luganda | Research-only | External dataset and license/code-review suitability not confirmed | Research only |
| Any Hugging Face ASR dataset for lug/lin/sna | Research-only | External unless Zindi confirms | Research only |

## Model Decisions

| Model | License posture | Current Use |
|---|---|---|
| `facebook/wav2vec2-xls-r-300m` | Open pretrained model, Apache-2.0 | Primary XLS-R 300M fine-tuning base on official WAXAL data |
| `openai/whisper-large-v3-turbo` | Open pretrained model | General teacher/baseline on official WAXAL data |
| `openai/whisper-large-v3` | Open pretrained model | Stronger teacher/baseline if compute allows |
| `Sunbird/asr-whisper-large-v3-salt` | Openly available pretrained model | Luganda teacher/baseline on official WAXAL data |
| `facebook/mms-1b-all` | Non-commercial license risk | Diagnostic only; do not use for final training unless approved |
| `Sunbird/asr-mms-salt` | Non-commercial license risk | Diagnostic only |
| `facebook/seamless-m4t-v2-large` | Non-commercial license risk | Diagnostic only, later if needed |

## Phase 1 Decision

Phase 1 does not train on external datasets and does not relabel WAXAL examples. It only:

- audits official WAXAL audio quality,
- runs openly available teacher models on official WAXAL audio,
- scores teacher disagreement against original WAXAL labels,
- creates clean/medium/noisy/excluded buckets using original labels,
- prepares a clean official-WAXAL subset for XLS-R 300M CTC fine-tuning.

If external data becomes strategically necessary, ask Zindi in the challenge discussion and save the response before using it.
