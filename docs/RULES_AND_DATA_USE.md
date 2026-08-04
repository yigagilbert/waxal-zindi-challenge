# Rules and Data Use

This project is being run as a competition solution for the Zindi Google WAXAL ASR Challenge. The operating rule for Phase 1 is conservative: use only the challenge-specified WAXAL data for training, and use external datasets only as research references unless Zindi explicitly confirms otherwise.

Source checked: https://zindi.africa/competitions/google-waxal-asr-challenge on 2026-07-03.

## Phase-2 endgame freeze — live pages rechecked 2026-07-30

Both live challenge pages were read again before the final H100 runs:

- Info: https://zindi.world/competitions/google-waxal-asr-challenge
- Data: https://zindi.world/competitions/google-waxal-asr-challenge/data

They still conflict. The Info page says only challenge-specified datasets may be
used. The challenge-specific Data page says public open-source speech or language
datasets may supplement the challenge data if they are publicly accessible,
legally licensed, and disclosed.

The remaining experiments use the strict intersection of those statements:

- Direct training, language-ID, and language-model data: only
  `google/WaxalNLP` train/validation and the Zindi-supplied challenge files.
- Phase-2 audio: inference and signal/transcript-derived routing only. It is
  never treated as labeled or pseudo-labeled ASR training data.
- Pretrained ASR: only checkpoints openly available to everyone, as explicitly
  permitted by the Info page. The current base is
  `Sunbird/asr-whisper-large-v3-salt` at revision
  `7448016c50bdec469b8454c9631c76fc1d1dd40e` (MIT; public, gated `auto`).
  The transcript-only LID view uses
  `huwenjie333/whisper-v3-ft-af51` at revision
  `3648a5b3bec96a9d72c5f96f7d5aa94add2a4a1f` (MIT; public and ungated).
- Packages: publicly available open-source packages only. No AutoML, paid API,
  private service, or credit-card-gated tool.
- Seeds, exact model revisions, commands, routing outputs, and final artifacts
  must be retained for code review.

Explicitly excluded from remaining runs even though the Data page appears to
permit them: Wikipedia LM text, Common Voice, FLEURS, SALT training data,
Makerere Radio, BibleTTS, private corpora, and any other external dataset.

The live page gives a close/reveal date of 2026-08-02, a limit of five
submissions per day, and requires two submissions to be selected. It
inconsistently describes the public/private split as both 30/70 and 20/80, so
the public leaderboard is treated as a weak selection sample.

## UNRESOLVED RULE CONFLICT (found 2026-07-09) — external data

The Info/Rules tab and the Data tab directly contradict each other on external data:

- **Info tab:** "You may use only the datasets specified for this challenge."
- **Data tab, Phase 1 About:** "Participants may supplement the provided challenge
  data with other publicly available open-source speech or language datasets. Any
  external datasets used must be publicly accessible, legally licensed for research
  or development, and disclosed in the final solution documentation."

This matters directly: our submitted champion model uses FLEURS (CC-BY-4.0) and
Sunbird/SALT (CC-BY-SA-4.0), which are external. Under the Info-tab reading that is
non-compliant; under the Data-tab reading it is allowed with disclosure (which we do).

ACTION REQUIRED before final code review: post in the Zindi discussion asking for an
official ruling quoting both lines, and save the response (screenshot + paste here).
Until then, keep full disclosure of every external source and license in this file so
the solution is defensible under the permissive reading. The Data-tab language is
Phase-specific and more recent, so it is the more likely authoritative statement, but
do not rely on that without written confirmation.

## Exact Rule Quotes

External datasets (Info tab — conflicts with Data tab above):

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
| Wikipedia Lingala (`wikimedia/wikipedia`, `20231101.ln`) | Used under Data-tab permissive reading, disclosed | CC-BY-SA-4.0 public text; TEXT-ONLY for the Lingala KenLM decode corpus (no audio, no labels) | Lingala LM corpus v2 (`data/lm_expanded_v2*`), added 2026-07-22 |

## Model Decisions

| Model | License posture | Current Use |
|---|---|---|
| `facebook/wav2vec2-xls-r-300m` | Open pretrained model, Apache-2.0 | Primary XLS-R 300M fine-tuning base on official WAXAL data |
| `openai/whisper-large-v3-turbo` | Open pretrained model | General teacher/baseline on official WAXAL data |
| `openai/whisper-large-v3` | Open pretrained model | Stronger teacher/baseline if compute allows |
| `Sunbird/asr-whisper-large-v3-salt` | Openly available pretrained model (gated public HF repo; terms accepted 2026-07-28) | Phase 1: Luganda teacher/baseline. **From 2026-07-28: candidate Phase-2 engine** — supports ach/nyn/xog/myx via SALT language tokens (repurposed Whisper slots); used inference-only with per-clip forced language codes from our transcript-LID clustering. |
| `facebook/mms-1b-all` | Non-commercial license risk | Diagnostic only; do not use for final training unless approved |
| `Sunbird/asr-mms-salt` | Non-commercial license risk | Diagnostic only |
| `facebook/seamless-m4t-v2-large` | Non-commercial license risk | Diagnostic only, later if needed |
| `huwenjie333/whisper-v3-ft-af51` | Openly available pretrained model (public HF repo) — "You may use pretrained models as long as they are openly available to everyone" | **USED for Phase-2 submissions from 2026-07-27** (`phase2_af51*.csv`): Phase 2 is an unseen-language generalization test (Acholi/Lango, Runyankole-Rukiga, Lusoga, Lumasaba, …) outside lin/lug/sna; af51's 51-African-language coverage makes it the appropriate engine there. (Phase-1 evaluation: WAXAL-lin 0.3397 vs champion 0.1683 — not used for Phase 1.) Inference-only; no fine-tuning performed on it. |
| `Sunbird/asr-whisper-51-african-languages` | Openly available pretrained model (gated public HF repo — anyone can accept terms; ours accepted 2026-07-28) | Candidate Phase-2 engine A/B vs af51; reportedly strongest when given the per-clip language code. Inference-only unless a fine-tune is logged here. |

Phase-2 fine-tuning (wired 2026-07-28, config `configs/whisper_salt_phase2_lora.yaml`):
LoRA adaptation of `Sunbird/asr-whisper-large-v3-salt` on `google/WaxalNLP` official train
splits — BOTH the challenge trio (lin/lug/sna, keeping the expected training languages) AND
the Phase-2 languages (ach/nyn/xog/myx). All data is the official challenge dataset family;
the engine is openly available; Phase-2 *test* clips are never used for training. Log the
resulting adapter checkpoint here if a submission uses it.

## Phase 1 Decision

Phase 1 does not train on external datasets and does not relabel WAXAL examples. It only:

- audits official WAXAL audio quality,
- runs openly available teacher models on official WAXAL audio,
- scores teacher disagreement against original WAXAL labels,
- creates clean/medium/noisy/excluded buckets using original labels,
- prepares a clean official-WAXAL subset for XLS-R 300M CTC fine-tuning.

If external data becomes strategically necessary, ask Zindi in the challenge discussion and save the response before using it.

## Semi-supervised use of the official unlabeled split (2026-08-03)

The corrected Phase-2 campaign trains a continuation of the champion on
pseudo-labeled audio from `google/WaxalNLP` **unlabeled** lin/sna splits — part
of the challenge-specified dataset, not external data. Labels are generated
exclusively by our own champion model (greedy/beam-agreement filtered); no
external model or service is involved. Phase-2 test audio is never used for
training or pseudo-labeling. Artifacts: `data/unlabeled_linsna/`,
`outputs/day4_h100/pseudo_labels_raw.csv`, config to follow as
`configs/xlsr_champion_ssl_continue.yaml`.
