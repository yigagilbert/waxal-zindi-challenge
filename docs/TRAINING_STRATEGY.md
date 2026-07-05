# Training Strategy

The current public score shows that generic Whisper LoRA is not enough for this challenge. WAXAL speech is natural, multilingual, and likely noisier than clean read-speech benchmarks. The next move is to improve data quality and model fit before spending more GPU hours.

## Why We Are Changing Direction

The previous Whisper LoRA run improved over zero-shot Whisper locally, but the public leaderboard score remained weak. The likely causes are:

- noisy or mismatched audio/transcript pairs,
- weak handling of Luganda by generic Whisper,
- language imbalance, especially lower Luganda volume,
- Whisper hallucination on low-resource speech,
- insufficient validation diagnostics before training,
- training on all examples with equal weight before identifying bad rows.

WAXAL-NET results indicate that WAXAL-domain fine-tuned compact ASR models can beat larger zero-shot systems. That supports trying a clean `facebook/wav2vec2-xls-r-300m` CTC run before repeating larger Whisper LoRA sweeps. Source: https://arxiv.org/abs/2606.02375.

## Phase 1 Scope

Phase 1 is diagnostic and competition-safe:

- use official WAXAL data only,
- do not train on external datasets,
- do not automatically relabel training rows,
- use teacher models only to flag suspicious labels,
- build quality buckets from original WAXAL labels,
- prepare for the first clean XLS-R 300M experiment.

## Sunbird Usage

`Sunbird/asr-whisper-large-v3-salt` is treated as a Luganda teacher. It should be run on Luganda validation first and optionally on Luganda train subsets later. Its predictions help identify suspicious Luganda rows, but the original WAXAL labels remain the training target in Phase 1.

Sunbird is not used as a reason to train many separate models immediately. The first serious follow-up is still one multilingual XLS-R 300M model trained on clean official WAXAL examples.

## Luganda Teacher Cleaning

After the first audio-quality audit, Luganda needs special handling because many Luganda clips are longer than 30 seconds, and validation shows that long Luganda clips are part of the real distribution. A hard duration discard is too aggressive.

The Luganda-only teacher-cleaning workflow is:

1. Run Sunbird Whisper on Luganda train audio.
2. Compare Sunbird predictions with original WAXAL Luganda labels under `no_punct_lower`.
3. Keep original labels when Sunbird agrees.
4. Replace labels only when Sunbird is plausible and disagreement is high, using `teacher-label-mode high_disagreement`.
5. Write high-uncertainty examples to a review manifest instead of silently training on them.
6. Keep Lingala and Shona from the existing clean WAXAL bucket.

This is still official-WAXAL-only training: no external audio dataset is added. The pretrained Sunbird model is used as an openly available teacher on official WAXAL train audio.

Key artifacts:

- `outputs/teachers/sunbird_whisper_lug_train.csv`
- `data/quality/lug_sunbird_clean_train.csv`
- `data/quality/lug_sunbird_review_train.csv`
- `data/quality/lug_sunbird_excluded_train.csv`
- `data/quality/clean_train_sunbird_lug.csv`
- `outputs/quality/luganda_teacher_cleaning_summary.json`

## Teacher Diagnostics

Teacher outputs are used for:

- disagreement scoring against original labels,
- detecting likely bad labels,
- detecting silence/no-speech cases,
- comparing model families per language,
- deciding whether per-language follow-up is justified.

Teacher outputs are not used for:

- replacing official labels,
- pseudo-labeling external data,
- training on official test IDs,
- final label correction without manual review.

## Experiment Order

1. Confirm data and rules.
2. Prepare official WAXAL train/validation audio cache.
3. Run `audit_audio_quality.py` on validation and train.
4. Run teacher inference on validation:
   - Sunbird Whisper for Luganda,
   - Whisper turbo for all languages,
   - Whisper large-v3 if compute allows.
5. Build quality buckets for train.
6. Run XLS-R 300M clean smoke training.
7. If smoke passes, train `xlsr_300m_balanced_clean_all`.
8. Evaluate per language and compare against:
   - current public score `0.5258`,
   - local Whisper LoRA validation baseline,
   - Whisper turbo validation baseline,
   - Sunbird Luganda validation baseline.
9. Only after the multilingual clean XLS-R result is known, decide whether to run per-language models.

## Lingala Recovery Update

After the XLS-R v2 submission, the main weakness is Lingala rather than Luganda. The public score is consistent with local validation once interpreted as:

```text
zindi_score = 1 - 0.5 * (WER + CER)
```

So the next strategy is not to restart from scratch. It is to recover Lingala.

Highest-priority Lingala model:

```text
Alvin-Nahabwe/wav2vec2-xls-r-300m-Fleurs_AMMI_AFRIVOICE_LRSC-ln-109hrs-v2
```

This model should be evaluated before any new full WAXAL training run because:

- it is an actual Lingala ASR model,
- it uses the same XLS-R 300M family as our current best model,
- it is reported as Apache-2.0,
- it is likely more useful for Lingala than generic Whisper or text-only correction.

New Lingala execution order:

1. Confirm Alvin model access with `make alvin-lingala-access`.
2. Run Alvin on WAXAL Lingala validation.
3. Compare Alvin against checkpoint-6000 on Lingala validation.
4. Run Alvin on WAXAL Lingala train as a teacher.
5. Build `data/quality/clean_train_alvin_lingala_v1.csv`.
6. Run Alvin on WAXAL Lingala test for sanity comparison.
7. Decide between direct routing, Alvin-initialized fine-tuning, or Alvin-assisted filtering.

Do not start external Lingala training until `KasuleTrevor/Lingala_100hrs` and source-level licenses are audited.

## Expected Artifacts

Audio quality:

- `outputs/quality/audio_quality_train.csv`
- `outputs/quality/audio_quality_train.summary.json`
- `outputs/quality/audio_quality_validation.csv`
- `outputs/quality/audio_quality_validation.summary.json`

Teacher predictions:

- `outputs/teachers/sunbird_whisper_lug_validation.csv`
- `outputs/teachers/whisper_turbo_validation.csv`

Quality buckets:

- `data/quality/clean_train.csv`
- `data/quality/medium_train.csv`
- `data/quality/noisy_train.csv`
- `data/quality/excluded_train.csv`
- `outputs/quality/quality_bucket_summary.json`

Training:

- `checkpoints/xlsr_300m_balanced_clean_all_smoke/`
- `checkpoints/xlsr_300m_balanced_clean_all/`
- `outputs/experiments/*.json`

## Stop/Early-Fail Criteria

Stop a run early if:

- validation loss does not improve for several evals,
- one language collapses while others improve,
- decoded validation text is mostly blanks/repeated characters,
- clean bucket size is unexpectedly tiny,
- bucket summary shows systematic audio/label corruption that needs manual inspection.

Move to full training only if:

- audio audit produces plausible duration and rate distributions,
- bucket counts retain enough examples per language,
- smoke training finishes and produces non-empty validation predictions/metrics,
- local validation trend is competitive with or better than the Whisper LoRA baseline.
