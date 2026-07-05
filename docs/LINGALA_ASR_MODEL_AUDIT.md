# Lingala ASR Model Audit

Date: 2026-07-05

## Priority Decision

`Alvin-Nahabwe/wav2vec2-xls-r-300m-Fleurs_AMMI_AFRIVOICE_LRSC-ln-109hrs-v2` is now the highest-priority Lingala model to evaluate because it matches our current best WAXAL architecture family: XLS-R 300M + CTC.

This model should be evaluated before another broad WAXAL training run.

## Primary Model: Alvin-Nahabwe Lingala XLS-R

| Field | Status |
|---|---|
| Model | `Alvin-Nahabwe/wav2vec2-xls-r-300m-Fleurs_AMMI_AFRIVOICE_LRSC-ln-109hrs-v2` |
| URL | https://huggingface.co/Alvin-Nahabwe/wav2vec2-xls-r-300m-Fleurs_AMMI_AFRIVOICE_LRSC-ln-109hrs-v2 |
| Access status | Access granted by account owner; executable access check still required on GPU instance |
| Architecture | Wav2Vec2/XLS-R CTC, inferred from model name and intended Transformers loading path |
| Base model | `facebook/wav2vec2-xls-r-300m` |
| License | Apache-2.0, per user-provided model details |
| Reported metrics | WER about `0.1939`, CER about `0.0632`, per user-provided model details |
| Stated/implied training data | FLEURS, AMMI, AFRIVOICE, LRSC, inferred from model name |
| Framework versions | To be recorded from `scripts/check_hf_asr_model_access.py` output |
| Tokenizer/vocab details | To be recorded from access-check output |
| Output style | To be measured on WAXAL Lingala validation; do not assume punctuation/casing behavior |

## Required Access Check

Run this before any evaluation or training:

```bash
export WAXAL_NO_SYNC=1
uv run --no-sync hf auth login
WAXAL_NO_SYNC=1 make alvin-lingala-access
```

Expected output:

```text
outputs/lingala_models/alvin_access_check.json
```

The access check verifies:

- config download,
- processor/tokenizer download,
- model weights download,
- inference on 1-3 official WAXAL Lingala validation examples.

The script never prints or stores the Hugging Face token.

## Safety Assessment

| Use | Current Decision | Reason |
|---|---|---|
| Direct inference baseline | Yes, evaluate first | Actual Lingala ASR model and same XLS-R family as current WAXAL model |
| Teacher model | Yes, after validation sanity passes | Can flag Lingala label/audio mismatches |
| Initialization for WAXAL Lingala fine-tuning | Candidate | Likely stronger Lingala initialization than generic XLS-R |
| Final routed model candidate | Candidate | Only if WAXAL validation and Phase 1 test sanity beat current checkpoint-6000 Lingala |
| Automatic relabeling | No | Teacher corrections must be reviewed/thresholded first |

## Risks

- Training data details are not fully documented locally yet.
- Model is gated; access and final-solution disclosure must be documented.
- Training sources may overlap with external datasets we later consider using directly.
- Possible WAXAL Phase 1 leakage must be checked if training sources include challenge-derived data.
- Domain mismatch is possible even with strong Lingala metrics.
- The model may output a normalization style that differs from WAXAL labels.

## Secondary Models

### `noirlab/whisper-large-v3-lingala-asr`

| Field | Value |
|---|---|
| URL | https://huggingface.co/noirlab/whisper-large-v3-lingala-asr |
| Base | `openai/whisper-large-v3` |
| License | Apache-2.0 |
| Training source stated on card | Google FLEURS |
| Reported WER | `14.8266` on Google FLEURS |
| Model size | about 2B parameters |
| Role | Secondary teacher/baseline after Alvin |
| Risk | Heavy inference cost; FLEURS-only domain; model card has sparse details |

### `BrainTheos/wav2vec2-large-mms-1b-all-lingala-ojpl`

| Field | Value |
|---|---|
| URL | https://huggingface.co/BrainTheos/wav2vec2-large-mms-1b-all-lingala-ojpl |
| Base | `facebook/mms-1b-all` |
| License | CC-BY-NC-4.0 |
| Reported WER | about `0.2698` |
| Role | Diagnostic/teacher-only unless license risk is explicitly accepted |
| Risk | Non-commercial license; sparse training-data documentation; larger model |

## Decision Gate

Proceed in this order:

1. Confirm Alvin model access and 3-sample inference.
2. Run Alvin on WAXAL Lingala validation.
3. Compare Alvin against current XLS-R v2 checkpoint-6000 on WAXAL Lingala validation.
4. Run Alvin on WAXAL Lingala train for teacher diagnostics.
5. Run Alvin on WAXAL Lingala test for output sanity.
6. Decide between direct routing, Alvin-initialized fine-tuning, or Alvin-assisted filtering only.
