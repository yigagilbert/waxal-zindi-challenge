# Phase 2 — No-Metadata Readiness

Phase 2's test set has **no language labels and unknown IDs** (no `lin_`/`lug_`/`sna_` prefixes).
Every submission pipeline — champion, continuation, or Whisper — must therefore identify the
language from audio, then apply the per-language LM/decode. This is already solved; this doc is the
checklist any **new** champion candidate must pass before we rely on it for Phase 2.

_Last updated: 2026-07-20._

## Current solution

`scripts/run_no_metadata_pipeline.py`:
1. Greedy CTC decode of each clip (one GPU pass, logits cached).
2. **Language ID** by scoring the greedy transcript under each per-language KenLM (lowest
   perplexity wins) — no metadata used. Measured ~98.6% LID accuracy on validation.
3. Per-language **beam+LM re-decode** of the cached logits (no second GPU pass).
4. Route each clip's output by predicted language.

## Readiness checklist for a NEW champion candidate

- [ ] Runs from audio alone — **no dependence on ID prefixes** anywhere in the path.
- [ ] LID uses the **expanded** LMs (`data/lm_expanded`), consistent with the decode LMs.
- [ ] Per-language decode params updated to the current best:
      **lin α0.9/β0.5 · lug α0.4/β−0.5 · sna α0.7/β−0.5, beam 400.**
- [ ] `--greedy-languages` is now **empty** — sna decodes with beam+LM (it flipped with the
      Afrivoice-expanded LM; the old "sna = greedy" override is obsolete).
- [ ] Re-run LID accuracy on validation with the expanded LMs; confirm ≥ ~98%.
- [ ] Degenerate-output guards (dot-only/empty/very-short) applied post-route.
- [ ] Output covers all IDs, aligns to SampleSubmission, passes `validate_submission.py`.

## For a Whisper champion

- Whisper needs no language prefix to decode, but still route LID → any language-specific
  post-processing/LM rescoring. Validate LID on Whisper greedy transcripts (perplexity-LID
  assumes reasonable transcripts; check it holds for Whisper output).

## Validate the no-metadata path (metadata-blind on the labeled validation)

```bash
python scripts/run_no_metadata_pipeline.py \
  --checkpoint champion/checkpoint-24000 \
  --dataset-dir data/processed_generalization_mix --split validation \
  --kenlm-dir data/lm_expanded --order 5 --beam-width 400 \
  --per-language-params '{"lin":{"alpha":0.9,"beta":0.5},"lug":{"alpha":0.4,"beta":-0.5},"sna":{"alpha":0.7,"beta":-0.5}}' \
  --output outputs/analysis/phase2_nometa_validation.json
```
Confirm the routed no-metadata combined ≈ the metadata-aware combined (~0.135). A large gap means
LID is misrouting — investigate before trusting Phase 2.

> Check `run_no_metadata_pipeline.py`'s exact flag names before running; adjust
> `--per-language-params` / `--greedy-languages` to match the current signature.

## Status

- Phase 1 (metadata) pipeline: **proven**, 0.861 public.
- Phase 2 (no-metadata) pipeline: **VALIDATED 2026-07-20** with the expanded LMs + current best
  params (lin 0.9/0.5, lug 0.4/−0.5, sna 0.7/−0.5, beam 400, sna no longer forced greedy).

### Validation result (champion/checkpoint-24000, generalization_mix validation, 4235)
Report: `outputs/analysis/no_metadata_validation_report_expanded.json`

- **LID accuracy = 98.56%** (lug perfect: 664/664; lin 1820/1844; sna 1690/1727).
- **Routed no-metadata combined = 0.1256** (WER 0.1741, CER 0.0771) vs **metadata-aware pooled
  ≈ 0.1275** — essentially identical (routed is marginally better). Per-language routed:
  lin 0.1711, lug 0.1146, sna 0.0831 — matches the metadata-aware per-language numbers.
- **The ~1.4% LID misroutes are almost all stub / near-empty clips** (greedy transcripts like
  "s", "bat", "tu", "", "m") — the unfixable ~1s stub clips (76 defaulted on empty transcript).
  On real-content clips LID is near-perfect, so routing costs ~nothing.

**Conclusion: the 0.861 pipeline is Phase-2-safe.** With no metadata, LID + routed decode
reproduces the metadata-aware score. Use `run_no_metadata_pipeline.py` (params file
`outputs/analysis/best_decode_params.json`, `--greedy-languages` empty, `--beam-width 400`,
`--kenlm-dir data/lm_expanded`) to generate the Phase-2 test submission when it drops.
