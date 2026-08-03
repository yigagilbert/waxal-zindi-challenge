# Corrected Phase-2 decision record — 2026-08-02/03

Zindi replaced the Phase-2 test set (announcement 2026-08-02) and extended the
deadline to **2026-08-10**. The withdrawn set was Ugandan (ach/nyn/xog/myx); the
corrected set is **Lingala + Shona**, the challenge's stated target languages.
All prior public scores were earned on the wrong data and were reset.

## Mandatory workflow (adopted after the alpha probe cost a submission)

**No candidate is submitted before it is scored on the labeled bench.**

```bash
python scripts/score_candidate.py --predictions <validation-bench predictions> --label "<name>"
```

The scorer reproduces the leaderboard metric, compares against the current
baseline, prints PASS/FAIL, and appends to `outputs/day4_h100/CANDIDATES.jsonl`.
Rule: **never submit a bench regression**; treat bench gains as a lower bound.

### Metric semantics (established from real submissions)

Two submissions differing *only* by restored casing returned byte-identical WER
(0.413864714) and different CER (0.137837 -> 0.131082). Therefore:

- **WER is case-insensitive**
- **CER is case-sensitive**
- score = 1 - (WER + CER)/2

### Bench calibration

| candidate | bench score | public score | bench delta | public delta |
|---|---:|---:|---:|---:|
| champion, no casing | 0.8779 | 0.724149 | — | — |
| champion + casing | 0.8814 | 0.727527 | +0.0035 | +0.0034 |
| champion, alpha 1.1/1.0 | 0.8727 | 0.689930 | -0.0087 | -0.0376 |

**Bench reliability depends on the class of change** (this is the operative rule):

| change class | bench transfers? | evidence |
|---|---|---|
| text / formatting | **yes, precisely** | casing +0.0035 predicted, +0.0034 actual |
| decode parameters | direction yes, magnitude no | alpha -0.0087 predicted, -0.0376 actual |
| **acoustic model** | **no** | robust +0.0019 predicted, **-0.0013 actual** |

The bench is in-domain (same speakers as training), so a model that generalizes
better to *unseen* speakers cannot be detected there. Only submit acoustic
changes with independent justification; treat bench PASS as authoritative only
for text-level changes. The bench-to-public offset (~0.154) is
acoustic domain shift: Phase 2 is new speakers and recordings by design.

Bench: 800 held-out WAXAL validation clips, 400 lin + 400 sna
(`outputs/day4_h100/bench_ids.csv`), weighted by the test mix (448 lin / 444 sna).

## Submissions

| # | file | public score | WER | CER | outcome |
|---|---|---:|---:|---:|---|
| 1 | `SUBMISSION_champion_v2.csv` | 0.724149 | 0.413865 | 0.137837 | baseline re-established |
| 2 | `SUBMISSION_champion_v2_cased.csv` | **0.727527** | 0.413865 | 0.131082 | **best** (+0.003378) |
| 3 | `SUBMISSION_alphaup_cased.csv` | 0.689930 | 0.463677 | 0.156462 | rejected (-0.037597) |
| 4 | `SUBMISSION_robust_v2_cased.csv` | 0.726259 | 0.414163 | 0.133319 | bench PASS but public **-0.001268** |

## Pipeline that produced the current best

1. Ingest 892 clips, 48 kHz -> 16 kHz (`prepare_phase2_test.py`).
2. LID by KenLM perplexity over the champion's greedy transcripts:
   **448 lin / 444 sna, zero Luganda**, no low-margin clips.
3. Champion `checkpoint-24000` (XLS-R 300M CTC), one acoustic pass, per-language
   beam+LM re-decode from cached logits: beam 400, lin alpha 0.8/beta 0.75,
   sna alpha 0.7/beta -0.5, `lm_phase2` (its lin LM is the o6 model, MD5-verified).
4. `postprocess_predictions.py` (single-dot repair).
5. `restore_casing.py` — capitalize sentence starts (CTC has no uppercase token).

## Experiments closed, with evidence

| branch | evidence | decision |
|---|---|---|
| Casing restoration | bench formatting floor 0.0449 -> 0.0079; sna formatting WER 0.1213 -> 0.0113 | **Shipped**, +0.0034 public |
| Higher LM weight (alpha 1.1/1.0) | bench 0.8727 vs 0.8814 | Rejected; public confirmed at -0.0376 |
| Sunbird-51 as second engine | bench 0.7487 vs 0.8814 (lin WER 0.4299 vs 0.1989) | Closed |
| ROVER ensemble champion + S51 | 4 rows changed, bench score unchanged | Closed |
| Forcing a trailing period | loses under both WER-normalization hypotheses; ours ends .!? 94.0% vs refs 95.0% (sna) | Rejected |
| Repetition-loop repair | 4/800 bench rows flagged, fallback better on 0; the 13 test flags are genuine repeated phrases that af51 reproduces | Closed — CTC does not loop like seq2seq |
| o6 Lingala LM | MD5-identical to `lm_phase2/lin_5gram.binary` | Already in use |
| Robustness continuation | bench 0.8833 (PASS) but public 0.726259 (-0.0013) | Rejected; established that bench cannot gate acoustic changes |
| Lexicon-constrained spelling repair | hypothesis OOV 0.77% vs reference 0.98%; only 2.6% of near-misses are OOV-vs-in-lexicon | Closed — beam+LM already constrains to the lexicon |
| Audio preprocessing / channel match | test vs validation: RMS 0.059/0.068, centroid 1762/1629 Hz, silence 31%/33% | Closed — no channel mismatch to correct |
| Word-bonus (beta) tuning | hyp/ref word ratio 0.976 lin / 0.985 sna; 81-91% of errors are substitutions, not del/ins | Closed — length is already correct |

## Open work

- **Robustness continuation — GATE PASSED** (`configs/xlsr_champion_robust_continue.yaml`):
  champion continued on its own in-domain lin/sna audio with stronger SpecAugment
  (time mask 0.05->0.10, channel masking on) plus dropout/layerdrop, LR 1e-5,
  3000 steps; required exposing `model.regularization` in `train_xlsr_ctc.py`.

  | | baseline | robust-3000 |
  |---|---:|---:|
  | lin WER | 0.1989 | **0.1916** |
  | lin CER | 0.1284 | 0.1292 |
  | sna WER | 0.1026 | **0.1023** |
  | sna CER | 0.0435 | **0.0430** |
  | bench score | 0.8814 | **0.8833** (+0.0019) |

  The gain is concentrated in Lingala WER — the weak language and 448/892 of the
  test set. In-loop `eval_cer` (~0.2594) is inflated by the trainer's known
  space-stripping quirk and is trend-only; the bench decode is the honest gate.
  Test candidate: `SUBMISSION_robust_v2_cased.csv`
  (sha256 `85837f8d357175fe9c34bf803bf9dabcaa067f2e2211dfd42e9ffd516d19c035`),
  485 of 892 rows differ from the current best.
## Error profile — where the remaining loss actually is

WER 0.4139 against CER 0.1311 is a 3.1x ratio. On the bench, **46.9% of
substituted words are within 1-2 characters of the reference**: spelling and
morphological variants (e.g. `vemutambo`/`vomutambo`), not mishearings. Errors
are 81-91% substitutions with correct word counts, and the lexicon is already
respected. Remaining word-choice errors are therefore arbitrated by the language
model, not the acoustic model or the decoder's length control.

- **CTC logit ensemble** (champion + robust-3000, log-prob averaging; required
  adding `--ensemble-checkpoints` to the pipeline): bench 0.8832 (+0.0018, PASS)
  but its gain derives from the robust model, whose bench gain did not transfer.
  Held as a diversity candidate rather than an expected gain.
- **Lingala is the weak language** (bench WER 0.1916-0.1989 vs sna 0.1023) and is
  448/892 of the test set. Its LM already includes Wikipedia-Lingala (36,510
  lines of 88,380 total); Shona's LM has no Wikipedia component. Expanding the
  language-model text is the one lever aimed squarely at word-choice errors.
