# Day-3 H100 decision record — 2026-07-31

Baseline entering the day: public `0.715391690`
(`phase2_strict_loopfix_mbrmargin_t0075_submission.csv`). Gap to captured third
place `0.720212909` is `0.004821`, 87% WER.

## Queued-job verdicts

1. **Sunbird-51 pinned-revision rerun — FAIL.** 800-clip validation, greedy,
   revision `2135317`: raw macro combined `0.2983` (ach 0.2672 / nyn 0.2675 /
   xog 0.3192 / myx 0.3391) vs the SALT anchor class (~0.25). The stale-weights
   hypothesis is dead; Sunbird-51 is closed permanently.
2. **Scored stochastic decode — succeeded technically** (24,320 rows =
   3,040 × 8 samples, zero rows with `sequence_score == 0`), enabling the rerun
   below.

## Score-aware weighted MBR — FAIL, branch closed

`select_weighted_nbest_mbr.py` tune rerun with real scores
(`has_real_scores: true`), primary = the protected t0075 validation candidate
(`0.2456792`), 5-fold nested:

- OOF gain: **−0.0014658** (8 switches).
- Full-validation optimum: **zero switches**.

With real log-probabilities, model likelihood at T=0.2 still cannot referee
candidates beyond the already-banked confidence-gated MBR. Selection-branch
tally: naive MBR, frequency-weighted MBR, three learned routers, and
score-weighted MBR all failed; only the validation-frozen margin gate ever
shipped. The 0.0316 oracle headroom requires references we do not have.
The scored Phase-2 decode was cancelled as its only consumer failed.

## Truncation repair — one surgical row

A residual-anomaly sweep of the t0075 submission (repetition + length ratio +
char-density vs audio duration) flagged 9 rows; manual review cleared 8 as
legitimate repetition or genuinely short utterances.

A naive fallback rule (`len(best)/len(af51) < 0.45, len(af51) > 80`) was
validated first and **rejected**: all 6 validation switches were catastrophic —
those clips have genuinely short references (one is a single word) and af51
hallucinates on them. Length ratio alone cannot distinguish short speech from
decoder truncation.

Audio duration can. `ID_NGAOW`: 24.04 s of audio, 60-char anchor output
(2.5 chars/s — physically implausible); 7/8 SALT samples truncate at the same
~60-char point; the eighth completes at 233 chars and af51 independently
produces a near-identical 229-char continuation. The other 7 lowest-density
rows (2.0–3.4 chars/s) show all-samples-and-af51 agreement on shortness and
were left untouched.

Repair: replace `ID_NGAOW` with the 233-char SALT sample (same engine and
orthography as the submission; cross-engine corroborated). One row changed.

Candidate: `phase2_ngaow_repair_submission.csv`
SHA-256 `bf3f30166566d0ab77d7d6e03f2b0837d99b232573e757b76adebd1b1c3f85d1`
1,500 rows, 1,500 unique IDs, exact official order, no empty targets, no
embedded newlines. Estimated public gain ≈ +0.001 (recovers ~28 deleted words
on one row).

## Test-weighted gating (methodology correction)

Phase-2 test routing is ach 500 / nyn 500 / myx 499 / xog 1, but every earlier
gate used a flat 4-language macro — giving xog 25% of the weight at 0.07% of the
test, and myx 25% at 33%. `scripts/score_test_weighted.py` re-scores under the
true mix. Re-checked candidates (test-weighted combined error):

| candidate | ach | myx | nyn | xog | macro4 | TEST-W |
|---|---:|---:|---:|---:|---:|---:|
| anchor lp0.8 | 0.2253 | 0.2950 | 0.2098 | 0.2732 | 0.2508 | 0.2433 |
| **t0075 (submitted)** | 0.2204 | 0.2858 | 0.2091 | 0.2723 | 0.2469 | **0.2385** |
| loop+mbrmargin | 0.2209 | 0.2860 | 0.2096 | 0.2721 | 0.2471 | 0.2388 |
| af51 | 0.2831 | 0.4368 | 0.3003 | 0.4017 | 0.3555 | 0.3400 |
| adapter on myx | — | 0.3078 | — | — | — | — |

No earlier decision flips: t0075 remains best and the mixed adapter still loses
on the full myx slice. The correction's value is directional — **myx carries a
0.077 error gap against nyn while occupying a third of the test**, so myx is the
entire remaining opportunity.

## Branches closed today

| branch | evidence | decision |
|---|---|---|
| 220-token cap truncation | test outputs max 151 tokens vs a 220 cap; mean ref/hyp token ratio 1.019 | Not a defect; closed |
| Cross-language token for myx | 200-clip probe: lug(50355) 0.5079, xog(50352) 0.5301 vs myx(50349) 0.3289 | myx token is correct; closed |
| myx error concentration | worst 10% of myx clips carry 19.9% of myx error (uniform ≈ 10%) | Errors are broad-based, not catastrophic; catastrophe hunting exhausted |
| Surface/format fixes | validation test-weighted deltas: force-capital +0.00005, strip `<skip>` +0.00003, strip final `.!?` +0.00167, strip all punctuation +0.00280, lowercase +0.01589 | All lose; raw output already matches reference conventions |
| myx length penalty | full 849-clip myx gate: lp0.8 0.2950, lp0.7 0.2931, lp0.6 0.2931 | Real but small (−0.0019 myx ⇒ ≈ +0.0003 public); ship only inside a bundle |

### Why the surface fixes fail: per-language annotator conventions

Checked against the WaxalNLP **training** references (the source the conventions
come from), not just validation:

| language | train refs starting uppercase | train rows containing `<skip>` |
|---|---:|---:|
| ach | 99.07% | 0.02% |
| nyn | 99.43% | 0.00% |
| xog | 99.20% | 0.23% |
| **myx** | **90.87%** | **2.20%** |

`<skip>` is an official annotation token marking inaudible spans: 266
occurrences in 184 train rows, 276 of the 293 tags in myx, appearing at start,
middle and end of transcripts (e.g. `<skip> bafubuha bataru bali...`,
`"<skip>" inzuki yaburukile ... "<skip>".`). The model emits it in 4 of 5
validation cases where the reference also has it.

Capitalization is informative, not noise: on the 8 validation rows where our
hypothesis starts lowercase, the reference is also non-uppercase in 7 (88%), and
forcing a capital on exactly those rows worsens them from 0.2375 to 0.2551.

All 14 lowercase rows and all `<skip>` rows in the current test submission are
myx. Both proposed fixes would therefore apply almost entirely to the weakest
language, which is a third of the test — they are rejected.

## Myx specialist LoRA — FAIL

`configs/whisper_salt_myx_lora.yaml`, myx-only (6,867 clips), lr 1e-4, r32.
`eval_loss` bottomed at step 750 (0.4786) then rose (0.4957 / 0.4969 / 0.5123) —
overfitting after ~1.7 epochs. Gate on all 849 myx validation clips at lp0.6:
base 0.2931 vs specialist-750 **0.3059** (+0.0128). Rejected. Ninth consecutive
negative training result.

## Removing the legacy myx adapter — the day's real gain

The submitted candidate still carries old mixed-LoRA adapter text on 254 myx
rows (retained on day 1 because it had been field-tested under the pre-fix
routing). Validation disagrees, and the matched-subset test settles it: scoring
only the myx clips that the *same* old lexical clustering would confidently
label myx — i.e. the population the 254 rows were drawn from:

| comparison | n | adapter | base lp0.6 | delta |
|---|---:|---:|---:|---:|
| matched subset (old-cluster myx) | 497 | 0.2978 | **0.2739** | **+0.0239** |
| all myx | 849 | 0.3078 | **0.2931** | +0.0148 |

The adapter is worse on exactly the clips where it is deployed, and by more than
its average deficit. The old public A/B that appeared to favour it (+0.00017)
ran under the broken routing at a different decode setting and is noise.

Since public score = 1 − (WER+CER)/2 and our combined metric is (WER+CER)/2,
254 rows × 0.0239 ÷ 1500 ≈ **+0.004**, plus ≈ +0.001 for the ID_NGAOW repair.

Candidate: `phase2_bundle_ngaow_noadapter_lp06.csv`
SHA-256 `813cd6924ad3eea6c98d55912b41b1b8afd10a569a2e61ea0b9f0de298d1d482`
254 rows differ from the 0.715392 submission (253 myx adapter rows replaced with
plain SALT lp0.6, 1 nyn = ID_NGAOW). 1,500 rows, unique IDs, exact order, no
empty targets, no embedded newlines, zero repetition loops.

## Length penalty — measured, then dropped

Full per-language validation sweeps: myx lp1.0 0.3036 / lp0.8 0.2950 / lp0.7
0.2931 / lp0.6 0.2931; ach lp0.8 0.2253 → lp0.6 0.2248; nyn lp0.8 0.2098 →
lp0.6 0.2099. lp0.6 is the myx/ach optimum, but applying it *on top of* the
protected loop+MBR candidate gains nothing (+0.00012) — its benefit overlaps
rows those operations already repaired. It is therefore used only as the
replacement text for the de-adaptered myx rows, not as a global change.

## Standing decisions

- Submit `phase2_ngaow_repair_submission.csv`; on any non-regression it becomes
  a selected submission alongside the t0075 candidate.
- No further selector, router, or margin work is justified; every branch is
  closed with out-of-fold evidence.
- GPU jobs are stopped; the VM is only needed to serve files and can be
  deallocated once the final selections are set (close/reveal 2026-08-02).
