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

Notes on the surface tests: `<skip>` is a legitimate annotation token (28
validation references contain it) and the model emits it correctly in 4 of 5
cases; references start with a capital only 96.6% of the time versus 99.6% for
our output, so forcing capitalization moves away from the target.

## Standing decisions

- Submit `phase2_ngaow_repair_submission.csv`; on any non-regression it becomes
  a selected submission alongside the t0075 candidate.
- No further selector, router, or margin work is justified; every branch is
  closed with out-of-fold evidence.
- GPU jobs are stopped; the VM is only needed to serve files and can be
  deallocated once the final selections are set (close/reveal 2026-08-02).
