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

## Standing decisions

- Submit `phase2_ngaow_repair_submission.csv`; on any non-regression it becomes
  a selected submission alongside the t0075 candidate.
- No further selector, router, or margin work is justified; every branch is
  closed with out-of-fold evidence.
- GPU jobs are stopped; the VM is only needed to serve files and can be
  deallocated once the final selections are set (close/reveal 2026-08-02).
