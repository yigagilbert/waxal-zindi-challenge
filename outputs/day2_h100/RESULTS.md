# Day-2 H100 decision record — 2026-07-31

## Leaderboard position

- Current best after the protected stochastic-MBR margin pass at threshold
  `0.075`: score `0.715391690`, CER `0.119741636`, WER `0.449474983`.
- Third place: score `0.720212909`, CER `0.118499503`, WER `0.441074678`.
- Gap to the previously captured third-place score: `0.004821219`.
- Under the displayed score formula
  `score = 1 - (WER + CER) / 2`, WER accounts for `0.004200153`
  (87.1%) of the remaining gap and CER for `0.000621066` (12.9%).

The remaining gap is mainly word selection and word-boundary/acoustic error, not
cosmetic character normalization.

## Submitted loop repair

Submitted:

`phase2_strict_loopfix_submission.csv`

It starts from the current best candidate and changes only three obvious repetition
hallucinations:

- `ID_TZEWZ`
- `ID_ZVRJE`
- `ID_XKGCE`

The rule is reference-free and deterministic: switch from SALT to the af51 fallback
only when a repeated 4-gram occurs at least four times and the fallback has a lower
maximum repetition count.

Validation result on all 3,040 official validation clips:

| candidate | WER | CER | combined error | gain vs base |
|---|---:|---:|---:|---:|
| current SALT anchor | 0.3820376 | 0.1168605 | 0.2494491 | — |
| strict loop fix | 0.3781416 | 0.1144968 | 0.2463192 | +0.0031299 |

All five validation switches were improvements. The submission has 1,500 rows,
1,500 unique IDs, exact test order, and no empty targets.

## Experiments closed

| branch | result | decision |
|---|---|---|
| Standard beam n-best | Beam 8/16 produced only 1.25 unique hypotheses per clip; beam 32 exceeded memory | Closed; no usable diversity for LM fusion |
| Train-only character 5-gram LM router | All language/fold thresholds selected “never switch” | Closed |
| Orthographic replacement mining | Held-out score worsened for every language; emitted zero rules | Closed |
| Naive stochastic 8-sample MBR | Full-validation combined error 0.2509943 vs anchor 0.2494491; every language worsened | Closed |
| Global confidence-gated MBR | Nested 5-fold gain +0.0013396; the exact protected post-loop operation at margin 0.1 gains +0.0004663 beyond the loop fix and transferred +0.0009668 publicly | Successful; adjacent threshold sweep completed |
| Per-language confidence-gated MBR | Nested 5-fold gain +0.0008468 and Acholi regressed | Closed; failed the predeclared +0.001 gate |
| Extend the existing adapter to all corrected-routing Myx clips | Full 849-clip Myx gate worsened combined error from 0.2943425 to 0.3070247 (WER +0.01065, CER +0.01472). The new decode reproduced 233/241 old-gate hypotheses exactly, so this is not a merge-path artifact. | Closed; the earlier 241-clip gate was not representative |
| More LID work | Fused validation accuracy 3038/3040; test routing 500 Ach / 500 Nyn / 499 Myx / 1 Xog | Closed; routing is no longer a material gap |

## Ranked remaining causes

1. Phase-2 acoustic/annotation domain shift. The official validation score implied by
   the current anchor is about 0.7506, but public is 0.7154. That approximately
   0.035-point transfer loss is much larger than any remaining decode lever.
2. SALT acoustic/lexical ceiling, especially Lumasaba/Myx. Myx is the weakest official
   validation language at 0.2943 combined error. The existing mixed LoRA appeared to
   help on the original 241-clip gate, but worsened the complete 849-clip Myx slice;
   the full fine-tune failed all four languages and showed forgetting.
3. WER-heavy residual errors. WER is 87.0% of the gap to the previously captured third
   place, so punctuation,
   casing, and broad spelling rewrites cannot close it.
4. Rare catastrophic repetition. Three repaired Phase-2 rows produced a very large
   `+0.008037749` public gain; no comparable residual loop remains.
5. Remaining LID errors. At 2/3040 validation errors, this is negligible.

## Post-0.714414 result

The strict loop fix gained `+0.008037749` publicly. The next gated candidate is:

`phase2_strict_loopfix_mbrmargin_submission.csv`

It retains the exact 0.714414 submission on every trusted adapter and loop-repair row,
then changes 20 additional rows using stochastic alternatives only when the
validation-frozen MBR anchor advantage exceeds `0.1`.

Exact operation-order validation:

| candidate | WER | CER | combined error | incremental gain |
|---|---:|---:|---:|---:|
| strict loop fix | 0.3781416 | 0.1144968 | 0.2463192 | — |
| protected loop + MBR margin | 0.3774365 | 0.1142693 | 0.2458529 | +0.0004663 |

The generic learned SALT/af51 fallback router was rejected: nested OOF gain was
`-0.0000248`. Broader repetition rules were also rejected because they selected
normal repeated descriptions and worsened validation.

The candidate scored `0.715381131` publicly: `+0.000966827` over the loop-only
submission. WER improved by `0.001969142`; CER worsened by only `0.000035490`.
This confirms the remaining gain is predominantly better word hypotheses rather
than formatting or character normalization.

## Next submission: margin 0.075

An exact adjacent-threshold sweep, with the same protected operation order, gave:

| margin | validation switches | Phase-2 switches | validation combined error | gain vs loop |
|---:|---:|---:|---:|---:|
| 0.050 | 285 | 89 | 0.2460945 | +0.0002247 |
| **0.075** | **132** | **38** | **0.2456765** | **+0.0006427** |
| 0.100 | 69 | 20 | 0.2458529 | +0.0004663 |
| 0.125 | 31 | 11 | 0.2464474 | -0.0001282 |
| 0.150 | 15 | 6 | 0.2463294 | -0.0000102 |

Margin `0.075` is the only justified next point. It improves validation WER and CER
relative to the submitted 0.1 candidate, preserves all 20 existing MBR changes and
all trusted adapter/loop repairs, and adds 18 Phase-2 rows. Margin 0.05 is already
over-switching; margins above 0.1 discard validation gain.

Submit:

`phase2_strict_loopfix_mbrmargin_t0075_submission.csv`

SHA-256:

`ba1dbadace3b07abe4d188b2cebdd781ff1663d2414ddd7211ab58a5d1010741`

The file has 1,500 rows, 1,500 unique IDs, exact official Phase-2 order, and no
empty transcriptions.

Public result: `0.715391690`, only `+0.000010559` over margin `0.1`. WER improved
by `0.000009288` and CER by `0.000011830`. This is effectively flat and closes
the adjacent-margin branch.

## Post-margin selector audit

- Eight stochastic samples contain substantial oracle diversity:
  top-sample combined validation error `0.2546666`, per-utterance oracle
  `0.2230548`, oracle headroom `0.0316118`; mean unique hypotheses `4.42`.
- The original MBR implementation discards duplicate frequency. A replacement
  frequency-weighted MBR with a deterministic-anchor prior was evaluated with
  nested five-fold tuning. It failed: OOF gain `-0.0005601`; the full-validation
  optimum was zero switches.
- A 45-feature learned router from the protected loop baseline to each row's
  best MBR alternative also failed: OOF gain `-0.0002293`, with only 2/5
  positive folds.
- The same nested router against the existing Sunbird-51 validation/test
  predictions selected zero switches in the final fit; OOF gain was
  `-0.0000451`. That artifact is not a submission candidate.
- Therefore candidate diversity is not the current bottleneck; reliable
  reference-free selection is. A scored stochastic validation decode is in
  progress because the original n-best artifact accidentally recorded every
  sequence score as `0.0`.

The public Sunbird-51 repository is a preview whose weights may change. The
current pinned revision is `213531767f739bc5b1e5dcb3e2ec9e112674ab67`, with
the distinct token map `ach=50357`, `nyn=50322`, `xog=50310`, `myx=50329`.
An 800-row validation rerun of that exact revision is queued behind the scored
SALT decode so an older preview artifact is not used as the final verdict.
