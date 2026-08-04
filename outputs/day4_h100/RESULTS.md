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

The scorer reproduces the observed leaderboard metric semantics, compares against
the exact current baseline, requires full bench coverage, and appends to
`outputs/day4_h100/CANDIDATES.jsonl`. It is change-class aware: text/decode gains
can pass, while acoustic gains are held for an independent out-of-domain gate.
Rule: **never submit a bench regression**. Do not treat acoustic bench gains as a
lower bound; the robust continuation proved that they can reverse publicly.

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

The bench is in-domain and the official speaker metadata now quantifies the
leakage: **799/800 rows are from speakers already present in training**. Across
all 3,571 Lingala/Shona validation rows, only five are unseen-speaker clips, too
few for a gate. A model that generalizes better to *unseen* speakers therefore
cannot be selected reliably here. Only submit acoustic
changes with independent justification; treat bench PASS as authoritative only
for text-level changes. The bench-to-public offset (~0.154) proves that the WAXAL
validation bench is a poor proxy for corrected Phase 2. New speakers/recordings
are likely the largest component, but language/text conventions and distribution
shift are confounded; it is not evidence of "pure acoustic domain shift."

Bench: 800 held-out WAXAL validation clips, 400 lin + 400 sna
(`outputs/day4_h100/bench_ids.csv`), weighted by the test mix (448 lin / 444 sna).

## Submissions

| # | file | public score | WER | CER | outcome |
|---|---|---:|---:|---:|---|
| 1 | `SUBMISSION_champion_v2.csv` | 0.724149 | 0.413865 | 0.137837 | baseline re-established |
| 2 | `SUBMISSION_champion_v2_cased.csv` | 0.727527 | 0.413865 | 0.131082 | improved baseline (+0.003378) |
| 3 | `SUBMISSION_alphaup_cased.csv` | 0.689930 | 0.463677 | 0.156462 | rejected (-0.037597) |
| 4 | `SUBMISSION_robust_v2_cased.csv` | 0.726259 | 0.414163 | 0.133319 | bench PASS but public **-0.001268** |
| 5 | `SUBMISSION_champion_v2_beam1200_cased.csv` | **0.728747** | 0.412406 | 0.130101 | **best** (+0.001220 vs beam 400), rank 44 at submission time |

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

## Robustness continuation — closed as a direct candidate

- **Robustness continuation** (`configs/xlsr_champion_robust_continue.yaml`):
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

Public WER 0.4139 against CER 0.1311 is a 3.1x ratio. On the speaker-leaky
validation bench, **46.9% of
substituted words are within 1-2 characters of the reference**: spelling and
morphological variants (e.g. `vemutambo`/`vomutambo`), not mishearings. Errors
are 81-91% substitutions with correct word counts, and the lexicon is already
respected. For this validation distribution, the remaining word-choice errors
are often arbitrated by the language model. That conclusion must not be
projected onto corrected Phase 2: no test references exist, and the acoustic
bench gain already reversed publicly.

- **CTC logit ensemble** (champion + robust-3000, log-prob averaging; required
  adding `--ensemble-checkpoints` to the pipeline): bench 0.8832 (+0.0018, PASS)
  but its gain derives from the robust model, whose bench gain did not transfer.
  Held as a diversity candidate rather than an expected gain.
- **Beam width 1200** (same champion, same LMs and alpha/beta): corrected 800-clip
  bench score 0.882157 vs exact 0.881433 baseline (**+0.000723**, decode-class
  PASS). It landed at **0.728746521** public (WER 0.41240573, CER 0.130101225),
  a **+0.001219635** public gain over beam 400. File:
  `SUBMISSION_champion_v2_beam1200_cased.csv`, SHA256
  `252d053f3574897576aaa9b74f47cdd71a349f9649498a4c094a9c94a84919c3`;
  80/892 rows differ from beam 400.
- **Wider-beam curve:** beam 1800 scored 0.882506 on the same 800 clips; beam
  2400 scored **0.882632**, another +0.000475 over beam 1200. The validated test
  file is `SUBMISSION_champion_v2_beam2400_cased.csv`, SHA256
  `485add22524845156283443d64d0eec619e8230aac37f06a64a016b5f71facef`;
  it changes 33/892 rows versus beam 1200. Beam 1800 is dominated and must not be
  submitted.
- **Lingala is the weak language** (bench WER 0.1916-0.1989 vs sna 0.1023) and is
  448/892 of the test set. Its LM already includes Wikipedia-Lingala (36,510
  lines of 88,380 total); Shona's LM has no Wikipedia component. Expanding the
  language-model text is the one lever aimed squarely at word-choice errors.
- **Shona Wikipedia LM — CLOSED NEGATIVE.** It added 60,000 normalized
  CC-BY-SA Wikipedia sentences to the immutable 48,627-line fielded Shona
  corpus. On the same 800-example validation slice, its best tested point was
  0.0916 combined (alpha 0.3/beta -0.5) versus 0.0850 for the fielded LM
  (alpha 0.7/beta -0.5), a **-0.0066 score regression**. Lower alpha was worse
  (0.1440 at 0.05). No test decode and no submission.

## OOD gate status

- Official WAXAL metadata: 799/800 current bench clips have speakers seen in
  train; the full validation set yields only five unseen-speaker clips.
- FLEURS validation is **invalid for champion/robust/ensemble selection**. The
  champion's generalization-mix builder defaulted to external train + validation
  + test, so all FLEURS splits entered its training set. This also invalidates
  every continuation initialized from the champion.
- The 602 prepared FLEURS validation clips can gate a fresh-from-base model only
  if its new training mix is restricted to FLEURS train. Beating the contaminated
  champion there is strong evidence; losing to it is inconclusive.
- A clean acoustic gate can use the speaker-capped Lingala/Shona audio in the
  accessible `evie-8/afrivoices` conversion, inherited from upstream
  CC-BY-4.0 DigitalUmuganda/Afrivoice. Promote on **greedy** metrics: the
  champion's KenLM already saw AfriVoices text, so LM-routed scores are not clean.
- **AfriVoices gate prepared and validated:** 600 rows (300 lin / 300 sna), 151
  speakers, 3.37 hours, max five clips/speaker. Champion greedy baseline:
  overall WER 0.33365, CER 0.08193, combined 0.20779; lin combined 0.23946,
  sna combined 0.17482. Any new acoustic model must beat this gate as well as
  the WAXAL sanity gate before a submission is allowed.

## Operational fixes from the beam/1B audit

- Decoder reports now record KenLM directory, order, beam width, language list,
  default language, and greedy-language overrides. Their omission caused one
  invalid 2400-beam run to use Shona greedy decoding and waste compute.
- The misleading default `greedy_languages=[sna]` was removed. Corrected Phase 2
  uses beam+LM for both lin and sna unless an explicit validated override is set.
- The decoder can shard a split contiguously and releases the acoustic model's
  GPU memory before CPU KenLM search. Four 223-row shards reproduced the same
  pipeline while cutting test-decode wall time.
- Fresh acoustic models can use `model.processor_name` plus an explicit
  `model.vocab_size`. The recovered champion tokenizer object has 112 entries
  because BOS/EOS are metadata-only additions, but its actual CTC head has 110
  logits. Pinning 110 prevents silent logit-ensemble incompatibility.

## Semi-supervised branch — opened 2026-08-03 (the one untapped in-domain resource)

Every labeled lin/sna clip is already in training (14,400 + 14,109 = complete
WaxalNLP train splits). The **unlabeled splits** (40 lin / 52 sna shards, new
speakers) are official challenge data and directly target the measured weakness:
bench 0.882 on training-speaker audio vs 0.729 public on unseen speakers.
XLS-R 1B from scratch was closed the same day (eval_loss rose 0.3907 -> 0.405 by
epoch 5.6; eval_cer 0.2645 worse than champion; no usable checkpoint) — scale
does not substitute for speaker diversity.

Plan (noisy-student):
1. `prepare_unlabeled_stream.py` — stream 12,000 clips/language to 16 kHz FLAC
   (streaming avoids a ~40 GB parquet cache; resumable manifest).
2. `pseudo_label_decode.py` — champion greedy + beam+LM per clip; greedy/beam
   agreement is the reference-free confidence filter (keep low-disagreement
   clips only), plus duration/length sanity.
3. Continue champion checkpoint-24000 on labeled + filtered pseudo, with the
   robustness SpecAugment settings (augmented student, clean teacher labels).
4. **Gates (both required):** AfriVoices OOD, 600 clips/151 unseen speakers —
   champion baseline WER 0.33365 / CER 0.08193 / combined 0.20779 — must
   improve; the 800-row bench must not regress materially. The in-domain bench
   alone is NOT a valid promotion gate for acoustic changes (established by the
   robust-continuation reversal).

Disk cleared for this run (96 GB): failed Whisper full-FT (35 GB, documented
negative), Ugandan-language datasets and caches, superseded checkpoints — the
LoRA adapters and robust-3000 were pushed to `yigagilbert/waxal-phase2-checkpoints`
first. `/mnt/serving_eval` (separate project) untouched.

Rules note: the unlabeled split is part of the challenge-specified WaxalNLP
dataset (not external data). Pseudo-labeling uses only our own model's outputs
on that official data; Phase-2 *test* audio remains inference-only. Disclosed
in `docs/RULES_AND_DATA_USE.md`.

## Submission ledger correction and beam closure — 2026-08-04

`eCRxcvG1` returned **0.728575887** — identical to nine decimals with
`vipX9aTt` from ~22 h earlier. The beam-2400 file had already been submitted by
the previous session (that was the unexplained ledger entry); today's upload was
a duplicate. Two process rules adopted:

1. **Submission ledger**: every submitted file's SHA-256 and public score are
   recorded here at submission time; check the ledger before uploading.
2. **Minimum-effect rule**: decode-class candidates require a bench delta of at
   least +0.002 before spending a submission. Beam 2400's +0.000475 bench gain
   came back as **-0.00017 public** — sub-millipoint decode gains are inside
   public-split noise.

| file | sha256 (prefix) | public |
|---|---|---:|
| SUBMISSION_champion_v2.csv | — | 0.724149 |
| SUBMISSION_champion_v2_cased.csv | 5e934535 | 0.727527 |
| SUBMISSION_alphaup_cased.csv | d374969c | 0.689930 |
| SUBMISSION_robust_v2_cased.csv | 85837f8d | 0.726259 |
| SUBMISSION_champion_v2_beam1200_cased.csv | — | **0.728747 (best)** |
| SUBMISSION_champion_v2_beam2400_cased.csv | 485add22 | 0.728576 (x2: vipX9aTt, eCRxcvG1) |

**Beam-width branch closed**: 400 -> 1200 gained +0.0012 public; 1200 -> 2400
lost -0.00017 public despite a positive bench delta. Optimum stands at 1200.

## Noisy-student chain — armed 2026-08-04

`sslchain` tmux session: waits for download+decode -> runs
`filter_pseudo_labels.py` (disagreement <= 0.10, rate/word sanity) -> attaches
kept rows as a `pseudo` split inside `data/processed/hf_dataset` -> if kept >=
10,000, launches `configs/xlsr_champion_ssl_continue.yaml` (champion
continuation, SpecAugment student, LR 2e-5, 6,000 steps) automatically.
Promotion requires BOTH gates: AfriVoices OOD improvement over combined 0.20779
AND no material bench regression. Stale tmux sessions from the previous
operator (1B, beam shards, OOD setup) were cleaned.

## Noisy-student semi-supervised branch — CLOSED 2026-08-04

Executed in full: 24,000 unlabeled WaxalNLP lin/sna clips streamed to FLAC,
pseudo-labeled by the champion (greedy + beam+LM), confidence-filtered by
greedy/beam disagreement <=0.10 (21,208 kept, 88%), student trained from
checkpoint-24000 on labeled+pseudo (51.5k clips, language-balanced) with the
SpecAugment robustness recipe, LR 2e-5, best at step 4000 by eval_cer.

Dual gate, identical tooling, fresh baselines:

| measurement | champion | student | verdict |
|---|---:|---:|---|
| AfriVoices OOD greedy (600 clips, 151 unseen speakers) | 0.2046 | **0.2010** | acoustic model genuinely improved |
| AfriVoices OOD routed (beam+LM, as deployed) | **0.0909** | 0.0929 | deployed system worse |
| 800-row bench, beam 1200 | **0.882157** | 0.879649 | -0.0025, breaches the -0.002 line |

**Key finding:** the new-speaker pseudo-data improved the raw acoustic model on
unseen speakers (greedy -1.8% rel), but the gain does not survive beam+LM
decoding — the LM already corrects the error class the student learned to
avoid, and the student's new errors are less LM-recoverable. Acoustic and LM
contributions overlap rather than compose.

Teacher+student logit ensemble (frame-level log-prob averaging): OOD routed
0.1018 vs 0.0909 — WORSE; ensemble greedy collapsed to 0.5989 because the
student's CTC alignments drifted (masking-heavy continuation), so frame-wise
averaging smears both models' peaky distributions. CTC frame ensembling only
works between alignment-compatible checkpoints.

Zero submissions spent on this branch. Student checkpoint-4000 archived at
`yigagilbert/waxal-phase2-checkpoints/xlsr_champion_ssl/checkpoint-4000`.

## Campaign state after closure

Deployed ceiling: champion + beam 1200 + casing = public **0.728747**.
Every branch is closed with direct evidence (engines, ensembles x3, formatting,
LM weight/text, loop/lexicon repair, beta, channel, robust continuation, 1B
scale, noisy-student). Remaining value is selection discipline for the private
70-80% reveal: pick two submissions that are strong AND diverse (max-of-two
under reshuffle), keep the record reproducible for top-10 code review.
