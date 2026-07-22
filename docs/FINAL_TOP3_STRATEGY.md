# Final Top-3 Strategy Review

Evidence-based answer to: *what is the best realistic path beyond 0.861?* Companion docs:
[TOP3_GAP_ANALYSIS.md](TOP3_GAP_ANALYSIS.md) (the gap is Lingala),
[MODEL_ZOO_ORACLE_ANALYSIS.md](MODEL_ZOO_ORACLE_ANALYSIS.md) (routing/ensemble upside).

_2026-07-20. Days to close: ~12. Phase-2 test drops ~07-26._

---

## 1. Re-evaluation of the "no more acoustic training" conclusion

Honest audit of each negative, including whether it was a *fair* test:

| Experiment | Hypothesis tested | Fair test? | Verdict on the verdict |
|---|---|---|---|
| XLS-R 1B | more capacity helps | **Mostly fair.** Tracked the 300M eval curve point-for-point over thousands of steps before the pre-registered kill. Capacity was not the binding constraint. | Negative stands. |
| clean-audio-v2 | cleaned + 3× external Lingala helps | **Fair** (full-length run). Worse everywhere incl lin. | Negative stands; established the read-speech trap. |
| clean_v3 continuation | continue converged champion at LR 2e-5 on clean_v3 | **Fair for what it tested** — but it only tested *low-LR continuation on an Afrivoice-dominated mix*. Flat loss = nothing in-domain to learn there. | Negative stands **for that mix**; says little about a WAXAL-heavy mix. |
| Whisper large-v3 LoRA | different architecture ceiling | **Fair as a cheap probe.** Plateauing at ~0.38 combined (val, greedy) by step 1000 with only 0.5% params trainable; decelerating. Full FT is 20× cost and Whisper has no KenLM-beam integration — our single biggest lever doesn't apply. | Negative stands for the 12-day window. |
| champion-recipe retrain on clean_v3 | same recipe + cleaner data | **NOT fully fair — the user's critique is partially right.** Three flaws: (a) early-stopped at step 9500 on a muddled in-loop metric (cleaned-val CER, broken word counts); (b) LR was still ~2.0e-4 (67% of peak) — the run **never entered the decay phase** where the champion gained its last points; (c) epoch math: champion did ~6.9 epochs (384k samples / 55k balanced rows); the retrain's balanced set is 105k rows, so even the full 24k steps = only 3.7 epochs, and it was killed at **1.45 epochs**. | **However** the raw-val gate showed lin greedy 0.403 vs champion 0.265 — a 0.14 deficit. Decay-phase gains are historically ~0.006–0.02, not 0.14. So the early stop was methodologically sloppy but almost certainly did not flip the outcome **on that mix**. The un-tested cell is: **champion mix (WAXAL+FLEURS+SALT only), cleaned audio, full schedule** — clean-audio-v2 added external Lingala, clean_v3 runs were OOD-dominated or truncated. That specific hypothesis ("does *cleaning alone* help, at full schedule?") has never had a fair full-length test. |

**Revised conclusion:** "no more acoustic training" was overstated in one respect. Two targeted
runs remain justified (and only these two):
1. **Lingala specialist** from the champion on in-domain lin data (cheap, targets 100% of the gap).
2. **One full-schedule champion-recipe run on the cleaned *in-domain* mix** (clean_v3 filtered to
   waxal+fleurs+salt sources; no early stopping; kill only on NaN/divergence). This is the clean
   test of the user's hypothesis with both confounds (truncation, OOD mix) removed.

Everything else acoustic stays dead.

## 2. Model-zoo / oracle reality check

The historical zoo (Alvin-Lingala, noirlab Whisper-lin, v2-ckpt6000, 1B)**' checkpoints and
prediction CSVs died with the old boxes** and were never git-tracked. The rebuildable candidate
set today: champion-greedy, champion-beam+LM (tuned), recipe-v3 ckpt-6500 (kept; worse overall
but possibly complementary per-sample), and any specialist this plan produces.
`scripts/build_prediction_router.py --mode oracle` computes per-sample best-of over these on
validation. **Gate: if oracle − best single ≥ 0.005 pooled, build the rules router (same script,
`--mode route`, CV-tuned); if < 0.005, routing is not where the points are.**

## 3. TTA (test-time augmentation)

Deferred, deliberately. Candidate-selection TTA (speed 0.95/1.05, gain-norm, trim) triples
decode cost for typical CTC gains of ~0.002–0.005, mostly on clips the LM already fixes. It is
strictly dominated by the lin-LM and specialist work in expected value per GPU-hour. Revisit only
if both land and GPU is idle before 07-26. (No `run_tta_decode.py` until then — building it now
would be motion, not progress.)

## 4. Audio LLMs / other architectures (requested assessment)

| Option | Path to beating 0.861 | Verdict |
|---|---|---|
| Whisper large-v3 full FT | LoRA probe plateaued ~0.38; full FT ~20× cost, no KenLM-beam lever, 12 days | **No.** |
| MMS-1B-all / SeamlessM4T | non-commercial license — documented as prohibited-risk for this prize competition | **No (rules).** Diagnostic-only stays diagnostic. |
| Qwen2-Audio / audio LLMs | no evidence of Lingala/Shona coverage; prompt-ASR WER on unseen low-resource languages is typically catastrophic; large setup cost | **No.** |
| NVIDIA NeMo (Fast)Conformer | training from scratch on 90k clips will not beat a fine-tuned 300M XLS-R; no pretrained lin/lug/sna checkpoint exists | **No.** |
| Sunbird whisper-large-v3-salt as **lug teacher/rescorer** | open model, lug only (15% share, already CER 0.040) | Marginal; not worth the days. |
| Bigger Lingala **text** for the KenLM | the one proven lever (lin 0.265→0.171 came from exactly this); lexical coverage is part of lin's residual error | **Yes — quick win #1.** Wikipedia-lin (CC-BY-SA), plus any CC/public-domain lin corpora; disclose per Data-tab rules. Try order 6 as well. |

## 5. The one full-schedule experiment (design, per request)

Config: `configs/xlsr_300m_champion_recipe_clean_v3_indomain_full.yaml`
- Start: base `facebook/wav2vec2-xls-r-300m` (identical to champion — this is a recipe repeat,
  not a continuation).
- Data: `clean_audio_v3` train **filtered to `source_datasets: [waxal_official_clean,
  google/fleurs, Sunbird/salt]`** (new additive trainer filter) = the champion's exact mix, but
  cleaned/trimmed audio and with the 3,906 defective rows excluded. Balanced set ≈ 52–55k rows →
  24k steps ≈ **6.7–7 epochs = epoch-parity with the champion.**
- LR 3e-4, warmup 2400, max_steps 24000, bf16, eff-batch 16 — champion recipe verbatim.
- **No early_stopping block.** Kill conditions: NaN/inf loss, or loss divergence >2× for 1k+
  steps. NOT early underperformance.
- Eval/save 500; `save_total_limit 5` + best-by-eval_cer retained.
- Gates (record, don't kill): raw-val beam+LM sweep at 8k / 16k / 24k. Decision only at 24k:
  promote iff pooled < 0.1276 − 0.003 with lin ≤ 0.171.
- Cost: ~9–10 GPU-h + ~18 GB disk (root). Expected curve: in-loop cer ~0.25 by 6k, real gains
  (if any) in the 16k–24k decay phase.
- Honest prior: ~25–30% it beats the champion at all; if it does, likely by 0.002–0.008 pooled.

## 6. Ranked plan

| # | Action | When | Cost | Expected public gain | Risk | Gate |
|---|---|---|---|---|---|---|
| 1 | **Lingala LM corpus expansion** (+ try 6-gram) → lin re-sweep | today, CPU | ~1–2 h | +0.002..+0.007 | none (decode-only) | lin sweep beats 0.1711 → new test decode + submit |
| 2 | **Oracle analysis** (`build_prediction_router.py --mode oracle`) on regenerable candidates | today, ~1 h GPU for candidate decodes | low | informational | none | oracle−best ≥0.005 → build router |
| 3 | **Lingala specialist**: continue champion on `processed_generalization_mix` filtered `languages: [lin]` (WAXAL-lin + FLEURS-lin, the audio the champion already knows), LR 5e-5, 4k steps | 24 h | ~3–4 GPU-h | +0.004..+0.018 (lin −0.01..−0.04) | moderate; forgetting irrelevant (routing-only use, Phase-2-safe via 98.5% LID) | raw-val lin beam+LM < 0.166 → route lin to specialist, resubmit |
| 4 | **Rules router** (if #2 gate passes) | 24 h, CPU | low | +0.002..+0.005 | low (CV-tuned) | CV gain ≥0.003 → apply to test |
| 5 | **Full-schedule clean-indomain retrain** (§5) | 2–3 days | ~10 GPU-h | 0 to +0.008 | high (prior ~25–30%) | run only when GPU otherwise idle; decide at 24k only |
| — | TTA | only if 1–5 exhausted | med | +0..+0.004 | low | — |
| ✗ | Whisper full FT, audio LLMs, MMS/Seamless, NeMo, 1B, more Afrivoice-heavy training | never (this competition) | — | — | — | — |

**Stacked realistic outcome: ~0.87–0.885 (solid top 5–7). Top 3 (0.9105) requires lin ≈0.09–0.10 —
reachable only if #3 or #5 overdelivers.** All items are Phase-2-compatible (specialist and router
key off *predicted* language / decoder-side features only).

## 7. Answers to the five closing questions

1. **Truly at the acoustic ceiling?** For *multilingual training on the mixes tried* — yes
   (5 experiments). Two cells were never fairly tested: lin-specialist fine-tuning, and
   full-schedule cleaned *in-domain* retraining. Both are now scheduled.
2. **Routing/ensemble upside left?** Unknown but measurable today; the historical zoo is lost, so
   the oracle bounds it with regenerable candidates. Expect modest (+0.002..0.005) unless the
   specialist lands (then routing is how it's deployed).
3. **One full-schedule retrain justified?** Yes — exactly one, §5 design, without the two flaws
   (truncation, OOD mix) that invalidated the last attempt. Low prior, bounded cost, run at idle
   priority.
4. **Audio LLMs / other architectures?** No — every candidate fails on license, language
   coverage, decode-lever compatibility, or time (§4).
5. **Single best next action:** **Lingala LM corpus expansion + lin re-sweep today** (highest
   certainty per hour), with the **Lingala specialist** as the headline 24-hour experiment —
   because the entire top-3 gap is Lingala.
