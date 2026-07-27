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

## 6b. Outcome log (updated as experiments complete)

| date | experiment | result | evidence |
|---|---|---|---|
| 2026-07-22 | LM expansion (#1) | **POSITIVE — banked.** lin 0.1711→0.1683 (val, o6+Wikipedia, α0.8/β0.75); public 0.8610→**0.86216** (`champ_lmv2_o6.csv`) | `outputs/analysis/lm_v2_lin_sweep_o6*.json` |
| 2026-07-22 | Oracle/router (#2) | **CLOSED.** oracle_upside 0.0033 < 0.005 gate (only real per-sample gap is lug, 15% share) | `outputs/analysis/model_zoo_oracle_validation.json` |
| 2026-07-22 | Lin specialist (#3) | **CLOSED.** Continuation on the champion's own lin slice ≈ parity (ckpt-1000 external: greedy 0.2705 / beam 0.1755 vs 0.2647 / 0.1683) — no new information in already-seen data. Side discovery: **in-loop eval on this box strips reference spaces** (276,500−227,817 = 48,683 = exact space count) → in-loop CER inflated ~2×, trend-only | `outputs/analysis/lin_specialist_v2_ckpt1000_check.json` |
| 2026-07-23 | Full-schedule clean in-domain run (#5) | **NEGATIVE — champion retained.** 24k steps / 7.7 epochs, no early stop, champion mix source-filtered (48,978 rows). External gate ckpt-24000: lin best 0.1881 vs champion 0.1683 (+0.020 worse); greedy 0.284 vs 0.265. Only remaining variable was the cleaned/**trimmed** audio → trimming creates a train/inference mismatch vs raw test audio. **Lesson: do not trim audio for this task.** Sixth-and-final acoustic negative; ceiling proven | `outputs/analysis/fullrun_ckpt24000_lin.json` |
| 2026-07-27 | Teacher evaluation: `huwenjie333/whisper-v3-ft-af51` | **DISCARDED** (>0.26 threshold). WAXAL-lin validation: WER 0.4534 / CER 0.2259 / combined **0.3397** — 2× worse than champion+LM (0.1683), worse than champion greedy (0.2647). Its 0.238 read-speech benchmark did not transfer to spontaneous WAXAL audio. Output is fluent real Lingala (no hallucination; one orthography-drift example) — domain mismatch, not model failure. Cannot route lin; cannot teach (worse than the student). | `outputs/analysis/whisper_af51_lin_eval.json` |
| 2026-07-27 | **Recovered model zoo** (`yigagilbert/waxal-private-artifacts`, old private HF dataset) | **IN PROGRESS.** Recovers artifacts declared lost with the first box: `clean_train_alvin_lingala_v1.csv` (teacher-cleaned lin manifest), **Alvin-Lingala predictions on validation AND test**, xlsr-v2 ckpt5000/5500/6000 predictions (val+test), old zoo analyses. Reopens the routing branch that was closed for lack of candidates — with test predictions already on disk (no model inference needed to apply a router to test). Next: fresh `language_safe` eval of Alvin-lin validation vs champion 0.1683; oracle with real candidates. | `recovered/outputs/lingala_models/*`, `outputs/analysis/alvin_lin_val_eval.json` (pending) |

| 2026-07-27 | **Phase 2 baseline** (`phase2_champion_nometa.csv`) | Pipeline ran clean on the new 1,500-clip no-metadata test: **0 empty transcripts, 0 dot-only, postprocess changed 0 rows** (no stub-clip defect in Phase 2). **LID mix FLIPPED vs Phase 1: lug 945 (63%) / sna 310 (21%) / lin 245 (16%)** — Luganda-dominant, Lingala (our weakest) shrank 44%→16%. Weighted val-equivalent ≈ 0.118 → projected public ~0.87–0.885. **Strategy flip: Luganda optimization is now worth ~4× Lingala** — next levers: Alvin (name says 313-hr Luganda ft) on lug validation; Wikipedia-lug LM expansion + lug re-sweep. | `outputs/analysis/phase2_nometa_report.json` |

| 2026-07-27 | Phase-2 collapse triage | Baseline scored **0.2829** (WER 0.93/CER 0.50) vs Phase-2 leaders ~0.65 (everyone collapsed from 0.9x — massive domain shift; 19–29 s clips, 16 kHz clean). Ruled out: CSV alignment, sample rate, stubs, **and LID** — new acoustic LID probe (`train_audio_lid.py`, val acc 99.4%) independently confirms the transcript-LID mix (lin 267/lug 961/sna 272 vs 245/945/310). Alvin closed on lug too (0.2409 vs champion 0.1259 greedy). Remaining suspects: KenLM-beam hallucination on half-heard audio vs raw acoustic collapse — decided by the greedy A/B submission. | `outputs/audio_lid/audio_lid_report.json` |

| 2026-07-27 | **Phase-2 mystery SOLVED: unseen-language generalization test** | af51 probe transcripts identify the 1,500 clips as **Acholi/Lango, Runyankole-Rukiga, Lusoga, Lumasaba** etc — not lin/lug/sna. Both in-house LIDs coerced clips into the trio; the champion transliterated correctly-heard content into Luganda phonology (ID_TBDTM comparison is the smoking gun) → 0.2829. Leaders (~0.65) run multilingual African models. **Pivot: af51 (previously discarded for Phase-1 domains) becomes the primary Phase-2 engine.** Submission `phase2_af51.csv`; refinement ladder = beams, normalization A/B, language forcing. | `outputs/analysis/phase2_whisper_lid_probe.csv` |

| 2026-07-27 | **`phase2_af51.csv` → PUBLIC 0.6773, RANK 2** | af51 full-set transcription (greedy, language_safe-normalized) jumped 0.283→0.6773 (WER 0.5005/CER 0.1449) — #2 behind Sophey (0.7087, WER 0.4588/CER 0.1237; gap 0.0314). Refinement ladder: beam-5 re-decode (running), Sunbird Ugandan-language ASR as alternative engine A/B, normalization A/B, per-cluster routing between engines. | Zindi submissions page |

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
