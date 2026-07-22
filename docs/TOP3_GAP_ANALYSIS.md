# Top-3 Gap Analysis

Where the remaining error actually lives, and what closing it is worth. Basis: validation
per-language combined (0.5·WER+0.5·CER) from the beam-400 expanded-LM decode
(`kenlm_sweep_lin_refine.json`, `kenlm_alpha_beta_sweep_expanded.json`), pooled by **test**
language shares (lin 1866/4253 = 43.9%, lug 638 = 15.0%, sna 1749 = 41.1%).

_Last updated: 2026-07-20._

## Current position

| | WER | CER | combined |
|---|---|---|---|
| lin (val, beam+LM tuned) | 0.2136 | 0.1285 | **0.1711** |
| lug | 0.1889 | 0.0404 | **0.1148** |
| sna | 0.1211 | 0.0502 | **0.0857** |
| **pooled (test-weighted)** | | | **0.1276** |
| **public (observed)** | 0.1886 | 0.0894 | **0.1390** → score **0.8610** |

Validation→public gap ≈ +0.011 combined (consistent across submissions).

## Targets

| rank | public score | combined error | reduction needed from 0.1390 |
|---|---|---|---|
| ~top 5 (oblivione 0.8919) | 0.892 | 0.108 | −0.031 (−22% rel) |
| **top 3 (Sabio 0.9105)** | 0.9105 | **0.0895** | **−0.0495 (−36% rel)** |
| top 1 (hashman 0.9345) | 0.9345 | 0.0655 | −0.0735 (−53% rel) |

## Where the gap is: it is Lingala. Full stop.

Pooled error decomposition (validation): lin contributes **0.0751** of the 0.1276 (59%),
sna 0.0352 (28%), lug 0.0172 (13%).

Counterfactuals (hold other languages fixed):

| scenario | pooled val | projected public |
|---|---|---|
| current | 0.1276 | 0.861 |
| lin → 0.15 | 0.1183 | ~0.871 |
| lin → 0.13 | 0.1095 | ~0.879 |
| lin → 0.10 | 0.0964 | ~0.893 (**top 5**) |
| **lin → sna level (0.086)** | **0.0902** | **~0.899–0.91 (top-3 territory)** |
| sna → 0.06 (already elite) | 0.1171 | ~0.872 (small: sna is near-ceiling) |
| lug → 0.09 (15% share) | 0.1239 | ~0.865 (small share) |

**Conclusion: no plan that does not materially fix Lingala can reach top 3.** sna (0.086) is
already at the top-cluster level; lug's share is too small to matter much.

## What kind of Lingala errors?

- lin CER 0.1285 is **2.6× sna's** (0.050) and **3.2× lug's** (0.040) → not a word-segmentation
  artifact; the acoustic model genuinely mis-hears Lingala at the character level.
- The LM lever already did its work: greedy 0.2647 → beam+LM 0.1711 (−0.094). The residual is
  acoustic + lexical (out-of-LM words), which is why further alpha/beta tuning plateaued
  (flat basin 0.171–0.172).
- Stub clips: ~70–76 unfixable ~1s organizer-defect clips are shared by all competitors; they
  bound everyone's ceiling equally (~0.5–1 point of pooled error).
- lug is the opposite profile (CER 0.040, WER 0.189): errors are word-boundary/morphology —
  LM/lexicon-improvable in principle, but only 15% of the test.

## Implication for strategy (see FINAL_TOP3_STRATEGY.md)

1. Every remaining lever must be judged by its **expected lin combined reduction**.
2. Cheap lin levers first: bigger Lingala LM corpus (lexical coverage), lin-specialist acoustic
   fine-tune from the champion, per-sample routing between decoders.
3. Realistic stacked best case (LM −0.01 lin, specialist −0.02..−0.04 lin, router −0.003 pooled)
   lands ~**0.87–0.885 public** — solid top 5–7. **Top 3 requires lin ≈ 0.09–0.10**, i.e. a
   ~40% lin error reduction — possible only if the specialist meaningfully works or the one
   full-schedule clean-retrain surprises. Be honest about that going in.
