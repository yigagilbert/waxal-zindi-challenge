# Champion Checkpoint Report

Anchor record for the current best model. **Do not overwrite these artifacts.** Every
continuation/fallback experiment is compared against the numbers here.

_Last updated: 2026-07-20._

## Identity

| Field | Value |
|---|---|
| Checkpoint (origin) | `checkpoints/xlsr_300m_generalization_mix/checkpoint-24000` (original training box) |
| Checkpoint (current box) | `champion/checkpoint-24000` — pulled from HF `yigagilbert/waxal-xlsr300m-champion` (`repo_type=model`) |
| Base architecture | `facebook/wav2vec2-xls-r-300m` — Wav2Vec2 XLS-R 300M, **CTC** head |
| Training config | [configs/xlsr_300m_generalization_mix.yaml](../configs/xlsr_300m_generalization_mix.yaml) |
| Tokenizer / vocab | CTC char vocab from the champion processor (bundled in the checkpoint dir) |
| Normalization | `language_safe` (train + eval) |

> Confirm the exact subdir on the box with `ls champion/`. If it is not `checkpoint-24000`,
> update `model.name` in the continuation configs accordingly.

## Training recipe (champion)

- Init: base `facebook/wav2vec2-xls-r-300m`, `freeze_feature_encoder: true`
- Data: `data/processed_generalization_mix` = WAXAL clean train + FLEURS + SALT, **language-balanced sampling**
- LR `3.0e-4`, warmup 2400, **24,000 steps** (~3.65 epochs), effective batch 16 (2 × grad-accum 8)
- fp16, gradient checkpointing, `group_by_length`, `metric_for_best_model: eval_cer`
- The eval curve plateaued over the last ~5k steps → checkpoint-24000 is the last **and** best.

## Validation metrics (in-domain WAXAL validation, 4,235)

Basis for all comparisons is **`data/processed_generalization_mix` validation**, decoded the
same way the leaderboard submission was (beam+LM). Test audio is raw/untrimmed, so validation
is always evaluated on **raw** (untrimmed) audio — never the cleaned/trimmed validation.

Champion + **expanded-LM** beam decode (from `outputs/analysis/kenlm_alpha_beta_sweep_expanded.json`
and `kenlm_sweep_lin_refine.json`):

| lang | greedy combined | best beam+LM combined | best (α, β) |
|------|-----------------|-----------------------|-------------|
| lin  | 0.2647          | **0.1711**            | (0.9, 0.5) @ beam 400 |
| lug  | 0.1259          | **0.1148**            | (0.4, −0.5) |
| sna  | 0.1817          | **0.0857**            | (0.7, −0.5) |

Pooled (test-weighted) projected combined ≈ **0.135** → projected public ≈ 0.865.

## Leaderboard submissions (public)

| Submission | Decode | Public score | WER | CER |
|---|---|---|---|---|
| `champ_expanded_lm.csv` | beam 100, lin(0.7,0.5)/lug(0.4,−0.5)/sna(0.7,−0.5) | 0.858467 | 0.1928 | 0.0903 |
| **`champ_expanded_lm_beam400.csv`** (selected) | beam 400, lin(0.9,0.5)/lug(0.4,−0.5)/sna(0.7,−0.5) | **0.861024** | 0.1886 | 0.0894 |

**Currently selected best = 0.861024.** Keep it selected unless a new candidate clearly beats it.

## Decoding strategy for the 0.861 submission

- **LM: yes** — KenLM 5-gram, `data/lm_expanded` (champion corpora **+ Afrivoice** text; sna corpus doubled to 33,749 lines)
- **Beam: yes** — pyctcdecode, beam width 400, unigram lexicon from `data/lm_expanded/<lang>.txt`
- Per-language params: lin α=0.9 β=0.5 · lug α=0.4 β=−0.5 · sna α=0.7 β=−0.5
- **Postprocessing: yes** — `postprocess_predictions.py` single-dot strip (min-run 3)
- Merge: `merge_predictions.py` (per-language → SampleSubmission order/coverage)
- `make_submission.py --empty-target "."`
- **Routing/fallback: NO** — pure XLS-R + per-language LM. (The old `alvin_lingala` routing model / manifest was lost with the prior box; not part of the 0.861 result.)

## Reproduce the champion submission (from `champion/checkpoint-24000`)

See the exact 3× `run_xlsr_inference.py --decoder-mode beam_lm ... --beam-width 400` +
merge + postprocess + make_submission chain in
[CHAMPION_CONTINUATION_ANALYSIS.md](CHAMPION_CONTINUATION_ANALYSIS.md) (same chain, swap the
`--checkpoint`).

## Do-not-touch list

- `champion/` (pulled checkpoint) — read-only source for continuation `model.name`
- HF `yigagilbert/waxal-xlsr300m-champion` — do not overwrite/force-push
- `outputs/submissions/champ_expanded_lm_beam400.csv` — the selected 0.861 submission
- `data/lm_expanded/` — the expanded LMs that produced the win
- `outputs/analysis/kenlm_alpha_beta_sweep_expanded.json`, `kenlm_sweep_lin_refine.json`
