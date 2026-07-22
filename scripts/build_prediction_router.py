#!/usr/bin/env python3
"""Model-zoo oracle + per-sample prediction router.

Two modes, both CPU-only, both leakage-safe:

--mode oracle (references REQUIRED, validation only)
    For N candidate prediction CSVs, compute each candidate's combined error, the per-sample
    ORACLE (pick the best candidate per sample by CER) and per-language oracle. The oracle is
    an upper bound on any router/ensemble built from these candidates: if
    (best_single - oracle) is small, routing cannot help much and a stronger model is needed.

--mode route
    Choose one candidate per sample using validation-tuned RULES over test-safe features only
    (no references needed at apply time): degenerate-output flags (dot-only / very-short /
    repeated n-gram), length, and optional per-language KenLM log-prob per char. Thresholds are
    tuned on validation with K-fold cross-validation to avoid overfitting; --apply then writes
    the routed CSV for any split (e.g. test) using the tuned rule.

Rules (deliberately simple, in priority order; candidate order = preference order, first is primary):
  1. If the primary's output is degenerate (dot-only/empty/very-short) and another candidate's
     is not, take the first non-degenerate candidate.
  2. If the primary shows a repeated n-gram loop and another candidate does not, switch.
  3. (optional, needs --kenlm-dir) If a candidate's LM log-prob/char beats the primary's by more
     than a tuned margin, switch. Margin tuned per-language by CV grid.

Usage (oracle):
  python scripts/build_prediction_router.py --mode oracle \
    --predictions champion_beam=outputs/predictions/nometa_validation_expanded.csv \
    --predictions champion_greedy=outputs/predictions/champ_greedy_validation.csv \
    --references data/processed_generalization_mix/validation.csv \
    --output outputs/analysis/model_zoo_oracle_validation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import id_language, read_prediction_csv, references_from_validation_csv  # noqa: E402
from waxal.scoring import edit_distance  # noqa: E402
from waxal.text_normalization import normalize_text  # noqa: E402
from waxal.utils import json_dump  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["oracle", "route"], required=True)
    parser.add_argument(
        "--predictions", action="append", required=True,
        help="name=path.csv per candidate. FIRST is the primary/preferred candidate. Repeat.",
    )
    parser.add_argument("--references", type=Path, default=None, help="validation.csv with ID/Target (required for oracle + tuning).")
    parser.add_argument("--kenlm-dir", type=Path, default=None, help="Enable LM-margin rule: dir with <lang>_<order>gram.binary.")
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--normalization", default="language_safe")
    parser.add_argument("--min-chars", type=int, default=3, help="Outputs shorter than this count as degenerate.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--margins", type=float, nargs="*", default=[0.0, 0.25, 0.5, 1.0, 2.0], help="LM logprob/char margins to grid-search (route mode).")
    parser.add_argument("--apply", type=Path, default=None, help="route mode: write the routed prediction CSV here.")
    parser.add_argument("--output", type=Path, required=True, help="Report JSON path.")
    return parser.parse_args()


def load_candidates(specs: list[str]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--predictions must be name=path, got {spec!r}")
        out[name] = {row["ID"]: row.get("Target", row.get("prediction", "")) for row in read_prediction_csv(Path(path))}
    return out


def cer(ref: str, hyp: str) -> float:
    ref_c, hyp_c = list(ref), list(hyp)
    if not ref_c:
        return 0.0 if not hyp_c else 1.0
    return edit_distance(ref_c, hyp_c) / len(ref_c)


def wer(ref: str, hyp: str) -> float:
    ref_w, hyp_w = ref.split(), hyp.split()
    if not ref_w:
        return 0.0 if not hyp_w else 1.0
    return edit_distance(ref_w, hyp_w) / len(ref_w)


def combined(ref: str, hyp: str) -> float:
    return 0.5 * wer(ref, hyp) + 0.5 * cer(ref, hyp)


def is_degenerate(text: str, min_chars: int) -> bool:
    stripped = text.replace(".", "").replace(" ", "")
    return len(stripped) < min_chars


def has_repeat_ngram(text: str, n: int = 3, times: int = 3) -> bool:
    words = text.split()
    if len(words) < n * times:
        return False
    grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts: dict[str, int] = defaultdict(int)
    for g in grams:
        counts[g] += 1
        if counts[g] >= times:
            return True
    return False


def load_lms(kenlm_dir: Path | None, order: int, languages: list[str]):
    if kenlm_dir is None:
        return {}
    try:
        import kenlm
    except ImportError:
        print("WARNING: kenlm not importable; LM-margin rule disabled.")
        return {}
    lms = {}
    for lang in languages:
        path = kenlm_dir / f"{lang}_{order}gram.binary"
        if path.exists():
            lms[lang] = kenlm.Model(str(path))
    return lms


def lm_logprob_per_char(lms, lang: str, text: str) -> float | None:
    model = lms.get(lang)
    if model is None or not text.strip():
        return None
    return model.score(text, bos=True, eos=True) / max(len(text), 1)


def route_one(sample: dict, names: list[str], lms, margin: float, min_chars: int) -> str:
    """Return chosen candidate name for one sample. sample: {name: text, '_lang': lang}."""
    primary = names[0]
    lang = sample["_lang"]
    # Rule 1/2: degenerate or looping primary -> first healthy alternative
    prim_text = sample[primary]
    prim_bad = is_degenerate(prim_text, min_chars) or has_repeat_ngram(prim_text)
    if prim_bad:
        for name in names[1:]:
            alt = sample[name]
            if not is_degenerate(alt, min_chars) and not has_repeat_ngram(alt):
                return name
    # Rule 3: LM margin
    if lms and margin is not None:
        prim_lp = lm_logprob_per_char(lms, lang, prim_text)
        best_name, best_lp = primary, prim_lp
        for name in names[1:]:
            lp = lm_logprob_per_char(lms, lang, sample[name])
            if lp is not None and (best_lp is None or lp > best_lp + margin):
                best_name, best_lp = name, lp
        return best_name
    return primary


def main() -> None:
    args = parse_args()
    candidates = load_candidates(args.predictions)
    names = [spec.partition("=")[0] for spec in args.predictions]
    ids = sorted(set.intersection(*(set(c) for c in candidates.values())))
    print(f"{len(ids)} common IDs across {len(names)} candidates: {names}")

    refs: dict[str, str] = {}
    if args.references is not None:
        id_set = set(ids)
        refs = {
            row["ID"]: normalize_text(row["Target"], args.normalization)
            for row in references_from_validation_csv(args.references)
            if row["ID"] in id_set
        }
        ids = [i for i in ids if i in refs]
        print(f"{len(ids)} IDs with references")

    samples = []
    for row_id in ids:
        s = {"_id": row_id, "_lang": id_language(row_id)}
        for name in names:
            s[name] = normalize_text(candidates[name][row_id], args.normalization)
        samples.append(s)

    languages = sorted({s["_lang"] for s in samples})
    report: dict = {"mode": args.mode, "candidates": names, "num_samples": len(samples), "languages": languages}

    if args.mode == "oracle":
        if not refs:
            raise SystemExit("--references is required for oracle mode")
        per_cand = {n: [] for n in names}
        oracle_scores, oracle_pick = [], defaultdict(int)
        by_lang: dict[str, dict] = {lang: {"oracle": [], **{n: [] for n in names}} for lang in languages}
        for s in samples:
            ref = refs[s["_id"]]
            scores = {n: combined(ref, s[n]) for n in names}
            best = min(scores, key=scores.get)
            oracle_pick[best] += 1
            oracle_scores.append(scores[best])
            by_lang[s["_lang"]]["oracle"].append(scores[best])
            for n in names:
                per_cand[n].append(scores[n])
                by_lang[s["_lang"]][n].append(scores[n])
        mean = lambda xs: sum(xs) / max(len(xs), 1)
        report["overall"] = {
            **{n: round(mean(per_cand[n]), 4) for n in names},
            "oracle": round(mean(oracle_scores), 4),
            "oracle_pick_counts": dict(oracle_pick),
        }
        best_single = min(mean(per_cand[n]) for n in names)
        report["overall"]["best_single"] = round(best_single, 4)
        report["overall"]["oracle_upside"] = round(best_single - mean(oracle_scores), 4)
        report["by_language"] = {
            lang: {**{n: round(mean(d[n]), 4) for n in names}, "oracle": round(mean(d["oracle"]), 4)}
            for lang, d in by_lang.items()
        }
        print(json.dumps(report["overall"], indent=2))
        print(json.dumps(report["by_language"], indent=2))

    else:  # route
        lms = load_lms(args.kenlm_dir, args.order, languages)
        chosen_margin: dict[str, float] = {}
        if refs:
            # K-fold CV per language over margins; pick the margin whose held-out mean wins.
            for lang in languages:
                lang_samples = [s for s in samples if s["_lang"] == lang]
                fold_size = max(len(lang_samples) // args.folds, 1)
                margin_scores = {}
                for margin in args.margins:
                    held = []
                    for k in range(args.folds):
                        fold = lang_samples[k * fold_size : (k + 1) * fold_size]
                        for s in fold:
                            pick = route_one(s, names, lms, margin, args.min_chars)
                            held.append(combined(refs[s["_id"]], s[pick]))
                    margin_scores[margin] = sum(held) / max(len(held), 1)
                chosen_margin[lang] = min(margin_scores, key=margin_scores.get)
                primary_score = sum(combined(refs[s["_id"]], s[names[0]]) for s in lang_samples) / max(len(lang_samples), 1)
                report.setdefault("cv", {})[lang] = {
                    "margins": {str(m): round(v, 4) for m, v in margin_scores.items()},
                    "chosen_margin": chosen_margin[lang],
                    "primary_alone": round(primary_score, 4),
                }
        else:
            chosen_margin = {lang: args.margins[0] for lang in languages}
            print("No references: applying rules with the first margin (no tuning).")
        report["chosen_margin"] = chosen_margin

        picks = {s["_id"]: route_one(s, names, lms, chosen_margin.get(s["_lang"], 0.0), args.min_chars) for s in samples}
        report["route_pick_counts"] = dict(defaultdict(int, {n: list(picks.values()).count(n) for n in names}))
        if refs:
            routed = [combined(refs[s["_id"]], s[picks[s["_id"]]]) for s in samples]
            primary = [combined(refs[s["_id"]], s[names[0]]) for s in samples]
            mean = lambda xs: sum(xs) / max(len(xs), 1)
            report["routed_combined"] = round(mean(routed), 4)
            report["primary_combined"] = round(mean(primary), 4)
            report["gain"] = round(mean(primary) - mean(routed), 4)
            print(f"primary={report['primary_combined']} routed={report['routed_combined']} gain={report['gain']}")
        if args.apply is not None:
            import csv

            with args.apply.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Target"])
                for s in samples:
                    writer.writerow([s["_id"], candidates[picks[s["_id"]]][s["_id"]]])
            print(f"Wrote routed predictions to {args.apply}")

    json_dump(report, args.output)
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
