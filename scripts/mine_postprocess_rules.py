#!/usr/bin/env python3
"""Mine per-language hypothesis-correction rules from a reference-scored bench.

Splits the bench 50/50 by ID hash: rules are mined on the MINE half (aligned
ref<->hyp word substitutions that are frequent AND consistent — low reverse-
direction evidence), then applied to the EVAL half and scored before/after.
Only ship rules that improve the held-out half. Guards against learning the
reference set's internal inconsistencies (e.g. Lusoga l/r, nh/n go both ways).

Usage (bench):
  python scripts/mine_postprocess_rules.py \
    --predictions outputs/predictions/salt_val_base.csv \
    --references data/phase2_train/validation.csv \
    --rules-out outputs/analysis/pp_rules.json

Apply surviving rules to test predictions (language from the routing table):
  python scripts/mine_postprocess_rules.py --apply \
    --rules outputs/analysis/pp_rules.json \
    --predictions outputs/predictions/phase2_salt_myxadapter_spliced.csv \
    --routing outputs/analysis/phase2_language_clusters.csv \
    --output outputs/predictions/phase2_pp.csv
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    rw, hw = ref.split(), hyp.split()
    sm = difflib.SequenceMatcher(None, rw, hw)
    errs = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")
    return errs, len(rw)


def char_errors(ref: str, hyp: str) -> tuple[int, int]:
    sm = difflib.SequenceMatcher(None, ref, hyp)
    errs = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")
    return errs, len(ref)


def apply_rules(text: str, rules: list[tuple[str, str]]) -> str:
    # Longest source phrase first; word-boundary exact match, case-insensitive,
    # preserving surrounding punctuation via token-stream matching.
    tokens = text.split()
    low = [t.lower().strip(".,!?;:") for t in tokens]
    for src, dst in rules:
        src_words = src.split()
        n = len(src_words)
        i = 0
        out_tokens, out_low = [], []
        while i < len(tokens):
            if low[i : i + n] == src_words:
                lead = re.match(r"^\W*", tokens[i]).group(0)
                trail = re.search(r"\W*$", tokens[i + n - 1]).group(0)
                replacement = dst.split()
                # keep capitalization of first source token
                if tokens[i][:1].isupper() and replacement:
                    replacement = [replacement[0].capitalize(), *replacement[1:]]
                if replacement:
                    replacement[0] = lead + replacement[0]
                    replacement[-1] = replacement[-1] + trail
                out_tokens.extend(replacement)
                out_low.extend(w.lower() for w in replacement)
                i += n
            else:
                out_tokens.append(tokens[i])
                out_low.append(low[i])
                i += 1
        tokens, low = out_tokens, out_low
    return " ".join(tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, default=None)
    parser.add_argument("--routing", type=Path, default=None, help="ID,language table (apply mode).")
    parser.add_argument("--rules", type=Path, default=None, help="Existing rules JSON (apply mode).")
    parser.add_argument("--rules-out", type=Path, default=Path("outputs/analysis/pp_rules.json"))
    parser.add_argument("--output", type=Path, default=None, help="Rewritten predictions (apply mode).")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--max-reverse-ratio", type=float, default=0.34)
    args = parser.parse_args()

    preds = {r["ID"]: r["Target"] for r in csv.DictReader(args.predictions.open(encoding="utf-8-sig"))}

    if args.apply:
        rules_by_lang = {k: [tuple(r) for r in v] for k, v in json.loads(args.rules.read_text()).items()}
        routing = {r["ID"]: r["language"] for r in csv.DictReader(args.routing.open(encoding="utf-8-sig"))}
        changed = 0
        rows = []
        for example_id, text in preds.items():
            lang = routing.get(example_id, "unk")
            new = apply_rules(text, rules_by_lang.get(lang, []))
            changed += new != text
            rows.append({"ID": example_id, "Target": new})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["ID", "Target"])
            w.writeheader()
            w.writerows(rows)
        print(f"Applied rules; {changed}/{len(rows)} rows changed -> {args.output}")
        return

    refs = {
        r["ID"]: (r["language"], r["Target"])
        for r in csv.DictReader(args.references.open(encoding="utf-8-sig"))
        if r["ID"] in preds
    }
    mine_ids = {k for i, k in enumerate(sorted(refs)) if i % 2 == 0}
    eval_ids = set(refs) - mine_ids
    print(f"bench: {len(refs)} clips -> mine {len(mine_ids)} / eval {len(eval_ids)}")

    # ---- mine on the MINE half -------------------------------------------------
    fwd = defaultdict(Counter)   # (lang)[(hyp_phrase, ref_phrase)] -> count
    rev = defaultdict(Counter)
    for k in mine_ids:
        lang, ref = refs[k]
        rw = [w.strip(".,!?;:") for w in ref.lower().split()]
        hw = [w.strip(".,!?;:") for w in preds[k].lower().split()]
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, rw, hw).get_opcodes():
            if tag == "replace" and 0 < i2 - i1 <= 2 and 0 < j2 - j1 <= 2:
                r_p, h_p = " ".join(rw[i1:i2]), " ".join(hw[j1:j2])
                fwd[lang][(h_p, r_p)] += 1
                rev[lang][(r_p, h_p)] += 1

    rules_by_lang: dict[str, list[tuple[str, str]]] = {}
    for lang, counter in fwd.items():
        kept = []
        for (h_p, r_p), c in counter.most_common():
            if c < args.min_count:
                break
            reverse = counter.get((r_p, h_p), 0)
            if reverse > args.max_reverse_ratio * c:
                continue  # reference-inconsistent pair (l/r, nh/n, ...)
            if len(h_p) < 2 or h_p == r_p:
                continue
            kept.append((h_p, r_p, c, reverse))
        kept.sort(key=lambda t: -len(t[0]))  # longest source first
        rules_by_lang[lang] = [(h, r) for h, r, _, _ in kept]
        print(f"\n{lang}: {len(kept)} rules")
        for h, r, c, rv in kept[:15]:
            print(f"   '{h}' -> '{r}'   (n={c}, reverse={rv})")

    # ---- validate on the EVAL half ----------------------------------------------
    print("\nHeld-out validation (raw text):")
    total_gain = {}
    for lang in rules_by_lang:
        we0 = we1 = wr = ce0 = ce1 = cr = 0
        for k in eval_ids:
            lg, ref = refs[k]
            if lg != lang:
                continue
            hyp0 = preds[k]
            hyp1 = apply_rules(hyp0, rules_by_lang[lang])
            e0, n = word_errors(ref, hyp0); e1, _ = word_errors(ref, hyp1)
            we0 += e0; we1 += e1; wr += n
            c0, m = char_errors(ref, hyp0); c1, _ = char_errors(ref, hyp1)
            ce0 += c0; ce1 += c1; cr += m
        if wr:
            print(f"  {lang}: WER {we0/wr:.4f} -> {we1/wr:.4f}  CER {ce0/cr:.4f} -> {ce1/cr:.4f}  "
                  f"({'KEEP' if (we1/wr + ce1/cr) < (we0/wr + ce0/cr) else 'DROP'})")
            total_gain[lang] = (we1/wr + ce1/cr) - (we0/wr + ce0/cr)

    shipped = {lang: rules for lang, rules in rules_by_lang.items() if total_gain.get(lang, 1) < 0}
    args.rules_out.parent.mkdir(parents=True, exist_ok=True)
    args.rules_out.write_text(json.dumps(shipped, indent=1, ensure_ascii=False))
    print(f"\nShipped rules for {sorted(shipped)} (held-out winners only) -> {args.rules_out}")


if __name__ == "__main__":
    main()
