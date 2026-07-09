#!/usr/bin/env python3
"""Export per-language LM corpora and optionally build KenLM decoders."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import read_csv_dicts  # noqa: E402
from waxal.text_normalization import POLICIES, normalize_text  # noqa: E402
from waxal.utils import ensure_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        action="append",
        required=True,
        help="CSV containing transcripts/text. Repeat for WAXAL and rule-safe external text.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/lm"))
    parser.add_argument("--language-column", default="language")
    parser.add_argument(
        "--text-column",
        action="append",
        default=["transcription", "Target", "text", "sentence"],
        help="Candidate text column name. Repeat to add more candidates.",
    )
    parser.add_argument("--normalization", choices=POLICIES, default="language_safe")
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--min-chars", type=int, default=2)
    parser.add_argument("--corpus-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_text_column(columns: list[str], candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"No text column found. Tried {candidates}; got columns {columns}")


def infer_language(row: dict[str, str], language_column: str) -> str:
    if language_column in row and row[language_column]:
        return row[language_column]
    example_id = row.get("ID") or row.get("id") or ""
    if "_" in example_id:
        return example_id.split("_", 1)[0]
    return "unknown"


def export_corpora(args: argparse.Namespace) -> dict[str, Path]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for path in args.csv:
        columns, rows, bad = read_csv_dicts(path)
        if bad:
            raise ValueError(f"{path} has malformed rows: {bad[:3]}")
        text_column = find_text_column(columns, args.text_column)
        for row in rows:
            lang = infer_language(row, args.language_column)
            text = normalize_text(row.get(text_column, ""), args.normalization)
            if len(text) >= args.min_chars:
                grouped[lang].append(text)

    ensure_dir(args.output_dir)
    outputs = {}
    for lang, texts in sorted(grouped.items()):
        out = args.output_dir / f"{lang}.txt"
        if out.exists() and not args.overwrite:
            raise FileExistsError(f"{out} exists. Pass --overwrite to replace it.")
        with out.open("w", encoding="utf-8") as f:
            for text in texts:
                f.write(text)
                f.write("\n")
        outputs[lang] = out
        print(f"Wrote {len(texts)} lines for {lang}: {out}")
    return outputs


def build_kenlm(corpora: dict[str, Path], args: argparse.Namespace) -> None:
    lmplz = shutil.which("lmplz")
    build_binary = shutil.which("build_binary")
    if args.corpus_only or not lmplz or not build_binary:
        print("\nKenLM binaries not found or --corpus-only set. Build manually with:")
        for lang, corpus in sorted(corpora.items()):
            arpa = args.output_dir / f"{lang}_{args.order}gram.arpa"
            binary = args.output_dir / f"{lang}_{args.order}gram.binary"
            print(f"lmplz -o {args.order} --text {corpus} --arpa {arpa}")
            print(f"build_binary {arpa} {binary}")
        return

    for lang, corpus in sorted(corpora.items()):
        arpa = args.output_dir / f"{lang}_{args.order}gram.arpa"
        binary = args.output_dir / f"{lang}_{args.order}gram.binary"
        if (arpa.exists() or binary.exists()) and not args.overwrite:
            raise FileExistsError(f"{arpa} or {binary} exists. Pass --overwrite to replace.")
        subprocess.run([lmplz, "-o", str(args.order), "--text", str(corpus), "--arpa", str(arpa)], check=True)
        subprocess.run([build_binary, str(arpa), str(binary)], check=True)
        print(f"Wrote KenLM binary for {lang}: {binary}")


def main() -> None:
    args = parse_args()
    corpora = export_corpora(args)
    build_kenlm(corpora, args)


if __name__ == "__main__":
    main()
