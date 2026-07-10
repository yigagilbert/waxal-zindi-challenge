#!/usr/bin/env python3
"""Harvest transcripts (TEXT ONLY, no audio decode) from WAXAL + external open
datasets into per-language KenLM corpora, then optionally build the LMs.

Why text-only: our proven lever is decoding, and bigger n-gram LMs sharpen it.
The audio from these sources hurt Phase 1 in the clean-audio-v2 experiment, so we
take only the transcripts. Audio columns are dropped before iteration so nothing
is decoded and no audio is written to disk (only parquet shards are fetched).

Every source is license-tagged for the code-review disclosure in
docs/RULES_AND_DATA_USE.md. Output goes to a SEPARATE dir (default
data/lm_expanded) so the working data/lm LMs stay intact for A/B and rollback.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import read_csv_dicts  # noqa: E402
from waxal.text_normalization import POLICIES, normalize_text  # noqa: E402
from waxal.utils import ensure_dir, json_dump  # noqa: E402

TEXT_COLUMN_CANDIDATES = (
    "transcription",
    "raw_transcription",
    "sentence",
    "text",
    "Target",
    "transcript",
    "normalized_text",
)

# Text-only harvest registry. Audio columns are dropped, never decoded.
# `gated` sources need `huggingface-cli login` + accepting the dataset terms first.
# Config/column names for the gated sources are best-guesses; a mismatch skips
# that source with a warning rather than crashing (adjust once you can inspect it).
HF_SOURCES = [
    {"dataset": "google/fleurs", "config": "ln_cd", "language": "lin", "license": "CC-BY-4.0"},
    {"dataset": "google/fleurs", "config": "lg_ug", "language": "lug", "license": "CC-BY-4.0"},
    {"dataset": "google/fleurs", "config": "sn_zw", "language": "sna", "license": "CC-BY-4.0"},
    {"dataset": "Sunbird/salt", "config": "multispeaker-lug", "language": "lug",
     "license": "CC-BY-SA-4.0", "splits": ["train", "dev", "test"]},
    {"dataset": "DigitalUmuganda/Afrivoice", "config": "Lingala", "language": "lin",
     "license": "CC-BY-4.0", "gated": True,
     "language_filter": {"column": "language", "values": ["lin", "Lingala", "lingala", "ln"]}},
    {"dataset": "DigitalUmuganda/Afrivoice", "config": "Shona", "language": "sna",
     "license": "CC-BY-4.0", "gated": True,
     "language_filter": {"column": "language", "values": ["sna", "Shona", "shona", "sn"]}},
    {"dataset": "yigagilbert/luganda-speech-cv-yogera", "config": "makerere-yogera-lug",
     "language": "lug", "license": "CC-BY-SA-4.0", "gated": True},
    {"dataset": "yigagilbert/luganda-speech-cv-yogera", "config": "common-voice-sample-packed-lug",
     "language": "lug", "license": "CC0-1.0", "gated": True},
]

DEFAULT_SPLITS = ["train", "validation", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/lm_expanded"))
    parser.add_argument("--languages", nargs="*", default=["lin", "lug", "sna"])
    parser.add_argument(
        "--csv",
        type=Path,
        action="append",
        default=None,
        help="Local text CSV to include (WAXAL anchor etc.). Repeat. Defaults to the generalization-mix train.csv.",
    )
    parser.add_argument("--language-column", default="language")
    parser.add_argument("--waxal-repeat", type=int, default=1, help="Repeat WAXAL/local-CSV lines N times to keep the LM domain-anchored.")
    parser.add_argument("--skip-hf", action="store_true", help="Only use local CSVs, no Hugging Face pulls.")
    parser.add_argument("--skip-source", action="append", default=[], help="dataset[:config] to skip. Repeat.")
    parser.add_argument("--max-lines-per-source", type=int, default=None, help="Cap lines per (source, split) so one corpus doesn't dominate.")
    parser.add_argument("--dedup", action="store_true", help="Drop exact-duplicate lines within a language.")
    parser.add_argument("--min-chars", type=int, default=2)
    parser.add_argument("--normalization", choices=POLICIES, default="language_safe")
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--corpus-only", action="store_true", help="Write corpora but do not build KenLM binaries.")
    parser.add_argument("--no-discount-fallback", action="store_true", help="Omit lmplz --discount_fallback (only for very large corpora).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def pick_text(row: dict, extra_first: tuple[str, ...] = ()) -> str:
    for key in (*extra_first, *TEXT_COLUMN_CANDIDATES):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def matches_language_filter(row: dict, language_filter: dict | None) -> bool:
    if not language_filter:
        return True
    column = language_filter.get("column")
    if column not in row:
        return True  # column absent -> can't filter, keep
    return row.get(column) in set(language_filter.get("values", []))


def harvest_local_csvs(csv_paths: list[Path], language_column: str, languages: set[str], normalization: str, min_chars: int, repeat: int):
    by_language: dict[str, list[str]] = defaultdict(list)
    provenance: list[dict] = []
    for path in csv_paths:
        columns, rows, bad = read_csv_dicts(path)
        if bad:
            print(f"WARNING: {path} has malformed rows; using the clean ones only.")
        counts: Counter[str] = Counter()
        for row in rows:
            language = row.get(language_column) or ""
            if languages and language not in languages:
                continue
            text = pick_text(row)
            norm = normalize_text(text, normalization)
            if len(norm) >= min_chars:
                for _ in range(max(1, repeat)):
                    by_language[language].append(norm)
                counts[language] += 1
        provenance.append({"source": str(path), "kind": "local_csv", "license": "competition_provided_or_local",
                           "lines_by_language": dict(sorted(counts.items())), "repeat": repeat})
        print(f"Local {path}: { {k: v for k, v in sorted(counts.items())} } (x{repeat})")
    return by_language, provenance


def harvest_hf_source(src: dict, languages: set[str], args: argparse.Namespace):
    from datasets import load_dataset

    lang = src["language"]
    if languages and lang not in languages:
        return [], None
    lines: list[str] = []
    total = 0
    for split in src.get("splits", DEFAULT_SPLITS):
        try:
            ds = load_dataset(src["dataset"], src.get("config"), split=split, streaming=True)
        except Exception as exc:
            print(f"  skip {src['dataset']}:{src.get('config')}/{split}: {type(exc).__name__}: {exc}")
            continue
        try:
            cols = list(ds.column_names) if ds.column_names else []
        except Exception:
            cols = []
        drop = [c for c in cols if "audio" in c.lower()]  # never decode audio
        if drop:
            try:
                ds = ds.remove_columns(drop)
            except Exception:
                pass
        n = 0
        try:
            for row in ds:
                if not matches_language_filter(row, src.get("language_filter")):
                    continue
                text = pick_text(row)
                norm = normalize_text(text, args.normalization)
                if len(norm) >= args.min_chars:
                    lines.append(norm)
                    n += 1
                if args.max_lines_per_source and n >= args.max_lines_per_source:
                    break
        except Exception as exc:
            print(f"  partial {src['dataset']}:{src.get('config')}/{split}: {type(exc).__name__}: {exc} (kept {n})")
        total += n
        print(f"  {src['dataset']}:{src.get('config')}/{split}: +{n} lines")
    provenance = {
        "source": src["dataset"], "config": src.get("config"), "kind": "hf_stream_text_only",
        "language": lang, "license": src["license"], "gated": bool(src.get("gated")), "lines": total,
    }
    return lines, provenance


def build_kenlm(corpus: Path, language: str, args: argparse.Namespace) -> None:
    lmplz = shutil.which("lmplz")
    build_binary = shutil.which("build_binary")
    arpa = args.output_dir / f"{language}_{args.order}gram.arpa"
    binary = args.output_dir / f"{language}_{args.order}gram.binary"
    if args.corpus_only or not lmplz or not build_binary:
        cmd = f"lmplz -o {args.order}{'' if args.no_discount_fallback else ' --discount_fallback'} --text {corpus} --arpa {arpa}"
        print(f"  (build manually) {cmd} ; build_binary {arpa} {binary}")
        return
    if (arpa.exists() or binary.exists()) and not args.overwrite:
        raise FileExistsError(f"{binary} exists. Pass --overwrite to replace.")
    lmplz_cmd = [lmplz, "-o", str(args.order), "--text", str(corpus), "--arpa", str(arpa)]
    if not args.no_discount_fallback:
        lmplz_cmd.insert(3, "--discount_fallback")
    subprocess.run(lmplz_cmd, check=True)
    subprocess.run([build_binary, str(arpa), str(binary)], check=True)
    print(f"Built {binary}")


def main() -> None:
    args = parse_args()
    languages = set(args.languages)
    csv_paths = args.csv if args.csv else [Path("data/processed_generalization_mix/train.csv")]
    csv_paths = [p for p in csv_paths if p.exists()]
    if not csv_paths:
        print("WARNING: no local WAXAL CSV found; the LM will lack the in-domain anchor.")

    by_language, provenance = harvest_local_csvs(
        csv_paths, args.language_column, languages, args.normalization, args.min_chars, args.waxal_repeat
    )

    skip = set(args.skip_source)
    if not args.skip_hf:
        for src in HF_SOURCES:
            key = f"{src['dataset']}:{src.get('config')}"
            if key in skip or src["dataset"] in skip:
                print(f"Skipping {key} (requested)")
                continue
            print(f"Harvesting {key} ({src['language']}, {src['license']}{', GATED' if src.get('gated') else ''}) ...")
            lines, prov = harvest_hf_source(src, languages, args)
            if lines:
                by_language[src["language"]].extend(lines)
            if prov:
                provenance.append(prov)

    ensure_dir(args.output_dir)
    summary: dict = {"output_dir": str(args.output_dir), "order": args.order, "normalization": args.normalization,
                     "provenance": provenance, "corpora": {}}
    for language in sorted(by_language):
        lines = by_language[language]
        if args.dedup:
            seen: set[str] = set()
            deduped = []
            for line in lines:
                if line not in seen:
                    seen.add(line)
                    deduped.append(line)
            lines = deduped
        corpus = args.output_dir / f"{language}.txt"
        with corpus.open("w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        summary["corpora"][language] = {"lines": len(lines), "corpus": str(corpus)}
        print(f"{language}: {len(lines)} lines -> {corpus}")
        build_kenlm(corpus, language, args)

    report = args.report or (args.output_dir / "lm_text_provenance.json")
    json_dump(summary, report)
    print(f"\nProvenance/report: {report}")
    print("Disclose these sources+licenses in docs/RULES_AND_DATA_USE.md before submitting a solution built on them.")


if __name__ == "__main__":
    main()
