#!/usr/bin/env python3
"""Run XLS-R / wav2vec2 CTC inference on a prepared WAXAL split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import TARGET_LANGUAGES, write_csv_rows  # noqa: E402
from waxal.text_normalization import normalize_text  # noqa: E402
from waxal.utils import clean_name  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, default=None)
    parser.add_argument(
        "--hf-processor",
        action="store_true",
        help="Load AutoProcessor from the checkpoint (HF repo id or local dir) instead of building "
        "a Wav2Vec2Processor from vocab.json. Required for non-Wav2Vec2 CTC models "
        "(e.g. w2v-BERT 2.0, which consumes input_features rather than raw input_values).",
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--languages", nargs="*", default=None, choices=list(TARGET_LANGUAGES))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--decoder-mode", choices=["greedy", "beam", "beam_lm"], default="greedy")
    parser.add_argument("--kenlm-model", type=Path, default=None)
    parser.add_argument(
        "--unigrams-file",
        type=Path,
        default=None,
        help="Word list for pyctcdecode lexicon scoring. Auto-derived from the KenLM corpus (e.g. data/lm/lin.txt) when omitted.",
    )
    parser.add_argument("--beam-width", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.5)
    return parser.parse_args()


def load_split(dataset_dir: Path, split: str):
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError("datasets is required. Install dependencies with `uv sync`.") from exc

    dataset_path = dataset_dir / "hf_dataset"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {dataset_path}. "
            "Run scripts/prepare_dataset.py first."
        )
    dataset_dict = load_from_disk(dataset_path)
    if split not in dataset_dict:
        raise ValueError(f"Split {split!r} not found in {dataset_path}. Available: {list(dataset_dict.keys())}")
    return dataset_dict[split]


def resolve_vocab_path(checkpoint: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Explicit vocab path does not exist: {explicit}")
        return explicit

    candidates = [
        checkpoint / "vocab.json",
        checkpoint.parent / "vocab.json",
        checkpoint.parent.parent / "vocab.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find vocab.json. Pass --vocab-path explicitly. "
        f"Checked: {[str(path) for path in candidates]}"
    )


def build_processor(vocab_path: Path):
    try:
        from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
    except ImportError as exc:
        raise RuntimeError("transformers is required. Install dependencies with `uv sync`.") from exc

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16_000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def clean_ctc_prediction(text: str, word_delimiter: str | None) -> str:
    delimiter = word_delimiter or "|"
    text = str(text).replace(delimiter, " ")
    return normalize_text(text, "raw")


def build_ctc_labels(tokenizer) -> list[str]:
    """Build pyctcdecode labels from a CTC tokenizer vocabulary.

    pyctcdecode requires unique labels: only the pad/blank slot may be empty.
    [UNK] is normalized to pyctcdecode's UNK token by the library itself;
    <s>/</s> stay literal and are stripped after decoding.
    """
    vocab = tokenizer.get_vocab()
    labels = [""] * len(vocab)
    word_delimiter = tokenizer.word_delimiter_token
    for token, idx in vocab.items():
        if token == word_delimiter:
            labels[idx] = " "
        elif token == tokenizer.pad_token:
            labels[idx] = ""
        else:
            labels[idx] = token
    return labels


def load_unigrams(kenlm_model: Path | None, unigrams_file: Path | None) -> list[str] | None:
    """Load lexicon words for pyctcdecode; auto-derive from the LM corpus file."""
    path = unigrams_file
    if path is None and kenlm_model is not None:
        # data/lm/lin_5gram.binary -> data/lm/lin.txt (corpus written by build_kenlm_decoders.py)
        candidate = kenlm_model.with_name(kenlm_model.name.split("_")[0] + ".txt")
        if candidate.exists():
            path = candidate
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Unigrams file not found: {path}")
    words: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            words.update(line.split())
    print(f"Loaded {len(words)} unigrams from {path}")
    return sorted(words)


def build_pyctcdecode_decoder(processor, args: argparse.Namespace):
    if args.decoder_mode == "greedy":
        return None
    if args.decoder_mode == "beam_lm" and args.kenlm_model is None:
        raise ValueError("--kenlm-model is required when --decoder-mode beam_lm")
    if args.kenlm_model is not None and not args.kenlm_model.exists():
        raise FileNotFoundError(f"KenLM model not found: {args.kenlm_model}")

    try:
        import logging

        from pyctcdecode import build_ctcdecoder

        # the multi-char <s>/</s>/[UNK] labels are intentional; silence the repeat warnings
        logging.getLogger("pyctcdecode").setLevel(logging.ERROR)
    except ImportError as exc:
        raise RuntimeError(
            "pyctcdecode is required for --decoder-mode beam/beam_lm. "
            "Install with `uv pip install --python .venv/bin/python pyctcdecode`."
        ) from exc

    return build_ctcdecoder(
        build_ctc_labels(processor.tokenizer),
        kenlm_model_path=str(args.kenlm_model) if args.kenlm_model else None,
        unigrams=load_unigrams(args.kenlm_model, getattr(args, "unigrams_file", None)),
        alpha=args.alpha,
        beta=args.beta,
    )


def decode_logits(processor, logits, pred_ids, decoder, args: argparse.Namespace) -> list[str]:
    tokenizer = processor.tokenizer
    if decoder is None:
        # tokenizer-level decode: Wav2Vec2Processor.batch_decode delegates here anyway, but
        # Wav2Vec2ProcessorWithLM (repos bundling their own pyctcdecode LM, e.g. the Alvin
        # w2v-bert checkpoints) overrides batch_decode to expect LOGITS — bypass it.
        decoded = tokenizer.batch_decode(pred_ids)
    else:
        logits_np = logits.detach().float().cpu().numpy()
        decoded = [decoder.decode(item, beam_width=args.beam_width) for item in logits_np]
        leftover_specials = [
            token
            for token in (
                tokenizer.unk_token,
                getattr(tokenizer, "bos_token", None),
                getattr(tokenizer, "eos_token", None),
                "⁇",  # pyctcdecode's UNK output token
            )
            if token
        ]
        for special in leftover_specials:
            decoded = [text.replace(special, " ") for text in decoded]
    return [clean_ctc_prediction(text, tokenizer.word_delimiter_token) for text in decoded]


def main() -> None:
    args = parse_args()
    try:
        import torch
        from transformers import AutoModelForCTC
    except ImportError as exc:
        raise RuntimeError("torch and transformers are required. Install dependencies with `uv sync`.") from exc

    if args.hf_processor:
        # HF repo id or local dir; the checkpoint's own processor knows its input format
        # (input_features for w2v-BERT, input_values for Wav2Vec2/XLS-R).
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(str(args.checkpoint))
    else:
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
        vocab_path = resolve_vocab_path(args.checkpoint, args.vocab_path)
        processor = build_processor(vocab_path)

    ds = load_split(args.dataset_dir, args.split)
    if args.languages:
        language_set = set(args.languages)
        ds = ds.filter(lambda row, wanted=language_set: row["language"] in wanted)
    if args.max_samples is not None:
        ds = ds.select(range(min(len(ds), args.max_samples)))
    if len(ds) == 0:
        raise ValueError("No examples selected for inference.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCTC.from_pretrained(str(args.checkpoint)).to(device)
    model.eval()
    decoder = build_pyctcdecode_decoder(processor, args)

    output = args.output
    if output is None:
        output = Path("outputs/predictions") / f"{clean_name(str(args.checkpoint))}_{args.split}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for start in range(0, len(ds), args.batch_size):
        batch = ds[start : start + args.batch_size]
        audios = [audio["array"] for audio in batch["audio"]]
        inputs = processor(
            audios,
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        decoded = decode_logits(processor, logits, pred_ids, decoder, args)
        for example_id, pred in zip(batch["ID"], decoded, strict=True):
            rows.append({"ID": example_id, "Target": pred})
        print(f"Predicted {len(rows)}/{len(ds)}", flush=True)

    write_csv_rows(output, rows, ["ID", "Target"])
    print(f"Wrote predictions to {output}")


if __name__ == "__main__":
    main()
