#!/usr/bin/env python3
"""Run deterministic Whisper-style inference on prepared WAXAL splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import write_csv_rows  # noqa: E402
from waxal.utils import clean_name  # noqa: E402


def load_split(dataset_dir: Path, split: str):
    """Load a prepared Hugging Face split from disk."""
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError("datasets is required. Install with `uv sync` first.") from exc
    dataset_path = dataset_dir / "hf_dataset"
    dataset_dict = load_from_disk(dataset_path)
    if split not in dataset_dict:
        raise ValueError(f"Split {split!r} not found in {dataset_path}. Available: {list(dataset_dict)}")
    return dataset_dict[split]


def adapter_base_model(adapter_path: str) -> str | None:
    """Read PEFT adapter base model from adapter_config.json, if present."""
    config_path = Path(adapter_path) / "adapter_config.json"
    if not config_path.exists():
        try:
            from huggingface_hub import hf_hub_download

            config_path = Path(hf_hub_download(repo_id=adapter_path, filename="adapter_config.json"))
        except Exception:
            return None
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return config.get("base_model_name_or_path")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="openai/whisper-large-v3-turbo")
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Optional immutable Hugging Face model revision/commit for reproducibility.",
    )
    parser.add_argument("--adapter-path", default=None, help="Optional local PEFT/LoRA checkpoint or Hugging Face repo ID.")
    parser.add_argument("--merge-adapter", action="store_true", help="Merge PEFT adapter before inference.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--languages", nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument(
        "--num-return-sequences",
        type=int,
        default=1,
        help="Return this many beam hypotheses per clip. Values >1 write a long-form n-best CSV.",
    )
    parser.add_argument("--length-penalty", type=float, default=None, help="Beam-search length penalty (generate default when omitted).")
    parser.add_argument("--do-sample", action="store_true", help="Use stochastic decoding instead of deterministic beam search.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preserve-generation-config",
        action="store_true",
        help="Keep the checkpoint's forced/suppressed token settings for a model-card-faithful A/B.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--language-csv",
        type=Path,
        default=None,
        help="CSV with ID,language columns: per-clip forced decoder language. "
        "'unk' or blank means auto-detect for that clip.",
    )
    parser.add_argument(
        "--language-map",
        nargs="*",
        default=None,
        help="Relabel language-csv values before forcing, e.g. --language-map ach=luo myx=lug. "
        "Map a value to empty (unk=) to leave those clips on auto-detect.",
    )
    parser.add_argument(
        "--force-language",
        default=None,
        help="Force a single language code for every clip (overrides --language-csv).",
    )
    args = parser.parse_args()
    if args.num_return_sequences < 1:
        parser.error("--num-return-sequences must be at least 1")
    if not args.do_sample and args.num_return_sequences > args.num_beams:
        parser.error("--num-return-sequences cannot exceed --num-beams")

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("torch, peft, and transformers are required. Install with `uv sync` first.") from exc
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ds = load_split(args.dataset_dir, args.split)
    if args.languages:
        language_set = set(args.languages)
        ds = ds.filter(lambda row: row["language"] in language_set)
    if args.max_samples is not None:
        ds = ds.select(range(min(len(ds), args.max_samples)))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

    model_name = args.model_name
    adapter_path = args.adapter_path
    model_path = Path(model_name)
    if adapter_path is None and (model_path / "adapter_config.json").exists():
        adapter_path = str(model_path)
        base_model = adapter_base_model(adapter_path)
        if base_model:
            model_name = base_model

    processor_source = adapter_path or model_name
    try:
        processor = AutoProcessor.from_pretrained(
            processor_source,
            revision=args.model_revision if adapter_path is None else None,
        )
    except Exception:
        processor = AutoProcessor.from_pretrained(
            model_name,
            revision=args.model_revision,
        )

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        revision=args.model_revision,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
        if args.merge_adapter:
            model = model.merge_and_unload()
    model = model.to(device)
    model.eval()
    if not args.preserve_generation_config:
        if hasattr(model.config, "forced_decoder_ids"):
            model.config.forced_decoder_ids = None
        if hasattr(model.config, "suppress_tokens"):
            model.config.suppress_tokens = []

    output = args.output
    if output is None:
        run_name = str(adapter_path) if adapter_path is not None else args.model_name
        output = Path("outputs/predictions") / f"{clean_name(run_name)}_{args.split}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    # ---- per-clip language forcing -------------------------------------------
    id_to_lang: dict[str, str] = {}
    if args.language_csv is not None:
        import csv as _csv

        with args.language_csv.open(encoding="utf-8-sig") as f:
            for row in _csv.DictReader(f):
                id_to_lang[row["ID"]] = (row.get("language") or "").strip()
    remap: dict[str, str] = {}
    for pair in args.language_map or []:
        src, _, dst = pair.partition("=")
        remap[src] = dst

    def clip_language(example_id: str) -> str | None:
        if args.force_language:
            return args.force_language
        code = id_to_lang.get(example_id, "")
        code = remap.get(code, code)
        if not code or code == "unk":
            return None
        return code

    tokenizer = processor.tokenizer
    lang_to_id = getattr(model.generation_config, "lang_to_id", None) or {}
    sot_id = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    transcribe_id = tokenizer.convert_tokens_to_ids("<|transcribe|>")
    notimestamps_id = tokenizer.convert_tokens_to_ids("<|notimestamps|>")

    def language_gen_kwargs(code: str | None) -> dict:
        """generate() kwargs that force `code`; empty dict = auto-detect.

        `code` may be a language name/ISO code, or a raw decoder token id (all digits)
        for models that repurpose stock Whisper language slots without renaming the
        token (e.g. Sunbird SALT: ach=50357, nyn=50354, xog=50352, myx=50349).
        """
        if code is None:
            return {}
        if code.isdigit():
            return {"prefix": [int(code), transcribe_id, notimestamps_id]}
        token = f"<|{code}|>"
        if token in lang_to_id:
            return {"language": code, "task": "transcribe"}
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            return {"prefix": [token_id, transcribe_id, notimestamps_id]}
        print(f"WARNING: language '{code}' has no token in this model; auto-detecting those clips.")
        return {}

    # forced_decoder_ids was deprecated/removed in newer transformers; discover the
    # working mechanism on the first forced batch and stick with it.
    forcing_mode: dict[str, str | None] = {"mode": None}

    all_ids = ds["ID"]
    gen_kwargs_cache: dict[str | None, dict] = {}
    groups: dict[str | None, list[int]] = {}
    for idx, example_id in enumerate(all_ids):
        code = clip_language(example_id) if (id_to_lang or args.force_language) else None
        groups.setdefault(code, []).append(idx)
    if len(groups) > 1 or next(iter(groups)) is not None:
        print("language groups:", {k or "auto": len(v) for k, v in sorted(groups.items(), key=lambda kv: kv[0] or "")})

    preds_by_id: dict[str, list[dict[str, str | int | float]]] = {}
    done = 0
    for code, indices in groups.items():
        if code not in gen_kwargs_cache:
            gen_kwargs_cache[code] = language_gen_kwargs(code)
        extra_kwargs = gen_kwargs_cache[code]
        subset = ds.select(indices)
        for start in range(0, len(subset), args.batch_size):
            batch = subset[start : start + args.batch_size]
            audios = [audio["array"] for audio in batch["audio"]]
            inputs = processor(
                audios,
                sampling_rate=16_000,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
            )
            if "input_features" not in inputs:
                raise ValueError(f"Expected processor to return input_features, got keys: {list(inputs)}")

            input_features = inputs["input_features"]

            if input_features.shape[-1] != 3000:
                raise ValueError(
                    f"Whisper expects input_features length 3000, "
                    f"got shape {tuple(input_features.shape)}"
                )

            model_dtype = next(model.parameters()).dtype

            input_features = input_features.to(device=device, dtype=model_dtype)

            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device=device)

            gen_kwargs = dict(
                input_features=input_features,
                attention_mask=attention_mask,
                do_sample=args.do_sample,
                num_beams=args.num_beams,
                num_return_sequences=args.num_return_sequences,
                max_new_tokens=args.max_new_tokens,
            )
            if args.temperature is not None:
                gen_kwargs["temperature"] = args.temperature
            if args.top_p is not None:
                gen_kwargs["top_p"] = args.top_p
            if args.top_k is not None:
                gen_kwargs["top_k"] = args.top_k
            if args.num_return_sequences > 1:
                gen_kwargs.update(return_dict_in_generate=True, output_scores=True)
            if args.length_penalty is not None:
                gen_kwargs["length_penalty"] = args.length_penalty
            prefix = extra_kwargs.get("prefix")
            if prefix is None:
                gen_kwargs.update(extra_kwargs)
            with torch.no_grad():
                if prefix is None:
                    generated = model.generate(**gen_kwargs)
                else:
                    modes = (
                        [forcing_mode["mode"]]
                        if forcing_mode["mode"]
                        else ["forced_decoder_ids", "decoder_input_ids"]
                    )
                    last_exc: Exception | None = None
                    for mode in modes:
                        try:
                            if mode == "forced_decoder_ids":
                                generated = model.generate(
                                    **gen_kwargs,
                                    forced_decoder_ids=[(i + 1, t) for i, t in enumerate(prefix)],
                                )
                            else:
                                decoder_input_ids = torch.tensor(
                                    [[sot_id, *prefix]] * input_features.shape[0],
                                    device=device,
                                    dtype=torch.long,
                                )
                                generated = model.generate(**gen_kwargs, decoder_input_ids=decoder_input_ids)
                        except (TypeError, ValueError) as exc:
                            last_exc = exc
                            continue
                        if forcing_mode["mode"] != mode:
                            forcing_mode["mode"] = mode
                            print(f"language forcing via {mode}")
                        break
                    else:
                        raise last_exc
            sequences = generated.sequences if hasattr(generated, "sequences") else generated
            decoded = processor.batch_decode(sequences, skip_special_tokens=True)
            sequence_scores = getattr(generated, "sequences_scores", None)
            if sequence_scores is None and hasattr(generated, "scores") and generated.scores:
                beam_indices = getattr(generated, "beam_indices", None)
                transition_scores = model.compute_transition_scores(
                    sequences,
                    generated.scores,
                    beam_indices=beam_indices,
                    normalize_logits=True,
                )
                # Forced/padded positions have a transition score of zero. Average
                # only genuinely generated tokens so clips of different lengths
                # remain comparable.
                generated_mask = torch.isfinite(transition_scores) & (transition_scores < 0)
                token_counts = generated_mask.sum(dim=1).clamp_min(1)
                finite_scores = torch.where(
                    generated_mask,
                    transition_scores,
                    torch.zeros_like(transition_scores),
                )
                average_scores = finite_scores.sum(dim=1) / token_counts
                scores = [float(x) for x in average_scores.detach().cpu().tolist()]
            elif sequence_scores is None:
                scores = [0.0] * len(decoded)
            else:
                scores = [float(x) for x in sequence_scores.detach().cpu().tolist()]
            expected = len(batch["ID"]) * args.num_return_sequences
            if len(decoded) != expected:
                raise RuntimeError(
                    f"Expected {expected} decoded hypotheses, got {len(decoded)}"
                )
            for batch_index, example_id in enumerate(batch["ID"]):
                candidates = []
                offset = batch_index * args.num_return_sequences
                for rank in range(args.num_return_sequences):
                    pos = offset + rank
                    candidates.append(
                        {
                            "rank": rank + 1,
                            "Target": decoded[pos].strip(),
                            "sequence_score": scores[pos],
                            "language": code or "auto",
                        }
                    )
                preds_by_id[example_id] = candidates
            done += len(batch["ID"])
            print(f"Predicted {done}/{len(ds)} [{code or 'auto'}]", flush=True)

    if args.num_return_sequences == 1:
        rows = [
            {"ID": example_id, "Target": preds_by_id[example_id][0]["Target"]}
            for example_id in all_ids
        ]
        fields = ["ID", "Target"]
    else:
        rows = [
            {"ID": example_id, **candidate}
            for example_id in all_ids
            for candidate in preds_by_id[example_id]
        ]
        fields = ["ID", "rank", "Target", "sequence_score", "language"]
    write_csv_rows(output, rows, fields)
    print(f"Wrote predictions to {output}")


if __name__ == "__main__":
    main()
