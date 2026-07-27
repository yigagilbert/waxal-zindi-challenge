#!/usr/bin/env python3
"""Audio-based language ID: linear probe on the champion encoder's pooled hidden states.

Why: the transcript-perplexity LID degrades when ASR quality collapses (Phase-2 domain
shift garbles greedy transcripts -> KenLM discrimination degrades -> misrouting).
Acoustic/phonotactic cues survive domain shift better, so a probe on the CTC model's own
encoder representations is a more robust router — and needs no external model (rules-safe).

Pipeline: extract mean-pooled last-hidden-state per clip (GPU, one pass) -> train a torch
linear classifier on WAXAL train labels -> report validation accuracy/confusion -> predict
languages for an unlabeled dataset (e.g. Phase-2 test) and write ID,language CSV + report.

Usage:
  python scripts/train_audio_lid.py \
    --checkpoint champion/checkpoint-24000 \
    --train-dataset-dir data/processed --train-split train --max-per-language 4000 \
    --eval-dataset-dir data/processed --eval-split validation \
    --predict-dataset-dir data/processed_phase2 --predict-split test \
    --output-dir outputs/audio_lid
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waxal.data import write_csv_rows  # noqa: E402
from waxal.utils import ensure_dir, json_dump  # noqa: E402

LANGS = ["lin", "lug", "sna"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-dataset-dir", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-dataset-dir", type=Path, default=None)
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--predict-dataset-dir", type=Path, default=None)
    parser.add_argument("--predict-split", default="test")
    parser.add_argument("--max-per-language", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/audio_lid"))
    return parser.parse_args()


def load_split(dataset_dir: Path, split: str):
    from datasets import load_from_disk

    dd = load_from_disk(dataset_dir / "hf_dataset")
    return dd[split] if hasattr(dd, "keys") else dd


def extract_features(model, processor, ds, *, batch_size: int, device, label: str):
    """Mean-pooled last hidden state per clip (float32 numpy [N, H])."""
    import numpy as np
    import torch

    feats, ids = [], []
    base = getattr(model, "wav2vec2", None) or getattr(model, "wav2vec2_bert", None) or model
    for start in range(0, len(ds), batch_size):
        batch = ds[start : start + batch_size]
        audios = [audio["array"] for audio in batch["audio"]]
        inputs = processor(audios, sampling_rate=16_000, return_tensors="pt", padding=True, return_attention_mask=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            hidden = base(**inputs).last_hidden_state  # [B, T', H]
        # mask padded frames where possible
        lengths = None
        if "attention_mask" in inputs and hasattr(model, "_get_feat_extract_output_lengths"):
            try:
                lengths = model._get_feat_extract_output_lengths(inputs["attention_mask"].sum(-1))
            except Exception:
                lengths = None
        for i in range(hidden.shape[0]):
            t = int(lengths[i]) if lengths is not None else hidden.shape[1]
            t = min(max(t, 1), hidden.shape[1])
            feats.append(hidden[i, :t].mean(dim=0).float().cpu().numpy())
        ids.extend(batch["ID"])
        if len(ids) % 200 < batch_size:
            print(f"  [{label}] features {len(ids)}/{len(ds)}", flush=True)
    return np.stack(feats), ids


def main() -> None:
    args = parse_args()
    import numpy as np
    import torch
    from transformers import AutoModelForCTC

    sys.path.insert(0, "scripts")
    from run_xlsr_inference import build_processor, resolve_vocab_path  # reuse champion processor path

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = build_processor(resolve_vocab_path(args.checkpoint, None))
    model = AutoModelForCTC.from_pretrained(str(args.checkpoint)).to(device)
    model.eval()
    ensure_dir(args.output_dir)

    # ---- train features (balanced subsample) ----
    train_ds = load_split(args.train_dataset_dir, args.train_split)
    if "language" not in train_ds.column_names:
        raise SystemExit("train split needs a language column")
    by_lang: dict[str, list[int]] = defaultdict(list)
    for idx, lang in enumerate(train_ds["language"]):
        if lang in LANGS:
            by_lang[lang].append(idx)
    rng = np.random.default_rng(42)
    keep: list[int] = []
    for lang in LANGS:
        idxs = by_lang[lang]
        rng.shuffle(idxs)
        keep.extend(idxs[: args.max_per_language])
    keep.sort()
    train_sub = train_ds.select(keep)
    print(f"train probe set: {Counter(train_sub['language'])}")
    X_train, _ = extract_features(model, processor, train_sub, batch_size=args.batch_size, device=device, label="train")
    y_train = np.array([LANGS.index(l) for l in train_sub["language"]])

    # ---- linear probe (pure torch; no sklearn dependency) ----
    Xt = torch.tensor(X_train, device=device)
    mu, sd = Xt.mean(0, keepdim=True), Xt.std(0, keepdim=True) + 1e-6
    Xt = (Xt - mu) / sd
    yt = torch.tensor(y_train, device=device)
    clf = torch.nn.Linear(Xt.shape[1], len(LANGS)).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(args.epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(clf(Xt), yt)
        loss.backward()
        opt.step()
    train_acc = float((clf(Xt).argmax(-1) == yt).float().mean())
    print(f"probe train accuracy: {train_acc:.4f}")

    report: dict = {"checkpoint": str(args.checkpoint), "train_size": len(y_train), "train_accuracy": train_acc}

    def predict(X: "np.ndarray") -> "np.ndarray":
        with torch.no_grad():
            Z = (torch.tensor(X, device=device) - mu) / sd
            return clf(Z).argmax(-1).cpu().numpy()

    # ---- validation accuracy/confusion ----
    if args.eval_dataset_dir is not None:
        eval_ds = load_split(args.eval_dataset_dir, args.eval_split)
        X_eval, eval_ids = extract_features(model, processor, eval_ds, batch_size=args.batch_size, device=device, label="eval")
        pred = predict(X_eval)
        true = [l for l in eval_ds["language"]]
        correct = sum(1 for p, t in zip(pred, true, strict=True) if LANGS[p] == t)
        confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for p, t in zip(pred, true, strict=True):
            confusion[t][LANGS[p]] += 1
        report["eval"] = {
            "accuracy": correct / max(len(true), 1),
            "num_examples": len(true),
            "confusion_true_vs_predicted": {t: dict(d) for t, d in confusion.items()},
        }
        print(f"validation LID accuracy: {report['eval']['accuracy']:.4f}")
        print(json.dumps(report["eval"]["confusion_true_vs_predicted"], indent=2))

    # ---- predict unlabeled set (e.g. Phase-2) ----
    if args.predict_dataset_dir is not None:
        pred_ds = load_split(args.predict_dataset_dir, args.predict_split)
        X_pred, pred_ids = extract_features(model, processor, pred_ds, batch_size=args.batch_size, device=device, label="predict")
        pred = predict(X_pred)
        rows = [{"ID": i, "language": LANGS[p]} for i, p in zip(pred_ids, pred, strict=True)]
        out_csv = args.output_dir / "audio_lid_predictions.csv"
        write_csv_rows(out_csv, rows, ["ID", "language"])
        report["predict"] = {
            "num_examples": len(rows),
            "assigned_counts": dict(Counter(r["language"] for r in rows)),
            "output": str(out_csv),
        }
        print(f"predicted mix: {report['predict']['assigned_counts']}")

    json_dump(report, args.output_dir / "audio_lid_report.json")
    print(f"Report: {args.output_dir / 'audio_lid_report.json'}")


if __name__ == "__main__":
    main()
