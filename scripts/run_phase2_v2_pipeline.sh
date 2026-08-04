#!/usr/bin/env bash
# RETIRED: this is the withdrawn-test Whisper pipeline, despite its old filename.
#
# It reproduces the old Acholi/Nyankore/Soga/Masaaba chain:
#   af51 auto decode -> text LID -> acoustic LID -> LID fusion
#   -> SALT forced decode (lp0.6) -> long-clip repair (>30s) -> loop repair
#   -> 8-sample stochastic decode -> MBR margin gate (frozen 0.075)
#
# The corrected test is Lingala/Shona and the current champion is XLS-R CTC plus
# per-language KenLM decoding. Keeping the historical body below is useful for
# provenance, but executing it would spend GPU time on the wrong model/languages.
#
# Every stage writes to outputs/day4_h100 and is skipped if its output exists,
# so the script is resumable. Stop on first error.
set -euo pipefail
echo "ERROR: retired withdrawn-test pipeline (ach/nyn/xog/myx Whisper stack)." >&2
echo "Use the corrected Lingala/Shona XLS-R CTC pipeline recorded in outputs/day4_h100/RESULTS.md." >&2
exit 2

cd "$(dirname "$0")/.."
source ~/waxal-venv/bin/activate

OUT=outputs/day4_h100
DS=data/processed_phase2_v2
mkdir -p "$OUT" outputs/models outputs/analysis
SALT=Sunbird/asr-whisper-large-v3-salt
AF51=huwenjie333/whisper-v3-ft-af51
MAP="ach=50357 nyn=50354 xog=50352 myx=50349"

step () { echo; echo "=== [$(date +%H:%M:%S)] $* ==="; }

# ---------------------------------------------------------------- 1. af51 auto
if [ ! -f "$OUT/af51_v2_auto_beam5.csv" ]; then
  step "af51 auto decode (LID source + fallback engine)"
  python scripts/run_whisper_inference.py --model-name "$AF51" \
    --dataset-dir "$DS" --split test --num-beams 5 --max-new-tokens 220 \
    --batch-size 8 --output "$OUT/af51_v2_auto_beam5.csv"
fi

# ------------------------------------------------------------- 2. text LID
if [ ! -f "$OUT/text_lid_v2.csv" ]; then
  step "text LID on af51 transcripts"
  python scripts/train_text_lid.py \
    --dataset-dir data/phase2_train \
    --eval-predictions outputs/day2_h100/af51_val_auto_full_beam5.csv \
    --predict-csv "$OUT/af51_v2_auto_beam5.csv" \
    --fit-validation-for-predict --unknown-threshold 0.80 \
    --routing-output "$OUT/text_lid_v2.csv" \
    --eval-details-output "$OUT/text_lid_v2_validation.csv" \
    --report-output "$OUT/text_lid_v2_report.json" \
    --model-output outputs/models/text_lid_v2.joblib
fi

# ---------------------------------------------------------- 3. acoustic LID
if [ ! -f "$OUT/audio_lid_v2/phase2_predictions.csv" ]; then
  step "acoustic LID on frozen SALT encoder"
  python scripts/train_whisper_audio_lid.py \
    --model-name "$SALT" --dataset-dir data/phase2_train \
    --predict-dataset-dir "$DS" --predict-split test \
    --max-train-per-language 1200 --batch-size 16 \
    --fit-validation-for-predict --output-dir "$OUT/audio_lid_v2"
fi

# -------------------------------------------------------------- 4. LID fusion
if [ ! -f "$OUT/lid_fused_v2.csv" ]; then
  step "fuse text + acoustic LID"
  python scripts/fuse_lid_predictions.py \
    --text-validation "$OUT/text_lid_v2_validation.csv" \
    --audio-validation "$OUT/audio_lid_v2/validation_predictions.csv" \
    --text-test "$OUT/text_lid_v2.csv" \
    --audio-test "$OUT/audio_lid_v2/phase2_predictions.csv" \
    --output "$OUT/lid_fused_v2.csv" \
    --validation-output "$OUT/lid_fused_v2_validation.csv" \
    --report-output "$OUT/lid_fusion_v2_report.json"
  python - <<'PY'
import csv, collections
rows = list(csv.DictReader(open("outputs/day4_h100/lid_fused_v2.csv", encoding="utf-8-sig")))
print("FUSED TEST ROUTING:", dict(collections.Counter(r["language"] for r in rows).most_common()))
PY
fi

# ------------------------------------------------- 5. SALT forced decode lp0.6
if [ ! -f "$OUT/salt_v2_forced_lp06.csv" ]; then
  step "SALT forced decode (beam 5, lp0.6)"
  python scripts/run_whisper_inference.py --model-name "$SALT" \
    --dataset-dir "$DS" --split test --num-beams 5 --length-penalty 0.6 \
    --max-new-tokens 220 --batch-size 8 \
    --language-csv "$OUT/lid_fused_v2.csv" --language-map $MAP \
    --output "$OUT/salt_v2_forced_lp06.csv"
fi

# --------------------------------------------- 6. long clips (>30 s) chunked
if [ ! -f "$OUT/salt_v2_longclips.csv" ]; then
  step "chunked decode for clips beyond the 30 s window"
  python scripts/decode_long_clips.py --model-name "$SALT" \
    --dataset-dir "$DS" --split test --min-duration 30 \
    --language-csv "$OUT/lid_fused_v2.csv" --language-map $MAP \
    --num-beams 5 --length-penalty 0.6 \
    --output "$OUT/salt_v2_longclips.csv"
fi
if [ ! -f "$OUT/salt_v2_base.csv" ]; then
  step "splice long-clip repairs into the base decode"
  python - <<'PY'
import csv
base = list(csv.DictReader(open("outputs/day4_h100/salt_v2_forced_lp06.csv", encoding="utf-8-sig")))
long_ = {r["ID"]: r["Target"] for r in csv.DictReader(open("outputs/day4_h100/salt_v2_longclips.csv", encoding="utf-8-sig"))}
n = 0
for r in base:
    if r["ID"] in long_ and long_[r["ID"]].strip() and long_[r["ID"]] != r["Target"]:
        r["Target"] = long_[r["ID"]]; n += 1
with open("outputs/day4_h100/salt_v2_base.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["ID", "Target"]); w.writeheader(); w.writerows(base)
print(f"long-clip rows replaced: {n}")
PY
fi

# ------------------------------------------------------------ 7. loop repair
if [ ! -f "$OUT/salt_v2_loopfix.csv" ]; then
  step "strict repetition-loop repair (af51 fallback)"
  python scripts/route_repetition_loops.py \
    --primary "$OUT/salt_v2_base.csv" --fallback "$OUT/af51_v2_auto_beam5.csv" \
    --ngram-order 4 --min-count 4 \
    --output "$OUT/salt_v2_loopfix.csv" --report "$OUT/loopfix_v2_report.json"
fi

# ------------------------------------------- 8. stochastic samples for MBR
if [ ! -f "$OUT/salt_v2_sample8_t02.csv" ]; then
  step "8-sample stochastic decode (T=0.2) for MBR"
  python scripts/run_whisper_inference.py --model-name "$SALT" \
    --dataset-dir "$DS" --split test --do-sample --temperature 0.2 \
    --num-return-sequences 8 --seed 42 --max-new-tokens 220 --batch-size 8 \
    --language-csv "$OUT/lid_fused_v2.csv" --language-map $MAP \
    --output "$OUT/salt_v2_sample8_t02.csv"
fi

# ------------------------------------------------ 9. MBR margin gate (0.075)
if [ ! -f "$OUT/salt_v2_mbr_decisions.csv" ]; then
  step "MBR risk scoring against the loop-repaired anchor"
  python scripts/select_nbest_mbr.py \
    --nbest "$OUT/salt_v2_sample8_t02.csv" --anchor "$OUT/salt_v2_loopfix.csv" \
    --normalization raw --output "$OUT/salt_v2_mbr_best.csv" \
    --output-report "$OUT/salt_v2_mbr_report.json" \
    --decision-log "$OUT/salt_v2_mbr_decisions.csv"
fi
if [ ! -f "$OUT/salt_v2_final.csv" ]; then
  step "apply frozen validation margin 0.075 (protects loop repairs)"
  python scripts/apply_mbr_margin.py \
    --decisions "$OUT/salt_v2_mbr_decisions.csv" --primary "$OUT/salt_v2_loopfix.csv" \
    --threshold 0.075 --output "$OUT/salt_v2_final.csv" \
    --report "$OUT/margin_v2_report.json"
fi

step "PIPELINE COMPLETE"
python - <<'PY'
import csv, collections, hashlib
rows = list(csv.DictReader(open("outputs/day4_h100/salt_v2_final.csv", encoding="utf-8-sig")))
ids = [r["ID"] for r in rows]
print("rows:", len(rows), "| unique:", len(set(ids)),
      "| empty:", sum(1 for r in rows if not r["Target"].strip()),
      "| newlines:", sum(1 for r in rows if "\n" in r["Target"] or "\r" in r["Target"]))
route = {r["ID"]: r["language"] for r in csv.DictReader(open("outputs/day4_h100/lid_fused_v2.csv", encoding="utf-8-sig"))}
print("routing:", dict(collections.Counter(route.get(i, "?") for i in ids).most_common()))
print("sha256:", hashlib.sha256(open("outputs/day4_h100/salt_v2_final.csv", "rb").read()).hexdigest())
PY
