# Fresh-Box Restore Runbook

## ⚡ Phase-2 new-VM playbook (definitive from 2026-07-28; Phase-1 restore below only for champion work)

State (07-29): best **0.699656** (`phase2_salt_forced_beam5.csv` — SALT engine, forced
language via clusters v1, beam 5, raw — keep SELECTED until beaten). SALT beat af51 0.6835;
the SALT+af51-unk splice scored LOWER (0.6967) → **af51 retired; SALT is base engine for
everything** (splices use SALT as base). Board compressed: ranks 6–26 span 0.709→0.6997;
#1 Yen 0.71897 (gap 0.0193) — the fine-tune (§5) is the differentiator. RAW text canonical.
Language map v1: ach 477 / nyn 401 / myx 267 / xog 84 / unk 271. Text-LID rung CLOSED
07-29: clusters-v3 (+77 forced) scored 0.698695 < 0.699656 — auto-detect was fine on the
unk residue. §5 fine-tune GATE RESULT (07-29 night): LoRA trained clean (eval_loss 0.504→0.433) but the
forced gate FAILED 3/4 languages (base already knows this corpus — no-new-information law);
PASSED only myx (combined 0.3236→0.3143, WER 0.500→0.478). Action = myx-only adapter splice
over the 0.699656 baseline (est. +0.002). Adapter at artifacts repo
`whisper_salt_phase2_lora/checkpoint-4000`. Continuation training closed. Remaining rungs:
Sunbird-51 forced A/B, beam 8–10 / length_penalty on base SALT, native-speaker per-cluster
review. Close 2026-08-03, 5 subs/day, one variable per submission.

### 0. Rent the GPU (Vast.ai)

**Recommended: 1× A100 80GB SXM** (~$1.1–1.4/hr) — the LoRA fine-tune is the long pole and
runs ~4–5h there vs ~10–13h on a 4090; 80GB also allows batch-16 eval generation and a
full-fine-tune fallback. Filters: disk **≥ 200GB**, CUDA ≥ 12.1 image, inet_down ≥ 500 Mbps,
verified host. Acceptable alternates: H100 80GB (faster, pricier), RTX 4090 24GB (works —
config is sized for it — but serializes the week), RTX 5090 32GB (cheap; needs the Blackwell
torch fix boxed below). Budget through close at A100 rates: ~$60–90.
On 80GB set `per_device_train_batch_size: 8, gradient_accumulation_steps: 2` in the config
(same effective 16).

### 1. Environment (~10 min)

Current box (2026-07-29): Azure `gyiga-finetuning-gpu-vm`, **H100 NVL 95GB** (sm_90, no
Blackwell fix needed) — on ≥80GB VRAM set `per_device_train_batch_size: 8,
gradient_accumulation_steps: 2` in the fine-tune config and `--batch-size 16` on decodes.

```bash
git clone https://github.com/yigagilbert/waxal-zindi-challenge && cd waxal-zindi-challenge
pip install uv 2>/dev/null || true
# venv, NOT --system: system installs hit Permission denied on /usr/local (seen 07-29)
uv venv ~/waxal-venv && source ~/waxal-venv/bin/activate
echo 'source ~/waxal-venv/bin/activate' >> ~/.bashrc   # tmux windows inherit it
uv pip install "datasets>=3.0,<4" soundfile librosa "transformers>=4.46" torch \
  huggingface_hub peft accelerate jiwer pyyaml
# `huggingface-cli` is a deprecated NO-OP on hub >=1.x — it prints a hint and does nothing.
# Use `hf` everywhere. Token MUST have accepted: google/WaxalNLP,
# Sunbird/asr-whisper-large-v3-salt, Sunbird/asr-whisper-51-african-languages.
hf auth login
nvidia-smi              # if sm_120 (5090/Blackwell) apply the torch fix below
tmux new -s main        # window 0 = decode/submissions, window 1 = training (Ctrl-b c)
```
(pyctcdecode/kenlm are Phase-1-only — skip unless doing champion work.)

### 2. Phase-2 data + prior artifacts (~10 min)

```bash
mkdir -p data/phase2 && cd data/phase2 \
  && curl -LO https://storage.googleapis.com/waxalphase2/audio.zip && unzip -q audio.zip && cd ../..
hf download yigagilbert/waxal-private-artifacts --repo-type dataset \
  --include "phase2/*" --local-dir artifacts_phase2   # Test_phase2.csv, all 07-27/28 predictions, routing table
cp artifacts_phase2/phase2/Test_phase2.csv data/phase2/
mkdir -p outputs/analysis outputs/predictions
cp artifacts_phase2/phase2/analysis/phase2_language_clusters.csv outputs/analysis/
cp artifacts_phase2/phase2/predictions/phase2_af51_beam5_raw.csv outputs/predictions/ \
  || ls artifacts_phase2/phase2/predictions/   # find the raw beam-5 af51 file if named differently
python scripts/prepare_phase2_test.py --audio-dir data/phase2/audio \
  --test-csv data/phase2/Test_phase2.csv          # -> data/processed_phase2 (1500 clips, 0 missing)
```

### 3. Submission track FIRST (window 0, ~2h GPU): SALT forced-language decode

SALT (`Sunbird/asr-whisper-large-v3-salt`) supports all four clusters via repurposed stock
Whisper slots (SALT_LANGUAGE_TOKENS_WHISPER): ach=50357, nyn=50354, xog=50352, myx=50349
(also lug=50355, teo=50353, lgg=50356, ttj=50351, kin=50350, swa=50318, eng=50259).
```bash
python scripts/run_whisper_inference.py --model-name Sunbird/asr-whisper-large-v3-salt \
  --dataset-dir data/processed_phase2 --split test --num-beams 5 --max-new-tokens 220 --batch-size 8 \
  --language-csv outputs/analysis/phase2_language_clusters.csv \
  --language-map ach=50357 nyn=50354 xog=50352 myx=50349 \
  --output outputs/predictions/phase2_salt_forced_beam5_raw.csv
head -12 outputs/predictions/phase2_salt_forced_beam5_raw.csv   # eyeball orthography per cluster first
python scripts/merge_predictions.py --predictions outputs/predictions/phase2_salt_forced_beam5_raw.csv \
  --order data/phase2/Test_phase2.csv --output outputs/submissions/phase2_salt_forced_beam5.csv
wc -l outputs/submissions/phase2_salt_forced_beam5.csv          # must be 1501
```
**Submission 1** = that file, NO normalization. **Submission 2** (zero GPU) = splice — SALT for
clustered clips, proven af51 for the 271 `unk` (SALT auto-detect on repurposed slots is
untrustworthy):
```bash
python scripts/splice_predictions.py \
  --base outputs/predictions/phase2_af51_beam5_raw.csv \
  --overlay outputs/predictions/phase2_salt_forced_beam5_raw.csv \
  --routing outputs/analysis/phase2_language_clusters.csv \
  --overlay-languages ach nyn xog myx \
  --output outputs/predictions/phase2_salt_af51_spliced.csv
python scripts/merge_predictions.py --predictions outputs/predictions/phase2_salt_af51_spliced.csv \
  --order data/phase2/Test_phase2.csv --output outputs/submissions/phase2_salt_af51_spliced.csv
```
**Submission 3 (optional A/B)**: `Sunbird/asr-whisper-51-african-languages`, same command —
but first check its language-code convention (SALT-style raw ids vs real `<|ach|>` tokens):
`python -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('Sunbird/asr-whisper-51-african-languages'); print([x for x in t.additional_special_tokens if 'ach' in x or 'nyn' in x][:10])"`

### 4. Training data (window 1, network/CPU — start while step 3 decodes)

```bash
python scripts/prepare_phase2_train.py --languages ach nyn sog=xog mas=myx \
  --max-per-language 8000 --output-dir data/phase2_train     # WaxalNLP phase-2 langs; prints row counts
  # NAMING TRAP (found 07-29): WaxalNLP has NO xog/myx configs — Lusoga is `sog_asr` and
  # Lumasaba is `mas_asr` (verified by transcript inspection: kh/tsi signatures; NOT Maasai).
  # hubcode=label keeps output labels aligned with our clusters + the SALT token map.
export WAXAL_RAW_DIR=$PWD/google-waxal-asr-challenge20260630-10570-elxebu   # git-tracked CSVs
python scripts/prepare_dataset.py --raw-dir "$WAXAL_RAW_DIR" --splits train validation
  # -> data/processed = the challenge trio lin/lug/sna (~18GB; run in tmux)
```

### 5. Fine-tune: SALT + trio + phase-2 languages (endgame lever)

Language-conditioned labels (SALT slots for Ugandan langs, stock `<|ln|>`/`<|sn|>` for
lin/sna — the challenge expects trio training; the mix also regularizes). Config:
`configs/whisper_salt_phase2_lora.yaml` (LoRA r32 all-modules, lr 2e-4, 4000 steps, eff. batch 16).
```bash
python scripts/train_whisper.py --config configs/whisper_salt_phase2_lora.yaml \
  --max-train-samples 64 --max-eval-samples 32 --max-steps 20        # SMOKE first
python scripts/train_whisper.py --config configs/whisper_salt_phase2_lora.yaml  # real run (tmux!)
# push every saved checkpoint to HF immediately (boxes die):
python -c "from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='checkpoints/whisper_salt_phase2_lora', path_in_repo='whisper_salt_phase2_lora', repo_id='yigagilbert/waxal-private-artifacts', repo_type='dataset', ignore_patterns=['**/optimizer.pt','**/rng_state*'])"
```
In-loop eval generates WITHOUT language forcing → trend-only (Phase-1 lesson). The honest
**GATE** is an external forced decode on the held-out WaxalNLP validation, adapter vs base:
```bash
for M in "" "--adapter-path checkpoints/whisper_salt_phase2_lora/checkpoint-XXXX"; do \
python scripts/run_whisper_inference.py --model-name Sunbird/asr-whisper-large-v3-salt $M \
  --dataset-dir data/phase2_train --split validation --max-samples 800 \
  --num-beams 5 --max-new-tokens 220 --batch-size 8 \
  --language-csv data/phase2_train/validation.csv \
  --language-map ach=50357 nyn=50354 xog=50352 myx=50349 \
  --output outputs/predictions/salt_val_$([ -n "$M" ] && echo adapter || echo base).csv; done
python scripts/evaluate_predictions.py --predictions outputs/predictions/salt_val_base.csv \
  --references data/phase2_train/validation.csv --normalization all --output outputs/analysis/salt_val_base.json
python scripts/evaluate_predictions.py --predictions outputs/predictions/salt_val_adapter.csv \
  --references data/phase2_train/validation.csv --normalization all --output outputs/analysis/salt_val_adapter.json
```
Adapter must beat base per-language on raw/loosest normalization to earn a test-set
submission (then: step-3 command + `--adapter-path`, splice unk from af51, submit).

### 6. Optional rungs (only if 5 is running/blocked)

- **Audio LID retrain** (sharpens the 271 unk clips; validates text clusters):
  `python scripts/train_audio_lid.py --languages ach nyn xog myx --checkpoint <any encoder;
  champion/checkpoint-24000 or pull from HF> --train-dataset-dir data/phase2_train
  --predict-dataset-dir data/processed_phase2 --predict-split test` — route confident unk
  reassignments through splice_predictions.
- Beam 8–10 / length-penalty / temperature-fallback on the best engine.

### Standing rules

Raw text (no normalization) for every Phase-2 submission · one variable per submission ·
keep the best submission SELECTED on Zindi · push predictions/submissions/checkpoints to
`yigagilbert/waxal-private-artifacts` under `phase2/` the moment they exist · every model/
dataset used gets a docs/RULES_AND_DATA_USE.md entry · never overwrite champion artifacts.


Bootstraps a brand-new GPU box (e.g. Vast RTX 4090) to full campaign state in ~3–4 h.
Written 2026-07-27 after the Azure box was deleted. Everything below is recoverable from
git + HF + the gated source datasets; nothing irreplaceable was ever box-local.

> Blackwell warning: on RTX 5090 / RTX PRO 6000 (sm_120) standard torch fails
> ("no kernel image") — fix: `uv pip uninstall torch && uv pip install torch --index-url
> https://download.pytorch.org/whl/cu128`, then `export WAXAL_NO_SYNC=1`. RTX 4090 (sm_89)
> and A100/H100 need nothing.

## 1. Environment (~10 min)

```bash
git clone https://github.com/yigagilbert/waxal-zindi-challenge && cd waxal-zindi-challenge
pip install uv 2>/dev/null || true
uv pip install --system "datasets>=3.0,<4" soundfile librosa "transformers>=4.46" torch \
  pyctcdecode kenlm huggingface_hub peft accelerate
hf auth login                # token with access to gated google/WaxalNLP + private repos (huggingface-cli is a no-op on hub >=1.x)
nvidia-smi                   # confirm VRAM (24 vs 48 GB -> whisper batch 8 vs 16)
export WAXAL_RAW_DIR=$PWD/google-waxal-asr-challenge20260630-10570-elxebu
export HF_DATASETS_CACHE=/dev/shm/hf_datasets_cache && mkdir -p $HF_DATASETS_CACHE
```

## 2. Champion checkpoint (~5 min)

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('yigagilbert/waxal-xlsr300m-champion', local_dir='champion', repo_type='model')"
ls champion/                 # note what's present — if an lm/ or lm_expanded/ folder exists,
                             # copy it to data/ and skip the matching rebuild in step 5
```

## 3. WAXAL audio (~1.5 h, tmux) — train/val/test only (unlabeled-split trap is fixed in code)

```bash
python scripts/prepare_dataset.py --raw-dir "$WAXAL_RAW_DIR"        # -> data/processed (~18GB)
python scripts/prepare_generalization_mix.py --include-fleurs --include-salt \
  --waxal-train-manifest data/processed/train.csv                    # -> data/processed_generalization_mix
```
Disk gotchas (all previously hit): datasets `.filter()/.shuffle()` caches write NEXT TO the
on-disk dataset, not to HF_DATASETS_CACHE — keep the dataset's disk with headroom. After the
mix is built, `rm -rf data/processed/hf_dataset` (the mix contains a WAXAL copy).

## 4. KenLM CLI (~5 min)

```bash
sudo apt-get -qq install -y build-essential cmake libboost-all-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
git clone -q https://github.com/kpu/kenlm.git ~/kenlm
mkdir -p ~/kenlm/build && cd ~/kenlm/build && cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null && make -j$(nproc) && cd -
export PATH=~/kenlm/build/bin:$PATH
```

## 5. Rebuild LMs (~30–60 min, CPU) — skip any component found in the champion repo (step 2)

```bash
# base corpora for all 3 languages (WAXAL train text x2 + FLEURS + SALT + Afrivoice manifests):
python scripts/collect_lm_text.py --waxal-from-hf --waxal-repeat 2 \
  --skip-source wikimedia/wikipedia \
  --max-lines-per-source 60000 --output-dir data/lm_expanded --order 5 --overwrite
# lin corpus v2 (+Wikipedia, sentence-split) and the 6-gram:
python scripts/collect_lm_text.py --languages lin --merge-corpus-dir data/lm_expanded \
  --skip-source google/fleurs --skip-source Sunbird/salt \
  --skip-source yigagilbert/luganda-speech-cv-yogera-filtered --skip-source DigitalUmuganda/Afrivoice \
  --max-lines-per-source 60000 --output-dir data/lm_expanded_v2_o6 --order 6 --overwrite
# phase-2 combined dir (kenlm binaries self-describe order; filename is just lookup):
mkdir -p data/lm_phase2
cp data/lm_expanded/{lug_5gram.binary,lug.txt,sna_5gram.binary,sna.txt} data/lm_phase2/
cp data/lm_expanded_v2_o6/lin_6gram.binary data/lm_phase2/lin_5gram.binary
cp data/lm_expanded_v2_o6/lin.txt data/lm_phase2/
mkdir -p outputs/analysis && cat > outputs/analysis/best_decode_params.json <<'JSON'
{"lin": {"alpha": 0.8, "beta": 0.75}, "lug": {"alpha": 0.4, "beta": -0.5}, "sna": {"alpha": 0.7, "beta": -0.5}}
JSON
# IMMEDIATELY make them durable this time:
python -c "from huggingface_hub import HfApi; a=HfApi(); a.upload_folder(folder_path='data/lm_expanded', path_in_repo='lm_expanded', repo_id='yigagilbert/waxal-xlsr300m-champion', repo_type='model'); a.upload_folder(folder_path='data/lm_expanded_v2_o6', path_in_repo='lm_expanded_v2_o6', repo_id='yigagilbert/waxal-xlsr300m-champion', repo_type='model'); a.upload_folder(folder_path='data/lm_phase2', path_in_repo='lm_phase2', repo_id='yigagilbert/waxal-xlsr300m-champion', repo_type='model')"
```
Sanity: rebuilt lin corpus ≈ 109k–127k lines / ~81k types; spot-check the champion decode on
lin validation reproduces combined ≈ 0.168 (α0.8/β0.75, o6, beam 400) before trusting the rebuild.

## 6. Per-language test decodes for future merges (~40 min GPU, or skip)

Either regenerate lug/sna test decodes (`run_xlsr_inference.py --split test --languages lug|sna
--decoder-mode beam_lm --kenlm-model data/lm_expanded/<l>_5gram.binary --unigrams-file
data/lm_expanded/<l>.txt` with lug α0.4/β−0.5, sna α0.7/β−0.5, beam 400), or split the
already-final `champ_lmv2_o6.csv` (on the Mac / Zindi) by ID prefix — those rows are already
postprocessed, so only reuse them whole, not through the postprocess step again.

## 7. Current decision point (2026-07-27)

Run the whisper-af51 teacher evaluation FIRST (thresholds in
[FINAL_TOP3_STRATEGY.md §6b](FINAL_TOP3_STRATEGY.md)):
```bash
python scripts/run_whisper_inference.py --model-name huwenjie333/whisper-v3-ft-af51 \
  --dataset-dir data/processed_generalization_mix --split validation --languages lin \
  --num-beams 1 --max-new-tokens 200 --batch-size 8 \
  --output outputs/predictions/whisper_af51_lin_val.csv
python scripts/evaluate_predictions.py --predictions outputs/predictions/whisper_af51_lin_val.csv \
  --references data/processed_generalization_mix/validation.csv --normalization language_safe \
  --output outputs/analysis/whisper_af51_lin_eval.json
```
< 0.168 → route lin to it (no training). 0.168–0.22 → af51 as pseudo-label teacher for
unlabeled WAXAL lin. > ~0.26 or hallucinating → discard, hold the 0.8622 floor.
