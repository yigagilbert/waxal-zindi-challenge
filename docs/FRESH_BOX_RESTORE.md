# Fresh-Box Restore Runbook

## ⚡ Phase-2 quickstart (use this from 2026-07-28 on; full Phase-1 restore below is only for champion work)

State as of 2026-07-27 night: **rank 2 @ 0.6817** (`phase2_af51_beam5.csv`, SELECTED); greedy was
0.6773; #1 Sophey 0.7087 (gap 0.0270). Engine: `huwenjie333/whisper-v3-ft-af51` (Phase 2 =
unseen Ugandan languages: Acholi/Lango, Runyankole-Rukiga, Lusoga, Lumasaba).

Fresh box to decoding in ~20 min:
```bash
git clone https://github.com/yigagilbert/waxal-zindi-challenge && cd waxal-zindi-challenge
uv pip install --system "datasets>=3.0,<4" soundfile librosa "transformers>=4.46" torch \
  pyctcdecode kenlm huggingface_hub peft accelerate && huggingface-cli login
mkdir -p data/phase2 && cd data/phase2 \
  && curl -LO https://storage.googleapis.com/waxalphase2/audio.zip && unzip -q audio.zip && cd ../..
# Test_phase2.csv: scp from Mac, or pull from the artifacts repo:
python -c "from huggingface_hub import hf_hub_download; import shutil; shutil.copy(hf_hub_download('yigagilbert/waxal-private-artifacts','phase2/Test_phase2.csv',repo_type='dataset'),'data/phase2/Test_phase2.csv')"
python scripts/prepare_phase2_test.py --audio-dir data/phase2/audio --test-csv data/phase2/Test_phase2.csv
# af51 downloads itself on first inference. All 07-27 outputs live in
# yigagilbert/waxal-private-artifacts under phase2/ (predictions, submissions, analysis, audio_lid).
```

Refinement ladder (one variable per submission, 5/day, close 2026-08-03):
1. ~~Raw-text A/B~~ — DONE 07-28: **0.6835** (beat normalized 0.6817). Raw punctuation/casing
   is canonical for all Phase-2 submissions; references contain punctuation. Rank 8, #1 = 0.7177.
2. **LID → language-forced decoding** (current rung): per-clip language codes from
   `scripts/cluster_phase2_languages.py` (map: ach 477 / nyn 401 / myx 267 / xog 84 / unk 271;
   routing table at `yigagilbert/waxal-private-artifacts` → `phase2/analysis/phase2_language_clusters.csv`).
   Primary engine: **`Sunbird/asr-whisper-large-v3-salt`** (gated; terms accepted 07-28) —
   supports ALL four Phase-2 clusters via repurposed stock Whisper language slots
   (SALT_LANGUAGE_TOKENS_WHISPER): ach=50357, nyn=50354, xog=50352, myx=50349
   (also lug=50355, teo=50353, lgg=50356, ttj=50351, kin=50350, swa=50318, eng=50259).
   Pass raw token ids through `--language-map`:
   ```bash
   python -c "from huggingface_hub import hf_hub_download; import shutil; shutil.copy(hf_hub_download('yigagilbert/waxal-private-artifacts','phase2/analysis/phase2_language_clusters.csv',repo_type='dataset'),'outputs/analysis/phase2_language_clusters.csv')"
   python scripts/run_whisper_inference.py --model-name Sunbird/asr-whisper-large-v3-salt \
     --dataset-dir data/phase2_processed --split test --num-beams 5 --max-new-tokens 220 --batch-size 8 \
     --language-csv outputs/analysis/phase2_language_clusters.csv \
     --language-map ach=50357 nyn=50354 xog=50352 myx=50349 \
     --output outputs/predictions/phase2_salt_forced_beam5_raw.csv
   ```
   `unk` rows auto-detect (risky on a repurposed-token model — see splice below). Secondary
   engine A/B: `Sunbird/asr-whisper-51-african-languages` same command (check its code
   convention on the box first — tokenizer added tokens vs SALT-style raw ids). NO
   normalization; merge with `merge_predictions.py --order`, wc -l 1501, submit. Both models
   disclosed in docs/RULES_AND_DATA_USE.md.
3. **Splice / per-cluster routing** (`scripts/splice_predictions.py`): SALT-forced overlay for
   clustered clips + af51 base for `unk` clips costs no extra GPU time; later route each
   cluster to whichever engine wins it (native-speaker eyeball proved decisive before).
4. **In-domain fine-tune** (CONFIRMED possible 07-28: WaxalNLP HAS train configs for the
   Phase-2 languages — ach_asr, nyn_asr, xog_asr, myx_asr): fine-tune the SALT/51-lang engine
   on those splits with `scripts/train_whisper.py` — in-domain beat external transfer every
   time in Phase 1; likely the endgame lever. Start early: deadline 2026-08-03.
5. Beam 8–10 / temperature-fallback decode; length-penalty tuning.


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
huggingface-cli login        # token with access to gated google/WaxalNLP + private repos
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
