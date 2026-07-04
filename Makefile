.PHONY: audit prepare-metadata check-env check-gpu restart-check prepare-tiny prepare-validation prepare-train prepare-test prepare-test-fast whisper-tiny sunbird-lug-tiny eval eval-tiny eval-sunbird-lug-tiny sunbird-lug-validation whisper-turbo-validation whisper-large-validation eval-sunbird-lug eval-whisper-turbo audit-audio-train audit-audio-validation teacher-sunbird-lug-validation teacher-sunbird-lug-train teacher-whisper-turbo-validation build-quality-buckets clean-luganda-with-sunbird train-xlsr-clean-smoke train-xlsr-clean-all train-xlsr-sunbird-lug-smoke train-xlsr-sunbird-lug-all train-xlsr-sunbird-lug-v2-smoke train-xlsr-sunbird-lug-v2-all xlsr-sunbird-lug-validation eval-xlsr-sunbird-lug xlsr-sunbird-lug-v2-validation eval-xlsr-sunbird-lug-v2 eval-xlsr-clean-all xlsr-smoke whisper-smoke backup-artifacts clean-pyc

RAW_DIR ?= $(WAXAL_RAW_DIR)
RAW_ARG := $(if $(RAW_DIR),--raw-dir "$(RAW_DIR)",)
DATASET_DIR ?= data/processed
SMOKE_DATASET_DIR ?= data/processed_smoke
PREDICTIONS ?= outputs/predictions/whisper_tiny_validation.csv
MODEL ?= openai/whisper-large-v3-turbo
BACKUP_DIR ?= ../waxal_artifact_backup
TEACHER_PREDICTIONS ?=
XLSR_CLEAN_PREDICTIONS ?= outputs/predictions/xlsr_300m_balanced_clean_all_validation.csv
XLSR_SUNBIRD_CHECKPOINT ?= checkpoints/xlsr_300m_balanced_sunbird_lug_all/checkpoint-6000
XLSR_SUNBIRD_PREDICTIONS ?= outputs/predictions/xlsr_300m_balanced_sunbird_lug_checkpoint6000_validation.csv
XLSR_SUNBIRD_V2_CHECKPOINT ?= checkpoints/xlsr_300m_balanced_sunbird_lug_all_v2/checkpoint-6000
XLSR_SUNBIRD_V2_PREDICTIONS ?= outputs/predictions/xlsr_300m_balanced_sunbird_lug_v2_checkpoint6000_validation.csv
UV_RUN := uv run $(if $(WAXAL_NO_SYNC),--no-sync,)

audit:
	$(UV_RUN) scripts/audit_data.py $(RAW_ARG) --output outputs/data_audit.json

prepare-metadata:
	$(UV_RUN) scripts/prepare_dataset.py $(RAW_ARG) --output-dir "$(DATASET_DIR)" --metadata-only

check-env:
	$(UV_RUN) scripts/check_gpu_env.py $(RAW_ARG)

check-gpu:
	$(UV_RUN) scripts/check_gpu_env.py $(RAW_ARG) --require-gpu --min-free-gb 100

restart-check: check-gpu

prepare-tiny:
	$(UV_RUN) scripts/prepare_dataset.py $(RAW_ARG) --output-dir "$(SMOKE_DATASET_DIR)" --streaming --max-per-language-split 3

prepare-validation:
	$(UV_RUN) scripts/prepare_dataset.py $(RAW_ARG) --output-dir "$(DATASET_DIR)" --splits validation

prepare-train:
	$(UV_RUN) scripts/prepare_dataset.py $(RAW_ARG) --output-dir "$(DATASET_DIR)" --splits train

prepare-test:
	$(UV_RUN) scripts/prepare_dataset.py $(RAW_ARG) --output-dir "$(DATASET_DIR)" --splits test

prepare-test-fast:
	$(UV_RUN) scripts/prepare_dataset.py $(RAW_ARG) --output-dir "$(DATASET_DIR)" --splits test --skip-duration

whisper-tiny:
	$(UV_RUN) scripts/run_whisper_inference.py --model-name "$(MODEL)" --dataset-dir "$(SMOKE_DATASET_DIR)" --split validation --max-samples 3 --output "$(PREDICTIONS)"

sunbird-lug-tiny:
	$(UV_RUN) scripts/run_whisper_inference.py --model-name "Sunbird/asr-whisper-large-v3-salt" --dataset-dir "$(SMOKE_DATASET_DIR)" --split validation --languages lug --max-samples 3 --output outputs/predictions/sunbird_lug_tiny_validation.csv

eval-tiny:
	$(UV_RUN) scripts/evaluate_predictions.py --predictions "$(PREDICTIONS)" --references "$(SMOKE_DATASET_DIR)/validation.csv" --normalization all

eval: eval-tiny

eval-sunbird-lug-tiny:
	$(UV_RUN) scripts/evaluate_predictions.py --predictions outputs/predictions/sunbird_lug_tiny_validation.csv --references "$(SMOKE_DATASET_DIR)/validation.csv" --languages lug --normalization all --output outputs/experiments/sunbird_lug_tiny_all_norms.json

sunbird-lug-validation:
	$(UV_RUN) scripts/run_whisper_inference.py --model-name "Sunbird/asr-whisper-large-v3-salt" --dataset-dir "$(DATASET_DIR)" --split validation --languages lug --output outputs/predictions/sunbird_whisper_lug_validation.csv

whisper-turbo-validation:
	$(UV_RUN) scripts/run_whisper_inference.py --model-name "openai/whisper-large-v3-turbo" --dataset-dir "$(DATASET_DIR)" --split validation --output outputs/predictions/whisper_turbo_validation.csv

whisper-large-validation:
	$(UV_RUN) scripts/run_whisper_inference.py --model-name "openai/whisper-large-v3" --dataset-dir "$(DATASET_DIR)" --split validation --output outputs/predictions/whisper_large_v3_validation.csv

eval-sunbird-lug:
	$(UV_RUN) scripts/evaluate_predictions.py --predictions outputs/predictions/sunbird_whisper_lug_validation.csv --references "$(DATASET_DIR)/validation.csv" --languages lug --normalization all --output outputs/experiments/sunbird_whisper_lug_validation_all_norms.json

eval-whisper-turbo:
	$(UV_RUN) scripts/evaluate_predictions.py --predictions outputs/predictions/whisper_turbo_validation.csv --references "$(DATASET_DIR)/validation.csv" --normalization all --output outputs/experiments/whisper_turbo_validation_all_norms.json

audit-audio-train:
	$(UV_RUN) scripts/audit_audio_quality.py --dataset-dir "$(DATASET_DIR)" --split train --output outputs/quality/audio_quality_train.csv

audit-audio-validation:
	$(UV_RUN) scripts/audit_audio_quality.py --dataset-dir "$(DATASET_DIR)" --split validation --output outputs/quality/audio_quality_validation.csv

teacher-sunbird-lug-validation:
	$(UV_RUN) scripts/run_teacher_inference.py --model-name "Sunbird/asr-whisper-large-v3-salt" --dataset-dir "$(DATASET_DIR)" --split validation --languages lug --output outputs/teachers/sunbird_whisper_lug_validation.csv

teacher-sunbird-lug-train:
	$(UV_RUN) scripts/run_teacher_inference.py --model-name "Sunbird/asr-whisper-large-v3-salt" --dataset-dir "$(DATASET_DIR)" --split train --languages lug --output outputs/teachers/sunbird_whisper_lug_train.csv

teacher-whisper-turbo-validation:
	$(UV_RUN) scripts/run_teacher_inference.py --model-name "openai/whisper-large-v3-turbo" --dataset-dir "$(DATASET_DIR)" --split validation --output outputs/teachers/whisper_turbo_validation.csv

build-quality-buckets:
	$(UV_RUN) scripts/build_quality_buckets.py --audio-quality outputs/quality/audio_quality_train.csv --metadata "$(DATASET_DIR)/train.csv" $(TEACHER_PREDICTIONS) --normalization language_safe --output-dir data/quality --summary-output outputs/quality/quality_bucket_summary.json

clean-luganda-with-sunbird:
	$(UV_RUN) scripts/clean_luganda_with_teacher.py --metadata "$(DATASET_DIR)/train.csv" --audio-quality outputs/quality/audio_quality_train.csv --teacher-predictions outputs/teachers/sunbird_whisper_lug_train.csv --base-clean-manifest data/quality/clean_train.csv --output-dir data/quality --summary-output outputs/quality/luganda_teacher_cleaning_summary.json --teacher-label-mode high_disagreement

train-xlsr-clean-smoke:
	$(UV_RUN) scripts/train_xlsr_ctc.py --config configs/xlsr_300m_balanced_clean_all.yaml --dataset-dir "$(DATASET_DIR)" --max-train-samples 12 --max-eval-samples 6 --max-steps 2 --output-dir checkpoints/xlsr_300m_balanced_clean_all_smoke

train-xlsr-clean-all:
	$(UV_RUN) scripts/train_xlsr_ctc.py --config configs/xlsr_300m_balanced_clean_all.yaml --dataset-dir "$(DATASET_DIR)"

train-xlsr-sunbird-lug-smoke:
	$(UV_RUN) scripts/train_xlsr_ctc.py --config configs/xlsr_300m_balanced_sunbird_lug_all.yaml --dataset-dir "$(DATASET_DIR)" --max-train-samples 12 --max-eval-samples 6 --max-steps 2 --output-dir checkpoints/xlsr_300m_balanced_sunbird_lug_smoke

train-xlsr-sunbird-lug-all:
	$(UV_RUN) scripts/train_xlsr_ctc.py --config configs/xlsr_300m_balanced_sunbird_lug_all.yaml --dataset-dir "$(DATASET_DIR)"

train-xlsr-sunbird-lug-v2-smoke:
	$(UV_RUN) scripts/train_xlsr_ctc.py --config configs/xlsr_300m_balanced_sunbird_lug_all_v2.yaml --dataset-dir "$(DATASET_DIR)" --max-train-samples 12 --max-eval-samples 6 --max-steps 2 --output-dir checkpoints/xlsr_300m_balanced_sunbird_lug_v2_smoke

train-xlsr-sunbird-lug-v2-all:
	$(UV_RUN) scripts/train_xlsr_ctc.py --config configs/xlsr_300m_balanced_sunbird_lug_all_v2.yaml --dataset-dir "$(DATASET_DIR)"

xlsr-sunbird-lug-validation:
	$(UV_RUN) scripts/run_xlsr_inference.py --checkpoint "$(XLSR_SUNBIRD_CHECKPOINT)" --dataset-dir "$(DATASET_DIR)" --split validation --batch-size 2 --output "$(XLSR_SUNBIRD_PREDICTIONS)"

eval-xlsr-sunbird-lug:
	$(UV_RUN) scripts/evaluate_predictions.py --predictions "$(XLSR_SUNBIRD_PREDICTIONS)" --references "$(DATASET_DIR)/validation.csv" --normalization all --output outputs/experiments/xlsr_300m_balanced_sunbird_lug_checkpoint6000_validation_all_norms.json

xlsr-sunbird-lug-v2-validation:
	$(UV_RUN) scripts/run_xlsr_inference.py --checkpoint "$(XLSR_SUNBIRD_V2_CHECKPOINT)" --dataset-dir "$(DATASET_DIR)" --split validation --batch-size 2 --output "$(XLSR_SUNBIRD_V2_PREDICTIONS)"

eval-xlsr-sunbird-lug-v2:
	$(UV_RUN) scripts/evaluate_predictions.py --predictions "$(XLSR_SUNBIRD_V2_PREDICTIONS)" --references "$(DATASET_DIR)/validation.csv" --normalization all --output outputs/experiments/xlsr_300m_balanced_sunbird_lug_v2_checkpoint6000_validation_all_norms.json

eval-xlsr-clean-all:
	$(UV_RUN) scripts/evaluate_predictions.py --predictions "$(XLSR_CLEAN_PREDICTIONS)" --references "$(DATASET_DIR)/validation.csv" --normalization all --output outputs/experiments/xlsr_300m_balanced_clean_all_validation_all_norms.json

xlsr-smoke:
	$(UV_RUN) scripts/train_xlsr_ctc.py --config configs/xlsr_300m.yaml --dataset-dir "$(SMOKE_DATASET_DIR)" --max-train-samples 6 --max-eval-samples 3 --max-steps 2 --output-dir checkpoints/xlsr_300m_smoke

whisper-smoke:
	$(UV_RUN) scripts/train_whisper.py --config configs/whisper_medium_lora.yaml --dataset-dir "$(SMOKE_DATASET_DIR)" --max-train-samples 3 --max-eval-samples 3 --max-steps 2 --output-dir checkpoints/whisper_medium_lora_smoke

backup-artifacts:
	mkdir -p "$(BACKUP_DIR)"
	rsync -a outputs/predictions/ outputs/experiments/ outputs/submissions/ checkpoints/ "$(BACKUP_DIR)"/
	rsync -a data/processed/prepare_report.json scripts/ configs/ docs/ README.md pyproject.toml uv.lock Makefile "$(BACKUP_DIR)"/

clean-pyc:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
