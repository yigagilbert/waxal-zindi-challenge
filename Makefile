.PHONY: audit prepare-metadata check-env check-gpu restart-check prepare-tiny prepare-validation prepare-train prepare-test prepare-test-fast whisper-tiny sunbird-lug-tiny eval eval-tiny eval-sunbird-lug-tiny sunbird-lug-validation whisper-turbo-validation whisper-large-validation eval-sunbird-lug eval-whisper-turbo audit-audio-train audit-audio-validation teacher-sunbird-lug-validation teacher-sunbird-lug-train teacher-whisper-turbo-validation build-quality-buckets clean-luganda-with-sunbird train-xlsr-clean-smoke train-xlsr-clean-all train-xlsr-sunbird-lug-smoke train-xlsr-sunbird-lug-all train-xlsr-sunbird-lug-v2-smoke train-xlsr-sunbird-lug-v2-all xlsr-sunbird-lug-validation eval-xlsr-sunbird-lug xlsr-sunbird-lug-v2-validation eval-xlsr-sunbird-lug-v2 xlsr-v2-validation-all analyze-xlsr-v2-validation-all xlsr-v2-test-all analyze-xlsr-v2-test-all submissions-xlsr-v2-all alvin-lingala-access alvin-lingala-validation compare-alvin-lingala-validation alvin-lingala-train-teacher lingala-alvin-diagnostics build-lingala-alvin-manifest alvin-lingala-test compare-alvin-lingala-test audit-lingala-100hrs train-alvin-lingala-smoke train-alvin-lingala eval-xlsr-clean-all xlsr-smoke whisper-smoke backup-artifacts clean-pyc

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
XLSR_V2_BASE ?= checkpoints/xlsr_300m_balanced_sunbird_lug_all_v2
XLSR_V2_CKPT5000_VALID ?= outputs/predictions/xlsr_v2_ckpt5000_validation.csv
XLSR_V2_CKPT5500_VALID ?= outputs/predictions/xlsr_v2_ckpt5500_validation.csv
XLSR_V2_CKPT6000_VALID ?= outputs/predictions/xlsr_v2_ckpt6000_validation.csv
XLSR_V2_CKPT5000_TEST ?= outputs/predictions/xlsr_v2_ckpt5000_test.csv
XLSR_V2_CKPT5500_TEST ?= outputs/predictions/xlsr_v2_ckpt5500_test.csv
XLSR_V2_CKPT6000_TEST ?= outputs/predictions/xlsr_v2_ckpt6000_test.csv
ALVIN_LINGALA_MODEL ?= Alvin-Nahabwe/wav2vec2-xls-r-300m-Fleurs_AMMI_AFRIVOICE_LRSC-ln-109hrs-v2
NOIRLAB_LINGALA_MODEL ?= noirlab/whisper-large-v3-lingala-asr
BRAINTHEOS_LINGALA_MODEL ?= BrainTheos/wav2vec2-large-mms-1b-all-lingala-ojpl
ALVIN_LINGALA_VALID ?= outputs/lingala_models/alvin_xlsr_lingala_validation.csv
ALVIN_LINGALA_TEST ?= outputs/lingala_models/alvin_xlsr_lingala_test.csv
ALVIN_LINGALA_TRAIN ?= outputs/lingala_models/alvin_xlsr_lingala_train.csv
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

xlsr-v2-validation-all:
	$(UV_RUN) scripts/run_xlsr_inference.py --checkpoint "$(XLSR_V2_BASE)/checkpoint-5000" --dataset-dir "$(DATASET_DIR)" --split validation --batch-size 2 --output "$(XLSR_V2_CKPT5000_VALID)"
	$(UV_RUN) scripts/run_xlsr_inference.py --checkpoint "$(XLSR_V2_BASE)/checkpoint-5500" --dataset-dir "$(DATASET_DIR)" --split validation --batch-size 2 --output "$(XLSR_V2_CKPT5500_VALID)"
	$(UV_RUN) scripts/run_xlsr_inference.py --checkpoint "$(XLSR_V2_BASE)/checkpoint-6000" --dataset-dir "$(DATASET_DIR)" --split validation --batch-size 2 --output "$(XLSR_V2_CKPT6000_VALID)"

analyze-xlsr-v2-validation-all:
	$(UV_RUN) scripts/analyze_prediction_distributions.py --predictions "$(XLSR_V2_CKPT5000_VALID)" "$(XLSR_V2_CKPT5500_VALID)" "$(XLSR_V2_CKPT6000_VALID)" --names ckpt5000 ckpt5500 ckpt6000 --references "$(DATASET_DIR)/validation.csv" --normalization all --output outputs/analysis/xlsr_v2_validation_checkpoint_analysis.json

xlsr-v2-test-all:
	$(UV_RUN) scripts/run_xlsr_inference.py --checkpoint "$(XLSR_V2_BASE)/checkpoint-5000" --dataset-dir "$(DATASET_DIR)" --split test --batch-size 2 --output "$(XLSR_V2_CKPT5000_TEST)"
	$(UV_RUN) scripts/run_xlsr_inference.py --checkpoint "$(XLSR_V2_BASE)/checkpoint-5500" --dataset-dir "$(DATASET_DIR)" --split test --batch-size 2 --output "$(XLSR_V2_CKPT5500_TEST)"
	$(UV_RUN) scripts/run_xlsr_inference.py --checkpoint "$(XLSR_V2_BASE)/checkpoint-6000" --dataset-dir "$(DATASET_DIR)" --split test --batch-size 2 --output "$(XLSR_V2_CKPT6000_TEST)"

analyze-xlsr-v2-test-all:
	$(UV_RUN) scripts/analyze_prediction_distributions.py --predictions "$(XLSR_V2_CKPT5000_TEST)" "$(XLSR_V2_CKPT5500_TEST)" "$(XLSR_V2_CKPT6000_TEST)" --names ckpt5000 ckpt5500 ckpt6000 --output outputs/analysis/xlsr_v2_test_checkpoint_analysis.json

submissions-xlsr-v2-all:
	$(UV_RUN) scripts/make_submission.py --predictions "$(XLSR_V2_CKPT5000_TEST)" $(RAW_ARG) --model-name xlsr_v2_ckpt5000 --output outputs/submissions/submission_xlsr_v2_ckpt5000.csv
	$(UV_RUN) scripts/make_submission.py --predictions "$(XLSR_V2_CKPT5500_TEST)" $(RAW_ARG) --model-name xlsr_v2_ckpt5500 --output outputs/submissions/submission_xlsr_v2_ckpt5500.csv
	$(UV_RUN) scripts/make_submission.py --predictions "$(XLSR_V2_CKPT6000_TEST)" $(RAW_ARG) --model-name xlsr_v2_ckpt6000 --output outputs/submissions/submission_xlsr_v2_ckpt6000.csv

alvin-lingala-access:
	$(UV_RUN) scripts/check_hf_asr_model_access.py --model-name "$(ALVIN_LINGALA_MODEL)" --dataset-dir "$(DATASET_DIR)" --split validation --languages lin --max-samples 3 --output outputs/lingala_models/alvin_access_check.json

alvin-lingala-validation:
	$(UV_RUN) scripts/run_lingala_model_inference.py --model-name "$(ALVIN_LINGALA_MODEL)" --dataset-dir "$(DATASET_DIR)" --split validation --batch-size 2 --output "$(ALVIN_LINGALA_VALID)" --token-stats-output outputs/lingala_models/alvin_xlsr_lingala_validation_token_stats.csv --overwrite

compare-alvin-lingala-validation:
	$(UV_RUN) scripts/compare_lingala_models.py --predictions "$(XLSR_V2_CKPT6000_VALID)" "$(ALVIN_LINGALA_VALID)" --names xlsr_v2_ckpt6000 alvin_xlsr_lingala --references "$(DATASET_DIR)/validation.csv" --audio-quality outputs/quality/audio_quality_validation.csv --normalization language_safe --baseline-name xlsr_v2_ckpt6000 --output outputs/lingala_models/lingala_model_comparison_metrics.json --markdown-output docs/LINGALA_MODEL_COMPARISON.md

alvin-lingala-train-teacher:
	$(UV_RUN) scripts/run_lingala_model_inference.py --model-name "$(ALVIN_LINGALA_MODEL)" --dataset-dir "$(DATASET_DIR)" --split train --batch-size 2 --output "$(ALVIN_LINGALA_TRAIN)" --token-stats-output outputs/lingala_models/alvin_xlsr_lingala_train_token_stats.csv --overwrite

lingala-alvin-diagnostics:
	$(UV_RUN) scripts/build_lingala_teacher_diagnostics.py --metadata "$(DATASET_DIR)/train.csv" --audio-quality outputs/quality/audio_quality_train.csv --teacher-predictions "$(ALVIN_LINGALA_TRAIN)" --output outputs/lingala_teacher/alvin_teacher_disagreement_lingala_train.csv --summary-output outputs/lingala_teacher/alvin_teacher_disagreement_lingala_train.summary.json

build-lingala-alvin-manifest:
	$(UV_RUN) scripts/build_lingala_alvin_manifest.py --base-clean-manifest data/quality/clean_train_sunbird_lug.csv --metadata "$(DATASET_DIR)/train.csv" --teacher-diagnostics outputs/lingala_teacher/alvin_teacher_disagreement_lingala_train.csv --output-dir data/quality --summary-output outputs/lingala_teacher/alvin_lingala_bucket_summary.json

alvin-lingala-test:
	$(UV_RUN) scripts/run_lingala_model_inference.py --model-name "$(ALVIN_LINGALA_MODEL)" --dataset-dir "$(DATASET_DIR)" --split test --batch-size 2 --output "$(ALVIN_LINGALA_TEST)" --token-stats-output outputs/lingala_models/alvin_xlsr_lingala_test_token_stats.csv --overwrite

compare-alvin-lingala-test:
	$(UV_RUN) scripts/compare_lingala_models.py --predictions "$(XLSR_V2_CKPT6000_TEST)" "$(ALVIN_LINGALA_TEST)" --names xlsr_v2_ckpt6000 alvin_xlsr_lingala --normalization language_safe --baseline-name xlsr_v2_ckpt6000 --output outputs/analysis/alvin_vs_xlsr6000_lingala_test_sanity.json --markdown-output docs/LINGALA_MODEL_COMPARISON.md

audit-lingala-100hrs:
	$(UV_RUN) scripts/audit_lingala_100hrs.py --streaming --max-rows-per-split 500 --metadata "$(DATASET_DIR)/train.csv" --test-ids "$(DATASET_DIR)/test.csv" --output outputs/lingala_external/lingala_100hrs_audit.json --source-stats-output outputs/lingala_external/lingala_100hrs_source_stats.csv

train-alvin-lingala-smoke:
	$(UV_RUN) scripts/train_xlsr_ctc.py --config configs/xlsr_300m_alvin_lingala_waxal_finetune.yaml --dataset-dir "$(DATASET_DIR)" --max-train-samples 12 --max-eval-samples 6 --max-steps 2 --output-dir checkpoints/xlsr_300m_alvin_lingala_waxal_smoke

train-alvin-lingala:
	$(UV_RUN) scripts/train_xlsr_ctc.py --config configs/xlsr_300m_alvin_lingala_waxal_finetune.yaml --dataset-dir "$(DATASET_DIR)"

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
