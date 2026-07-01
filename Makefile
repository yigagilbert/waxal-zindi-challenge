.PHONY: audit prepare-metadata check-env check-gpu prepare-tiny whisper-tiny eval xlsr-smoke whisper-smoke clean-pyc

RAW_DIR ?= $(WAXAL_RAW_DIR)
RAW_ARG := $(if $(RAW_DIR),--raw-dir "$(RAW_DIR)",)
DATASET_DIR ?= data/processed
SMOKE_DATASET_DIR ?= data/processed_smoke
PREDICTIONS ?= outputs/predictions/whisper_tiny_validation.csv
MODEL ?= openai/whisper-large-v3-turbo

audit:
	uv run scripts/audit_data.py $(RAW_ARG) --output outputs/data_audit.json

prepare-metadata:
	uv run scripts/prepare_dataset.py $(RAW_ARG) --output-dir "$(DATASET_DIR)" --metadata-only

check-env:
	uv run scripts/check_gpu_env.py $(RAW_ARG)

check-gpu:
	uv run scripts/check_gpu_env.py $(RAW_ARG) --require-gpu --min-free-gb 100

prepare-tiny:
	uv run scripts/prepare_dataset.py $(RAW_ARG) --output-dir "$(SMOKE_DATASET_DIR)" --streaming --max-per-language-split 3

whisper-tiny:
	uv run scripts/run_whisper_inference.py --model-name "$(MODEL)" --dataset-dir "$(SMOKE_DATASET_DIR)" --split validation --max-samples 3 --output "$(PREDICTIONS)"

eval:
	uv run scripts/evaluate_predictions.py --predictions "$(PREDICTIONS)" --references "$(SMOKE_DATASET_DIR)/validation.csv" --normalization all

xlsr-smoke:
	uv run scripts/train_xlsr_ctc.py --config configs/xlsr_300m.yaml --dataset-dir "$(SMOKE_DATASET_DIR)" --max-train-samples 6 --max-eval-samples 3 --max-steps 2 --output-dir checkpoints/xlsr_300m_smoke

whisper-smoke:
	uv run scripts/train_whisper.py --config configs/whisper_medium_lora.yaml --dataset-dir "$(SMOKE_DATASET_DIR)" --max-train-samples 3 --max-eval-samples 3 --max-steps 2 --output-dir checkpoints/whisper_medium_lora_smoke

clean-pyc:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
