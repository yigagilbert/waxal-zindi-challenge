# Changes Log

## Vast.ai Restart Fixes

### Whisper Padding and Truncation

`scripts/run_whisper_inference.py` now calls the processor with:

```python
padding="max_length"
truncation=True
return_attention_mask=True
```

This prevents short batches from producing mel features shorter than Whisper's expected 3000-frame input.

### Whisper Dtype Casting

Whisper inference now casts `input_features` to the model parameter dtype before generation:

```python
model_dtype = next(model.parameters()).dtype
input_features = inputs["input_features"].to(device=device, dtype=model_dtype)
```

This fixes float32 input features being passed into a float16 CUDA model.

### RTX 5090 PyTorch Repair

The previous RTX 5090 instance failed with the locked `torch==2.6.0+cu124` build because RTX 5090 requires support for compute capability `sm_120`.

The working repair was:

```bash
uv pip uninstall --python .venv/bin/python torch torchaudio torchvision -y
uv pip install --python .venv/bin/python --upgrade torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

After repair, use `uv run --no-sync` or `WAXAL_NO_SYNC=1 make ...` so uv does not restore the locked torch version.

`scripts/check_gpu_env.py` now checks the compiled CUDA architecture list and runs a real CUDA matmul smoke test. This catches the RTX 5090 failure before inference or training starts.

### Tiny Prediction Evaluation Caveat

Tiny inference may generate only a few predictions. Evaluate those against the matching tiny reference file:

```bash
data/processed_smoke/validation.csv
```

Do not evaluate tiny predictions against:

```bash
data/processed/validation.csv
```

That produces many expected `missing_predictions` and is not a model failure.

### Staged Data Preparation

`scripts/prepare_dataset.py` now supports selected splits:

```bash
uv run scripts/prepare_dataset.py --splits validation
uv run scripts/prepare_dataset.py --splits train
uv run scripts/prepare_dataset.py --splits test
```

When `data/processed/hf_dataset` already exists, unrequested cached splits are preserved.

### Per-Language Evaluation

`scripts/evaluate_predictions.py` now supports `--languages`, so Luganda-only Sunbird predictions can be scored against Luganda references without reporting Lingala and Shona as missing.

### Whisper LoRA PEFT Wrapper

`scripts/train_whisper.py` no longer defaults LoRA to `task_type="SEQ_2_SEQ_LM"`. That PEFT wrapper sends `input_ids` into the model, but Whisper training uses `input_features`. The generic LoRA wrapper preserves Whisper's audio-input forward path.
