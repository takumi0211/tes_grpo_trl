# TES GRPO Training Pipeline

## Project Overview
This repository contains the tooling used to fine-tune the `openai/gpt-oss-20b` model with Generalized Reinforced Policy Optimization (GRPO). The training recipe combines MXFP4 checkpoint loading, LoRA adapters, reward-weighted sampling, and TRL's GRPO trainer to produce a lightweight adapter that can be merged back into the base model or served directly via PEFT. Supporting scripts cover dataset conversion, reward shaping, adapter testing, quantized export, and downstream inference.

## Key Features
- Single-GPU GRPO fine-tuning flow targeting GPT-OSS 20B with MXFP4 dequantization to bfloat16 during training.
- LoRA configuration using the PEFT library with `target_modules="all-linear"` for broad coverage of linear layers.
- Streaming prompt sampler (`StepStream`) that draws diverse prompts and reshapes reward tensors for TRL.
- Reward computation utilities that transform Q-value columns into normalized reward distributions.
- Tooling to convert legacy prompts into Harmony chat format expected by GPT-OSS chat templates.
- Export and inference scripts for both merged full models and adapter-only deployments (Transformers, Hugging Face Hub, Ollama).

## Repository Layout
- `train_grpo.py` — Main training entry point; defines model, tokenizer, GRPO configuration, and training loop.
- `data/` — Example prompt datasets (Harmony formatted CSVs) and auxiliary prompt files.
- `data_reward.py` — Dataset loader, reward function implementation, and optional completion logging to CSV.
- `step_stream.py` — Iterable dataset feeding prompts and reward tensors into GRPOTrainer.
- `calculate_reward.py` — Converts `q_action_*` columns to `reward_action_*` via Boltzmann softmax.
- `convert_to_harmony.py` — Converts legacy CSV prompts into Harmony chat prompts using the model's template.
- `test_lora_adapter.py` / `test_lora_adapter_quantized.py` — Smoke tests for evaluating adapters with and without MXFP4 quantization.
- `export_quantized_model.py` — Merges a trained LoRA adapter into the base model, attaches quantization metadata, and optionally uploads to Hugging Face Hub.
- `run_exported_model.py` — Runs text generation against a merged model artifact.
- `Modelfile` — Example Ollama configuration for attaching a LoRA adapter to a GPT-OSS base image.
- `runs/` — Default output directory for new GRPO checkpoints, trainer states, and tokenizer artifacts.

## Prerequisites
- **Hardware:** Hopper-class GPU (e.g., NVIDIA H100) with >= 80 GB VRAM recommended for MXFP4 + BF16 training.
- **Operating system:** Linux with CUDA-capable drivers (tested with NVIDIA driver ≥ 570 and CUDA 12.8).
- **Python:** 3.11 or 3.12.
- **Dependencies:** PyTorch 2.8.0 (CUDA 12.8 build), Transformers ≥ 4.57.1, TRL 0.23.1, PEFT ≥ 0.17.1, vLLM 0.10.2, FlashAttention 2, Triton ≥ 3.4, `liger-kernel`, `datasets`, `pandas`, `accelerate`, `huggingface_hub`, `packaging`, `ninja`.

> Tip: The scripts assume CUDA is available. MXFP4 inference paths (`Mxfp4Config`) require Compute Capability ≥ 7.5.

## Environment Setup
1. **Install uv (optional but recommended):**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"
   ```
2. **Create a virtual environment:**
   ```bash
   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install -U pip wheel setuptools
   ```
   (Standard `python -m venv` and `pip` workflows also work.)
3. **Install core dependencies:**
   ```bash
   uv pip install "vllm==0.10.2" \
     --extra-index-url https://wheels.vllm.ai/0.10.2/ \
     --config-settings vllm:torch-backend=auto

   uv pip install --no-build-isolation \
     "transformers>=4.57.1" \
     "trl==0.23.1" \
     "peft>=0.17.1" \
     "accelerate>=1.10.0" \
     datasets pandas \
     "huggingface_hub>=0.25" \
     packaging ninja \
     "kernels>=0.10" \
     "triton>=3.4" \
     "liger-kernel" \
     "flash-attn==2.8.3"
   ```
4. **Confirm GPU visibility:**
   ```bash
   nvidia-smi
   ```

If PyTorch is not already installed via vLLM, add
```bash
uv pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.8.0" "torchvision==0.19.0" "torchaudio==2.8.0"
```

## Data Preparation
1. **Understand the CSV schema.** Each prompt row should include:
   - `prompt` — Raw or legacy text describing the decision task.
   - `q_action_0` … `q_action_3` — Action value estimates.
   - Optional reward columns (`reward_action_*`) if already computed.
2. **Convert prompts to Harmony chat format:**
   ```bash
   python convert_to_harmony.py data/dynamic_co2_factor_hourly_3day_0505_to_0507_q_dataset.csv
   ```
   The script adds the `_harmony` suffix and uses the GPT-OSS chat template so the tokenizer sees properly formatted system/user turns.
3. **Generate reward distributions from Q-values:**
   ```bash
   python calculate_reward.py data --tau 0.05 --overwrite
   ```
   This applies a Boltzmann softmax (temperature `tau`) to populate `reward_action_*` columns used by GRPO.
4. **Optional: validate column coverage.** Ensure every CSV referenced by the training run has all required columns before launching training.

## Training with GRPO
Run the training script once the environment and data are ready:
```bash
python train_grpo.py
```
Key configuration constants (edit inside `train_grpo.py` if you need to change them):
- `MODEL_ID` — Base checkpoint (defaults to `openai/gpt-oss-20b`).
- `OUT` — Output directory under `runs/` where adapters, tokenizer, and trainer state are saved.
- Sampling hyperparameters: `NUM_GENERATIONS`, `PROMPTS_PER_STEP`, `TRAIN_BATCH_SIZE`, `MAX_PROMPT_LEN`, `MAX_COMPLETION_LEN`.
- Optimization budget: `TOTAL_STEPS`, `SAVE_STEPS`, learning rate, gradient accumulation steps.

The script loads MXFP4 weights, dequantizes to bfloat16 for training, instantiates LoRA adapters via PEFT, and wires up `GRPOTrainer`. Prompts stream from `StepStream`, which randomly samples prompts each micro-step and replicates them `NUM_GENERATIONS` times so TRL can draw multiple completions per prompt.

### Logging and Rewards
- Reward computation is handled by `data_reward.reward_fn`, which parses completions for bracketed actions (`[0]` … `[3]`). Invalid or truncated generations return `NaN` so the trainer can mask them.
- Set `GRPO_LOG_COMPLETIONS=1` (default) to log prompts, completions, actions, and rewards to `runs/micro_step_completions.csv`. Use `GRPO_COMPLETION_LOG_PATH` to override the file location.
- `GRPO_STEPS_PER_GENERATION` is derived from gradient accumulation to maintain consistent micro-step indexing. You can override it via environment variable if you modify batching behavior.

## Evaluating the Adapter
Use the test scripts to sanity-check your adapter directory:
1. Edit `test_lora_adapter.py` and set the constants near the top (`ADAPTER_PATH`, `PROMPT_PATH`, decoding parameters, and optionally `BASE_MODEL_ID`) to match your run directory under `runs/`.
2. Run:
   ```bash
   python test_lora_adapter.py
   ```
3. Repeat the same process with `test_lora_adapter_quantized.py` if you want to exercise MXFP4 inference without dequantizing:
   ```bash
   python test_lora_adapter_quantized.py
   ```

## Exporting a Merged Model
After training, you can bake the LoRA weights back into the base model for standalone deployment:
1. Edit `export_quantized_model.py` to point `ADAPTER_PATH` at your fresh run directory under `runs/`, and adjust `OUTPUT_DIR` as needed.
2. Export with optional Hugging Face Hub upload:
   ```bash
   export HF_TOKEN=<your-hf-token>
   export HF_REPO_ID=<username>/<private-repo>
   python export_quantized_model.py
   ```
   The script merges the adapter via PEFT, annotates the `quantization_config` with MXFP4 metadata, and copies tokenizer assets. Set `PUSH_TO_HUB = False` if you only need local artifacts.

## Inference with a Merged Model
Use `run_exported_model.py` to run generations against the merged checkpoint:
1. Update the constants at the top of the script (`MODEL_ID`, `PROMPT_PATH`, decoding parameters) to reference your exported folder or remote repository.
2. Execute:
   ```bash
   python run_exported_model.py
   ```
Ensure the configured `MODEL_ID` resolves to the merged checkpoint and that a CUDA-capable GPU is available for MXFP4 inference.

## Serving via Ollama
The included `Modelfile` demonstrates how to load the trained adapter into Ollama:
1. Pull the GPT-OSS base image:
   ```bash
   ollama pull gpt-oss:20b
   ```
2. Update the `ADAPTER` path inside `Modelfile` to point at the adapter directory created under `runs/`.
3. Create and run the custom Ollama model:
   ```bash
   ollama create gptoss-20b-custom -f Modelfile
   ollama run gptoss-20b-custom
   ```

## Troubleshooting Tips
- **Out-of-memory errors:** Lower `MAX_COMPLETION_LEN`, reduce `NUM_GENERATIONS`, or enable `GRPOConfig.use_vllm=True` with tighter `vllm_gpu_memory_utilization` to offload decoding.
- **MXFP4 support errors:** Confirm `transformers` ≥ 4.57.1 and `triton` ≥ 3.4 are installed; verify the GPU has Compute Capability ≥ 7.5.
- **FlashAttention build issues:** Install `packaging` and `ninja` first, then retry with `MAX_JOBS=4` to limit parallel compilation.
- **Hub upload failures:** Make sure `HF_TOKEN` has `write` permission and that `HF_REPO_ID` references an existing or creatable private model repo.

## Next Steps
- Add automated evaluation metrics or preference model feedback loops.
- Integrate experiment tracking (e.g., Weights & Biases) for training dashboards.
- Extend dataset tooling to cover additional decision phases or action spaces.
