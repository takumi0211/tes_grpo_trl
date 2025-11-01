# train_grpo_vllm.py
from __future__ import annotations

from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config, logging as hf_logging
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig
import torch
from torch.utils.data import IterableDataset
from data_reward import load_prompt_dataset, reward_fn
import os
import random
import logging

# ---------------------------
# ハイパーパラメータ
# ---------------------------
MODEL_ID = "openai/gpt-oss-20b"
OUT = "runs/grpo_gptoss20b_lora4_vllm_server"

TOTAL_STEPS = 10
PROMPTS_PER_STEP = 1
NUM_GENERATIONS = 4
GRADIENT_ACCUMULATION_STEPS = 4
TRAIN_BATCH_SIZE = NUM_GENERATIONS
MAX_PROMPT_LEN = 1000
MAX_COMPLETION_LEN = 2600
SEED = 42

# vLLM server mode 接続設定（環境変数で上書き可能）
VLLM_SERVER_HOST = os.getenv("VLLM_SERVER_HOST", "localhost")
VLLM_SERVER_PORT = int(os.getenv("VLLM_SERVER_PORT", "8000"))
VLLM_SERVER_BASE_URL = os.getenv(
    "VLLM_SERVER_BASE_URL",
    f"http://{VLLM_SERVER_HOST}:{VLLM_SERVER_PORT}",
)
VLLM_TP_SIZE = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "2"))

# 学習側は eager Attention を固定（生成側の vLLM で FlashAttention-3 を利用）
ATTENTION_IMPL = "kernels-community/vllm-flash-attn3"

# ---------------------------
# ロギング
# ---------------------------
os.makedirs(OUT, exist_ok=True)
logger = logging.getLogger("train_grpo_vllm")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    _stream_handler = logging.StreamHandler()
    _stream_handler.setFormatter(_formatter)
    _file_handler = logging.FileHandler(os.path.join(OUT, "training.log"), mode="a")
    _file_handler.setFormatter(_formatter)
    logger.addHandler(_stream_handler)
    logger.addHandler(_file_handler)
    logger.propagate = False

hf_logging.set_verbosity_info()
hf_logging.enable_default_handler()
hf_logging.enable_explicit_format()

# ---------------------------
# トークナイザー & モデル
# ---------------------------
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

quant_cfg = Mxfp4Config(dequantize=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    quantization_config=quant_cfg,
    attn_implementation=ATTENTION_IMPL,
    use_cache=False,
    device_map="auto",
)
logger.info("Loaded policy model dtype=%s", next(model.parameters()).dtype)

expert_params = []
for name, module in model.named_modules():
    if "mlp.experts" in name and (name.endswith("gate_up_proj") or name.endswith("down_proj")):
        expert_params.append(name)

lora = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules="all-linear",
    target_parameters=expert_params,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)

logger.info(
    "LoRA target parameters enumerated (%d): %s",
    len(expert_params),
    expert_params if expert_params else "all linear layers",
)
trainable_lora_params = [
    f"{name} shape={tuple(param.shape)}"
    for name, param in model.named_parameters()
    if param.requires_grad and "lora" in name.lower()
]
if trainable_lora_params:
    preview = trainable_lora_params[:20]
    remainder = len(trainable_lora_params) - len(preview)
    suffix = f" ... (+{remainder} more)" if remainder > 0 else ""
    logger.info(
        "Trainable LoRA parameters (%d): %s%s",
        len(trainable_lora_params),
        preview,
        suffix,
    )
else:
    logger.warning("No LoRA parameters detected as trainable.")

# ---------------------------
# データストリーム
# ---------------------------
base = load_prompt_dataset()
random.seed(SEED)


class StepStream(IterableDataset):
    """マイクロステップ毎に PROMPTS_PER_STEP × NUM_GENERATIONS を供給するストリーム。"""

    KEEP_KEYS = (
        "reward_action_0",
        "reward_action_1",
        "reward_action_2",
        "reward_action_3",
        "prompt",
    )

    def __init__(self, base_ds, k, num_generations):
        self.base = base_ds
        self.k = k
        self.n = len(base_ds)
        self.num_generations = num_generations
        dense_keys = [key for key in self.KEEP_KEYS if key in base_ds.features and key != "prompt"]
        if "prompt" in base_ds.features:
            dense_keys.append("prompt")
        self.keys = dense_keys

    def __iter__(self):
        while True:
            idxs = random.sample(range(self.n), self.k)
            for i in idxs:
                row = self.base[i]
                sample = {}
                for key in self.keys:
                    value = row[key]
                    if key == "prompt":
                        sample[key] = value
                    else:
                        sample[key] = torch.atleast_1d(torch.tensor(value, dtype=torch.float32))
                for _ in range(self.num_generations):
                    yield {
                        key: (value.clone() if isinstance(value, torch.Tensor) else value)
                        for key, value in sample.items()
                    }


stream = StepStream(base, k=PROMPTS_PER_STEP, num_generations=NUM_GENERATIONS)
logger.info(
    "StepStream configured | prompts_per_micro_step=%d | num_generations=%d | dataset_rows=%d | keep_keys=%s",
    PROMPTS_PER_STEP,
    NUM_GENERATIONS,
    len(base),
    stream.keys,
)

# ---------------------------
# GRPO Config（vLLM server mode）
# ---------------------------
args = GRPOConfig(
    output_dir=OUT,
    max_steps=TOTAL_STEPS,
    learning_rate=5e-5,
    bf16=True,
    fp16=False,
    gradient_checkpointing=True,
    seed=SEED,
    accelerator_config={"split_batches": True},
    logging_steps=1,

    # vLLM server mode 設定
    use_vllm=True,
    vllm_mode="server",
    vllm_tensor_parallel_size=VLLM_TP_SIZE,
    vllm_server_host=VLLM_SERVER_HOST,
    vllm_server_port=VLLM_SERVER_PORT,
    vllm_server_base_url=VLLM_SERVER_BASE_URL,
    vllm_server_timeout=60,
    vllm_gpu_memory_utilization=0.8,
    vllm_kv_cache_dtype="fp8",
    vllm_enable_sleep_mode=False,

    # 生成設定
    num_generations=NUM_GENERATIONS,
    generation_batch_size=TRAIN_BATCH_SIZE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    max_prompt_length=MAX_PROMPT_LEN,
    max_completion_length=MAX_COMPLETION_LEN,
    generation_kwargs={
        "use_cache": True,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.95,
        "eos_token_id": tok.eos_token_id,
    },
)

micro_batch_completions = TRAIN_BATCH_SIZE
total_completions_per_update = TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
split_batches_flag = getattr(args.accelerator_config, "split_batches", None)
logger.info(
    "Generation config | num_generations=%d | generation_batch_size=%s | per_device_train_batch_size=%d | "
    "grad_accum=%d | split_batches=%s | completions_per_micro_step=%d | completions_per_update=%d | "
    "vllm_base_url=%s",
    NUM_GENERATIONS,
    args.generation_batch_size,
    args.per_device_train_batch_size,
    GRADIENT_ACCUMULATION_STEPS,
    split_batches_flag,
    micro_batch_completions,
    total_completions_per_update,
    VLLM_SERVER_BASE_URL,
)


# ---------------------------
# 学習実行
# ---------------------------
trainer = GRPOTrainer(
    model=model,
    processing_class=tok,
    args=args,
    reward_funcs=reward_fn,
    train_dataset=stream,
)
logger.info("Starting training | total_steps=%d | output_dir=%s", TOTAL_STEPS, OUT)
trainer.train()

trainer.save_model(OUT)
tok.save_pretrained(OUT)
logger.info("Training artifacts saved to %s", OUT)
print("✅ finished")
