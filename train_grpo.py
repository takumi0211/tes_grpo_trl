# train_grpo_single_gpu.py
from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config, logging as hf_logging
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig
import torch
from torch.utils.data import IterableDataset 
from data_reward import load_prompt_dataset, reward_fn
import os, random, logging

MODEL_ID = "openai/gpt-oss-20b"
OUT = "runs/grpo_gptoss20b_lora4_tes"

TOTAL_STEPS = 100
PROMPTS_PER_STEP = 4
NUM_GENERATIONS = 4
MAX_PROMPT_LEN = 1000
MAX_COMPLETION_LEN = 2500
SEED = 42

# --- Logging setup ---
os.makedirs(OUT, exist_ok=True)
logger = logging.getLogger("train_grpo")
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

# --- Tokenizer ---
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# --- MXFP4 -> BF16 にデクオンして学習（公式ルート） ---
# 参考: OpenAI/Transformersのcookbook・ブログで Mxfp4Config(dequantize=True) を明記。:contentReference[oaicite:5]{index=5}
quant_cfg = Mxfp4Config(dequantize=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16, 
    quantization_config=quant_cfg,
    attn_implementation="eager",  # 学習側はeagerが安定
    use_cache=False,               # 勾配チェックポイントと相性良
    device_map="auto",
)

# --- LoRA r=4 ---
# LoRA: MoE MLP を確実に含めるために全層を自動列挙
expert_params = []
for name, module in model.named_modules():
    if "mlp.experts" in name and (name.endswith("gate_up_proj") or name.endswith("down_proj")):
        expert_params.append(name)

lora = LoraConfig(
    r=4, lora_alpha=8,
    target_modules="all-linear",
    target_parameters=expert_params,      # ← 固定列挙から自動列挙に
    task_type="CAUSAL_LM",            
)
model = get_peft_model(model, lora)

# ----------------- Dataset (データローダ) ----------------
base = load_prompt_dataset()
random.seed(SEED)

class StepStream(IterableDataset):
    """毎ステップちょうどk件のプロンプトをランダム抽出して流すストリーム

    注意: HF/Accelerate はワーカー間でバッチを結合するとき、
    末端の型が PyTorch Tensor でないと `concatenate` で失敗します。
    そのため学習に不要な列（例: `timestep`, `timestamp`, `optimal_action`,
    `q_action_*` などのメタ情報）は落とし、報酬に使う列のみ通します。
    """

    KEEP_KEYS = (
        "reward_action_0",
        "reward_action_1",
        "reward_action_2",
        "reward_action_3",
        "prompt",
    )

    def __init__(self, base_ds, k):
        self.base = base_ds
        self.k = k
        self.n = len(base_ds)
        # データセットに実際に存在するキーとの積集合を使う
        dense_keys = [k for k in self.KEEP_KEYS if k in base_ds.features and k != "prompt"]
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
                        sample[key] = torch.atleast_1d(
                            torch.tensor(value, dtype=torch.float32)
                        )
                yield sample

stream = StepStream(base, k=PROMPTS_PER_STEP)
logger.info(
    "StepStream configured | prompts_per_step=%d | dataset_rows=%d | keep_keys=%s",
    PROMPTS_PER_STEP,
    len(base),
    stream.keys,
)

# ----------------- TRL/GRPO + vLLM (colocate) -----------------
# colocate: 学習プロセス内でvLLMを起動（省メモリのため sleep を有効化）。
# ※ vLLM 0.10.2 を使用（TRLのサポートバージョン）
args = GRPOConfig(
    output_dir=OUT,
    max_steps=TOTAL_STEPS,
    learning_rate=5e-5,
    bf16=True,
    gradient_checkpointing=True,
    seed=SEED,
    accelerator_config={"split_batches": True},

    # 生成エンジン（vLLM）
    use_vllm=False,
    # vllm_mode="colocate",
    # vllm_gpu_memory_utilization=0.35,  # 学習と取り合わないよう枠を抑える
    # # vllm_kv_cache_dtype="fp8",
    # vllm_enable_sleep_mode=True,       # 生成←→学習の切替でVRAMを返す（初回のみ起床遅延あり）

    # 「1ステップ=12プロンプト×各8生成」を担保
    num_generations=NUM_GENERATIONS,
    generation_batch_size=PROMPTS_PER_STEP,  # 12 prompts
    per_device_train_batch_size=PROMPTS_PER_STEP,    # 12
    gradient_accumulation_steps=1,     # 1
    
    # 長さまわり
    max_prompt_length=MAX_PROMPT_LEN,
    max_completion_length=MAX_COMPLETION_LEN,
)
total_completions_per_step = PROMPTS_PER_STEP * NUM_GENERATIONS
split_batches_flag = getattr(args.accelerator_config, "split_batches", None)
logger.info(
    "Generation config | num_generations=%d | generation_batch_size=%s | per_device_train_batch_size=%d | "
    "split_batches=%s | total_completions_per_step=%d",
    NUM_GENERATIONS,
    args.generation_batch_size,
    args.per_device_train_batch_size,
    split_batches_flag,
    total_completions_per_step,
)

# 実行
trainer = GRPOTrainer(
    model=model,
    processing_class=tok,   # 現行API名（左パディング必須）
    args=args,
    reward_funcs=reward_fn,
    train_dataset=stream,   # ← 毎ステップ12件だけ供給
)
logger.info("Starting training | total_steps=%d | output_dir=%s", TOTAL_STEPS, OUT)
trainer.train()

# 保存（LoRAアダプタ形式）
trainer.save_model(OUT)
tok.save_pretrained(OUT)
logger.info("Training artifacts saved to %s", OUT)
print("✅ finished")
