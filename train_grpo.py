# train_grpo_single_gpu.py
from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig
import torch
from torch.utils.data import IterableDataset 
from data_reward import load_prompt_dataset, reward_fn
import os, random

MODEL_ID = "openai/gpt-oss-20b"
OUT = "runs/grpo_gptoss20b_lora4_tes"

TOTAL_STEPS = 100
PROMPTS_PER_STEP = 12
NUM_GENERATIONS = 8
MAX_PROMPT_LEN = 1000
MAX_COMPLETION_LEN = 4000 
SEED = 42

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

# --- LoRA r=4（一般的な対象） ---
# LoRA: MoE MLP を確実に含めるために全層を自動列挙（例）
expert_params = []
for name, module in model.named_modules():
    if "mlp.experts" in name and (name.endswith("gate_up_proj") or name.endswith("down_proj")):
        expert_params.append(name)

lora = LoraConfig(
    r=4, lora_alpha=8,
    target_modules="all-linear",
    target_parameters=expert_params,      # ← 固定列挙から自動列挙に
    task_type="CAUSAL_LM",                # Enum でも可
)
model = get_peft_model(model, lora)

# ----------------- Dataset (データローダ) ----------------
base = load_prompt_dataset()
random.seed(SEED)

class StepStream(IterableDataset):
    """毎ステップちょうどk件のプロンプトをランダム抽出して流すストリーム"""
    def __init__(self, base_ds, k):
        self.base = base_ds
        self.k = k
        self.n = len(base_ds)
        self.keys = list(base_ds.features.keys())

    def __iter__(self):
        while True:
            idxs = random.sample(range(self.n), self.k)  # 重複なしで16件
            for i in idxs:
                row = self.base[i]
                yield {k: row[k] for k in self.keys}

stream = StepStream(base, k=PROMPTS_PER_STEP)

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

    # 生成エンジン（vLLM）
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.35,  # 学習と取り合わないよう枠を抑える
    vllm_enable_sleep_mode=True,       # 生成←→学習の切替でVRAMを返す（初回のみ起床遅延あり）

    # 「1ステップ=12プロンプト×各8生成」を担保
    num_generations=NUM_GENERATIONS,
    generation_batch_size=PROMPTS_PER_STEP * NUM_GENERATIONS,  # 12 prompts * 8 generations
    per_device_train_batch_size=PROMPTS_PER_STEP,    # 12
    gradient_accumulation_steps=1,     # 1
    
    # 長さまわり
    max_prompt_length=MAX_PROMPT_LEN,
    max_completion_length=MAX_COMPLETION_LEN,
)

# 実行
trainer = GRPOTrainer(
    model=model,
    processing_class=tok,   # 現行API名（左パディング必須）
    args=args,
    reward_funcs=reward_fn,
    train_dataset=stream,   # ← 毎ステップ12件だけ供給
)
trainer.train()

# 保存（LoRAアダプタ形式）
trainer.save_model(OUT)
tok.save_pretrained(OUT)
print("✅ finished")