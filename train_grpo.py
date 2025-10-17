#!/usr/bin/env python3
"""
Training entrypoint for GRPO fine-tuning of unsloth/gpt-oss-20b-4bit.

This script expects the CSV datasets in ./data as described in memo.md.
It loads prompts and pre-computed Q-values, samples multiple completions
per prompt, and uses the Q-values as the reward signal during GRPO.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence, Optional

import pandas as pd
import torch
from datasets import Dataset
from transformers import set_seed
from trl import GRPOConfig, GRPOTrainer


# Conversation parsing -----------------------------------------------------------------

ROLE_PREFIX = re.compile(r"^(system|user|assistant):\s*", re.IGNORECASE)


def _flush_message(messages: list[dict[str, str]], role: str | None, lines: list[str]) -> None:
    if role is None or not lines:
        return
    content = "\n".join(lines).strip()
    if not content:
        return
    messages.append({"role": role.lower(), "content": content})


def to_chat_messages(raw_prompt: str) -> list[dict[str, str]]:
    """
    Convert the serialized prompt from the CSV into a structured chat format.

    The CSV encodes successive messages prefixed by `system:`, `user:`, etc.
    """
    messages: list[dict[str, str]] = []
    role: str | None = None
    buffer: list[str] = []

    for line in raw_prompt.splitlines():
        match = ROLE_PREFIX.match(line)
        if match:
            _flush_message(messages, role, buffer)
            role = match.group(1)
            buffer = [line[match.end() :]]
        else:
            buffer.append(line)

    _flush_message(messages, role, buffer)
    return messages


# Data loading -------------------------------------------------------------------------

Q_VALUE_COLUMNS = [f"q_action_{i}" for i in range(4)]


@dataclass(frozen=True)
class PromptRecord:
    prompt: list[dict[str, str]]
    q_values: list[float]
    optimal_action: int
    metadata: dict[str, object]


def _safe_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value))


def load_prompt_records(data_root: Path) -> list[PromptRecord]:
    csv_paths = sorted(data_root.glob("*_q_dataset.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No *_q_dataset.csv files found under {data_root}")

    records: list[PromptRecord] = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            q_values = [_safe_float(row[col]) for col in Q_VALUE_COLUMNS]
            prompt = to_chat_messages(str(row["prompt"]))
            metadata = {
                "source_dataset": row.get("dataset"),
                "timestep": int(row["timestep"]),
                "timestamp": row.get("timestamp"),
                "tes_energy": _safe_float(row["tes_energy"]),
                "tes_energy_normalized": _safe_float(row["tes_energy_normalized"]),
                "is_business_hour": int(row["is_business_hour"]),
            }
            records.append(
                PromptRecord(
                    prompt=prompt,
                    q_values=q_values,
                    optimal_action=int(row["optimal_action"]),
                    metadata=metadata,
                )
            )
    return records


def build_dataset(records: Sequence[PromptRecord]) -> Dataset:
    dataset_dicts = []
    for rec in records:
        dataset_dicts.append(
            {
                "prompt": rec.prompt,
                "q_values": rec.q_values,
                "optimal_action": rec.optimal_action,
                "metadata_json": json.dumps(rec.metadata, ensure_ascii=True),
            }
        )
    return Dataset.from_list(dataset_dicts)


# Reward shaping -----------------------------------------------------------------------

ACTION_PATTERN = re.compile(r"^\s*\[(\d)\]\s*$")


def extract_action(completion: object) -> int | None:
    """
    Parse the generated completion and extract the action index.
    Returns None if the expected `[digit]` format is not found.
    """
    if isinstance(completion, list):
        # Conversational completions are lists of {"role": "...", "content": "..."}
        if not completion:
            return None
        content = completion[-1].get("content", "")
    else:
        content = str(completion)

    stripped = content.strip()
    last_line = stripped.splitlines()[-1] if stripped else ""
    match = ACTION_PATTERN.match(last_line)
    if not match:
        return None

    action = int(match.group(1))
    if action < 0 or action >= len(Q_VALUE_COLUMNS):
        return None
    return action


def softmax(values: Sequence[float]) -> list[float]:
    max_val = max(values)
    exps = [math.exp(v - max_val) for v in values]
    denom = sum(exps)
    if denom == 0.0:
        return [1.0 / len(values)] * len(values)
    return [x / denom for x in exps]


def build_q_reward_fn(
    invalid_penalty: float,
    reward_mode: str,
    temperature: float,
) -> callable:
    def reward_function(
        *,
        prompts: list,
        completions: list,
        q_values: list[list[float]],
        optimal_action: list[int],
        trainer_state,
        **_: dict,
    ) -> list[float]:
        rewards: list[float] = []
        for completion, sample_qs in zip(completions, q_values):
            action = extract_action(completion)
            if action is None:
                rewards.append(invalid_penalty)
            else:
                qs = [float(x) for x in sample_qs]
                if reward_mode == "raw":
                    rewards.append(qs[action])
                    continue

                # Advantage relative to best action
                max_q = max(qs)
                advantages = [q - max_q for q in qs]

                if reward_mode == "advantage":
                    rewards.append(advantages[action])
                elif reward_mode == "softmax":
                    tau = max(temperature, 1e-5)
                    scaled = [adv / tau for adv in advantages]
                    weights = softmax(scaled)
                    rewards.append(weights[action])
                else:
                    raise ValueError(f"Unsupported reward mode: {reward_mode}")
        return rewards

    return reward_function


def _find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    checkpoints = sorted(
        (p for p in output_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda path: int(path.name.split("-")[-1]) if path.name.split("-")[-1].isdigit() else -1,
    )
    if checkpoints:
        return checkpoints[-1]
    return None


# Trainer assembly ---------------------------------------------------------------------

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def infer_prompt_stats(dataset: Dataset, tokenizer, add_generation_prompt: bool = True) -> tuple[int, float]:
    prompt_lengths: list[int] = []
    for row in dataset:
        encoded = tokenizer.apply_chat_template(
            row["prompt"],
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=True,
        )
        prompt_lengths.append(len(encoded["input_ids"]))

    return max(prompt_lengths), float(sum(prompt_lengths) / len(prompt_lengths))


def prepare_model_and_tokenizer(args) -> tuple[torch.nn.Module, object]:
    try:
        from unsloth import FastLanguageModel
    except NotImplementedError as exc:  # pragma: no cover - raised on CPU-only machines
        raise RuntimeError(
            "Unsloth requires a CUDA, HIP, or XPU capable device. "
            "Please rerun this script on a GPU machine."
        ) from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.full_precision,
        offload_embedding=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=TARGET_MODULES,
        lora_alpha=args.lora_alpha or args.lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    model.config.use_cache = False
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def configure_trainer(
    args,
    model,
    tokenizer,
    dataset: Dataset,
    reward_fn,
) -> GRPOTrainer:
    max_prompt_length, avg_prompt_length = infer_prompt_stats(dataset, tokenizer)
    max_completion_length = args.max_seq_length - max_prompt_length
    if max_completion_length <= 0:
        raise ValueError(
            f"max_seq_length ({args.max_seq_length}) is too small for the prompts "
            f"(max prompt tokens={max_prompt_length}). Increase --max-seq-length."
        )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        bf16=True,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        report_to=args.report_to,
        remove_unused_columns=False,
    )

    print(
        f"[info] max_prompt_tokens={max_prompt_length} "
        f"(avg={avg_prompt_length:.1f}), max_completion_tokens={max_completion_length}"
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=dataset,
    )
    return trainer


# CLI ----------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO finetuning with Q-value rewards.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing *_q_dataset.csv files.")
    parser.add_argument("--model-name", type=str, default="unsloth/gpt-oss-20b", help="Base checkpoint to finetune.")
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=10000,
        help="Total max sequence length (prompt + completion). Defaults to 10000 tokens for long reasoning windows.",
    )
    parser.add_argument("--full-precision", action="store_true", help="Disable 4-bit loading and use full precision LoRA.")
    parser.add_argument("--lora-rank", type=int, default=4, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha; defaults to 2 * lora_rank.")
    parser.add_argument("--num-generations", type=int, default=8, help="Number of completions sampled per prompt.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1, help="Prompts per GPU.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio for LR scheduler.")
    parser.add_argument("--lr-scheduler", type=str, default="linear", help="LR scheduler type.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum training steps.")
    parser.add_argument("--logging-steps", type=int, default=5, help="Logging interval (steps).")
    parser.add_argument("--save-steps", type=int, default=50, help="Checkpoint interval (steps).")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling.")
    parser.add_argument("--invalid-penalty", type=float, default=-1.0, help="Reward assigned to malformed outputs.")
    parser.add_argument(
        "--q-reward-mode",
        choices=["raw", "advantage", "softmax"],
        default="softmax",
        help="How to map Q-values to scalar rewards.",
    )
    parser.add_argument(
        "--q-temperature",
        type=float,
        default=0.1,
        help="Temperature used when --q-reward-mode=softmax to control sharpness.",
    )
    parser.add_argument("--seed", type=int, default=3407, help="Random seed.")
    parser.add_argument("--report-to", type=str, default="none", help="Logging integrations (e.g., wandb).")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for trainer artifacts.")
    parser.add_argument(
        "--skip-save-lora",
        action="store_true",
        help="Disable saving the LoRA adapter weights after training.",
    )
    parser.add_argument(
        "--skip-save-mxfp4",
        action="store_true",
        help="Disable saving merged weights in mxfp4 format.",
    )
    parser.add_argument(
        "--skip-save-16bit",
        action="store_true",
        help="Disable saving merged weights in 16bit format.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint folder to resume from, or 'latest' to auto-detect the most recent checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    records = load_prompt_records(args.data_dir)
    dataset = build_dataset(records)
    reward_fn = build_q_reward_fn(args.invalid_penalty, args.q_reward_mode, args.q_temperature)

    model, tokenizer = prepare_model_and_tokenizer(args)
    trainer = configure_trainer(args, model, tokenizer, dataset, reward_fn)

    print(f"[info] Starting GRPO training for {len(dataset)} prompts...")
    resume_path: Optional[str] = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint.lower() in {"latest", "last"}:
            latest = _find_latest_checkpoint(args.output_dir)
            if latest is None:
                print(f"[warn] No checkpoint-* directories found under {args.output_dir}; starting fresh.")
            else:
                resume_path = str(latest)
                print(f"[info] Resuming from latest checkpoint: {resume_path}")
        else:
            resume_path = args.resume_from_checkpoint
            if not Path(resume_path).exists():
                print(f"[warn] Checkpoint path {resume_path} does not exist; starting fresh.")
                resume_path = None
            else:
                print(f"[info] Resuming from checkpoint: {resume_path}")

    trainer.train(resume_from_checkpoint=resume_path)
    print("[info] Training complete. Run trainer.save_state() or merge adapters as needed.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_save_lora:
        adapter_dir = args.output_dir / "lora_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        print(f"[info] Saved LoRA adapter and tokenizer to {adapter_dir}")

    if not args.skip_save_mxfp4:
        mxfp4_dir = args.output_dir / "merged_mxfp4"
        mxfp4_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained_merged(mxfp4_dir, tokenizer, save_method="mxfp4")
        print(f"[info] Saved merged mxfp4 model to {mxfp4_dir}")

    if not args.skip_save_16bit:
        fp16_dir = args.output_dir / "merged_16bit"
        fp16_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained_merged(fp16_dir, tokenizer, save_method="merged_16bit")
        print(f"[info] Saved merged 16bit model to {fp16_dir}")


if __name__ == "__main__":
    main()
