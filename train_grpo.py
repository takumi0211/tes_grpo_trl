#!/usr/bin/env python3
"""
Training entrypoint for GRPO fine-tuning of unsloth/gpt-oss-20b-4bit.

This script expects the CSV datasets in ./data as described in memo.md.
It loads prompts and pre-computed Q-values, samples multiple completions
per prompt, and uses the Q-values as the reward signal during GRPO.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence, Optional

import pandas as pd
import torch
from datasets import Dataset
import unsloth  # ensure unsloth patches are applied before transformers/trl imports
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


class MetricsTracker:
    def __init__(self, csv_path: Optional[Path], ema_alpha: float = 0.1):
        self.csv_path = Path(csv_path) if csv_path else None
        self.ema_alpha = ema_alpha
        self.ema_regret: Optional[float] = None
        self.ema_reward: Optional[float] = None
        if self.csv_path:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.csv_path.exists():
                with self.csv_path.open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        ["global_step", "parse_rate", "acc_opt", "mean_regret", "mean_reward"]
                    )

    def update(
        self,
        step: int,
        parse_rate: float,
        acc_opt: float,
        batch_regret: Optional[float],
        batch_reward: Optional[float],
    ) -> None:

        if batch_regret is not None:
            if self.ema_regret is None:
                self.ema_regret = batch_regret
            else:
                self.ema_regret = self.ema_alpha * batch_regret + (1 - self.ema_alpha) * self.ema_regret
        if batch_reward is not None:
            if self.ema_reward is None:
                self.ema_reward = batch_reward
            else:
                self.ema_reward = self.ema_alpha * batch_reward + (1 - self.ema_alpha) * self.ema_reward

        if self.csv_path:
            with self.csv_path.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        step,
                        f"{parse_rate:.6f}",
                        f"{acc_opt:.6f}",
                        f"{(self.ema_regret if self.ema_regret is not None else float('nan')):.6f}",
                        f"{(self.ema_reward if self.ema_reward is not None else float('nan')):.6f}",
                    ]
                )


class GenerationLogger:
    """
    Streams every generated completion alongside its prompt and reward details.
    """

    def __init__(self, csv_path: Optional[Path]):
        self.csv_path = Path(csv_path) if csv_path else None
        if self.csv_path:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.csv_path.exists():
                with self.csv_path.open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "global_step",
                            "batch_index",
                            "prompt_json",
                            "completion_json",
                            "parsed_action",
                            "optimal_action",
                            "reward_value",
                            "regret_value",
                            "q_values_json",
                        ]
                    )

    def log(
        self,
        *,
        step: Optional[int],
        batch_index: int,
        prompt: object,
        completion: object,
        parsed_action: Optional[int],
        reward_value: float,
        q_values: Sequence[float],
        optimal_action: int,
        regret_value: Optional[float],
    ) -> None:
        if not self.csv_path:
            return

        prompt_json = json.dumps(prompt, ensure_ascii=True)
        if isinstance(completion, (list, dict)):
            completion_json = json.dumps(completion, ensure_ascii=True)
        else:
            completion_json = json.dumps(str(completion), ensure_ascii=True)
        q_values_json = json.dumps([float(x) for x in q_values], ensure_ascii=True)

        with self.csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "" if step is None else step,
                    batch_index,
                    prompt_json,
                    completion_json,
                    "" if parsed_action is None else parsed_action,
                    optimal_action,
                    reward_value,
                    "" if regret_value is None else regret_value,
                    q_values_json,
                ]
            )


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
    tracker: Optional[MetricsTracker],
    generation_logger: Optional[GenerationLogger],
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
        parse_success = 0
        acc_opt_count = 0
        regret_sum = 0.0
        reward_sum = 0.0

        current_step = getattr(trainer_state, "global_step", None)
        is_world_zero = getattr(trainer_state, "is_world_process_zero", True)
        if callable(is_world_zero):
            is_world_zero = is_world_zero()
        should_log = generation_logger is not None and is_world_zero

        for batch_index, (prompt_entry, completion, sample_qs, opt_action) in enumerate(
            zip(prompts, completions, q_values, optimal_action)
        ):
            qs = [float(x) for x in sample_qs]
            max_q = max(qs)
            action = extract_action(completion)
            regret_value: Optional[float] = None

            if action is None:
                reward_value = invalid_penalty
                rewards.append(reward_value)
                reward_sum += reward_value
            else:
                advantages = [q - max_q for q in qs]

                if reward_mode == "advantage":
                    reward_value = advantages[action]
                elif reward_mode == "softmax":
                    tau = max(temperature, 1e-5)
                    scaled = [adv / tau for adv in advantages]
                    weights = softmax(scaled)
                    reward_value = weights[action]
                else:
                    raise ValueError(f"Unsupported reward mode: {reward_mode}")

                rewards.append(reward_value)
                parse_success += 1
                reward_sum += reward_value
                regret_value = max_q - qs[action]
                regret_sum += regret_value
                best_action = max(range(len(qs)), key=lambda i: qs[i])
                if action == best_action:
                    acc_opt_count += 1

            if should_log:
                generation_logger.log(
                    step=current_step,
                    batch_index=batch_index,
                    prompt=prompt_entry,
                    completion=completion,
                    parsed_action=action,
                    reward_value=reward_value,
                    q_values=qs,
                    optimal_action=int(opt_action),
                    regret_value=regret_value,
                )

        total = len(completions)
        if (
            tracker
            and getattr(trainer_state, "is_world_process_zero", True)
            and total > 0
            and getattr(trainer_state, "global_step", None) is not None
        ):
            parse_rate = parse_success / total
            acc_opt = acc_opt_count / parse_success if parse_success > 0 else 0.0
            avg_reward = reward_sum / total if total > 0 else None
            avg_regret = regret_sum / parse_success if parse_success > 0 else None
            tracker.update(
                trainer_state.global_step,
                parse_rate,
                acc_opt,
                avg_regret,
                avg_reward,
            )
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
    tokenizer.model_max_length = args.max_seq_length
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

    num_generations = args.num_generations
    if num_generations is None or num_generations <= 0:
        raise ValueError("--num-generations must be a positive integer.")

    generation_batch_size = args.generation_batch_size if args.generation_batch_size is not None else 1
    if generation_batch_size <= 0:
        raise ValueError("--generation-batch-size must be a positive integer.")

    if args.steps_per_generation is not None and args.steps_per_generation <= 0:
        raise ValueError("--steps-per-generation must be a positive integer when provided.")
    if args.steps_per_generation is not None:
        steps_per_generation = args.steps_per_generation
    else:
        # Default vertical schedule: gather all completions over ceil(G / batch_size) micro-steps.
        steps_per_generation = max(1, math.ceil(num_generations / generation_batch_size))

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
        num_generations=num_generations,
        generation_batch_size=generation_batch_size,
        steps_per_generation=steps_per_generation,
        temperature=args.temperature,
        top_p=args.top_p,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        report_to=args.report_to,
        remove_unused_columns=False,
        optim=args.optim,
    )
    # Preserve the intended steps_per_generation while letting TRL clone the args safely.
    global_batch = max(1, training_args.per_device_train_batch_size * getattr(training_args, "world_size", 1))
    effective_generation_batch = training_args.generation_batch_size or global_batch
    computed_steps = (
        training_args.steps_per_generation
        if training_args.steps_per_generation is not None
        else max(1, (effective_generation_batch or global_batch) // global_batch)
    )
    training_args.__dict__["_target_steps_per_generation"] = computed_steps
    training_args.steps_per_generation = None

    print(
        f"[info] max_prompt_tokens={max_prompt_length} "
        f"(avg={avg_prompt_length:.1f}), max_completion_tokens={max_completion_length}"
    )
    if training_args.num_generations and effective_generation_batch:
        parallel_per_prompt = effective_generation_batch / training_args.num_generations
    else:
        parallel_per_prompt = None
    schedule_msg = (
        "[info] Generation schedule: "
        f"num_generations={training_args.num_generations}, "
        f"generation_batch_size={effective_generation_batch}, "
        f"steps_per_generation={computed_steps}"
    )
    if parallel_per_prompt is not None:
        schedule_msg += f", parallel_completions_per_prompt≈{parallel_per_prompt:.2f}"
    if (
        training_args.num_generations
        and effective_generation_batch
        and effective_generation_batch < training_args.num_generations
    ):
        schedule_msg += " (vertical low-memory mode)"
    print(schedule_msg)
    print(f"[info] Optimizer backend: {training_args.optim}")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=dataset,
    )

    # Enforce tight generation bounds so Unsloth does not fall back to extremely long defaults.
    model.generation_config.max_length = args.max_seq_length
    model.generation_config.max_new_tokens = max_completion_length
    if hasattr(trainer, "generation_config"):
        trainer.generation_config.max_length = args.max_seq_length
        trainer.generation_config.max_new_tokens = max_completion_length

    return trainer


# CLI ----------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO finetuning with Q-value rewards.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing *_q_dataset.csv files.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="unsloth/gpt-oss-20b-4bit",
        help="Base checkpoint to finetune (defaults to unsloth/gpt-oss-20b-4bit).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=5000,
        help="Total max sequence length (prompt + completion). Defaults to 5000 tokens for long reasoning windows.",
    )
    parser.add_argument("--full-precision", action="store_true", help="Disable 4-bit loading and use full precision LoRA.")
    parser.add_argument("--lora-rank", type=int, default=4, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha; defaults to 2 * lora_rank.")
    parser.add_argument("--num-generations", type=int, default=4, help="Number of completions sampled per prompt.")
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=1,
        help="Batch size used during response generation. Defaults to 1 (sequential 'vertical' generation).",
    )
    parser.add_argument(
        "--steps-per-generation",
        type=int,
        default=None,
        help=(
            "Number of micro generation loops to collect num-generations samples per step. "
            "Defaults to ceil(num-generations / generation-batch-size) when omitted."
        ),
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=1, help="Prompts per GPU.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument(
        "--optim",
        type=str,
        default="paged_adamw_8bit",
        help="Optimizer identifier passed to GRPOConfig, e.g. 'paged_adamw_8bit' or 'adamw_torch'.",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio for LR scheduler.")
    parser.add_argument("--lr-scheduler", type=str, default="linear", help="LR scheduler type.")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum training steps.")
    parser.add_argument("--logging-steps", type=int, default=5, help="Logging interval (steps).")
    parser.add_argument("--save-steps", type=int, default=20, help="Checkpoint interval (steps).")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling.")
    parser.add_argument("--invalid-penalty", type=float, default=-1.0, help="Reward assigned to malformed outputs.")
    parser.add_argument(
        "--q-reward-mode",
        choices=["advantage", "softmax"],
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
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint folder to resume from, or 'latest' to auto-detect the most recent checkpoint.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("metrics.csv"),
        help="Destination CSV file for streaming training metrics (parse rate, regret, reward). Use 'none' to disable.",
    )
    parser.add_argument(
        "--metrics-ema-alpha",
        type=float,
        default=0.1,
        help="Smoothing factor for the exponential moving average applied to regret and reward metrics.",
    )
    parser.add_argument(
        "--generations-csv",
        type=Path,
        default=Path("generations.csv"),
        help="Destination CSV for saving every generated completion. Use 'none' to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    records = load_prompt_records(args.data_dir)
    dataset = build_dataset(records)

    metrics_path: Optional[Path]
    if args.metrics_csv and str(args.metrics_csv).lower() not in {"", "none", "null"}:
        metrics_path = Path(args.metrics_csv)
    else:
        metrics_path = None

    metrics_tracker = MetricsTracker(metrics_path, args.metrics_ema_alpha) if metrics_path else None
    if metrics_path:
        print(
            f"[info] Streaming metrics to {metrics_path}. "
            f"Run watch_metrics.py --metrics-csv {metrics_path} for live plots."
        )

    generations_path: Optional[Path]
    if args.generations_csv and str(args.generations_csv).lower() not in {"", "none", "null"}:
        generations_path = Path(args.generations_csv)
    else:
        generations_path = None

    generation_logger = GenerationLogger(generations_path) if generations_path else None
    if generations_path:
        print(f"[info] Streaming generated completions to {generations_path}.")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
        print(f"[info] Detected CUDA devices ({device_count}): {', '.join(gpu_names)}")
    elif torch.backends.mps.is_available():
        print("[info] Detected Apple Metal (MPS) backend.")
    elif torch.backends.hip.is_built():
        device_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
        print(f"[info] Detected HIP devices ({device_count}): {', '.join(gpu_names)}")
    else:
        print("[warn] No GPU backend detected. Unsloth requires CUDA/HIP/XPU; training will fail on CPU-only setups.")

    reward_fn = build_q_reward_fn(
        args.invalid_penalty,
        args.q_reward_mode,
        args.q_temperature,
        metrics_tracker,
        generation_logger,
    )

    model, tokenizer = prepare_model_and_tokenizer(args)
    trainer = configure_trainer(args, model, tokenizer, dataset, reward_fn)
    target_steps = getattr(trainer.args, "_target_steps_per_generation", None)
    if target_steps is not None:
        trainer.args.steps_per_generation = target_steps
        trainer.args.__dict__.pop("_target_steps_per_generation", None)
        if getattr(trainer.args, "generation_batch_size", None) is None:
            global_batch = max(1, trainer.args.per_device_train_batch_size * getattr(trainer.args, "world_size", 1))
            trainer.args.generation_batch_size = global_batch * max(1, target_steps)

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

    save_on_rank_zero = getattr(trainer, "is_world_process_zero", None)
    should_save = save_on_rank_zero() if callable(save_on_rank_zero) else True

    if should_save:
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


if __name__ == "__main__":
    main()
