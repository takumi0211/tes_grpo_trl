#!/usr/bin/env python3
"""
GRPO fine-tuning entrypoint powered by Hugging Face Transformers/TRL instead of Unsloth.

This script mirrors the reward shaping and data handling used in train_grpo.py but
loads the base model via AutoModelForCausalLM, applies LoRA adapters with PEFT,
and enables FlashAttention-friendly configuration for H100 GPUs.
"""
# ======================================================================================
# ここから下は Hugging Face ベースの GRPO トレーナー実装です。
# 日本語による詳細コメントを大量に入れており、各処理の意図や注意点を
# 手元で素早く把握できるようにしています。英語コメントと併記する代わりに、
# セクションごとに「なぜこの設定が必要なのか」「H100 で気を付けるべき点は何か」
# といった背景知識を文章で記録しています。
# ======================================================================================

# --------------------------------------------------------------------------------------
# 全体構成メモ（日本語）
# --------------------------------------------------------------------------------------
# - データ読み込み部では CSV → datasets.Dataset への変換を行い、元々の unsloth 版と
#   全く同じ形式で Q 値を扱います。
# - 報酬関数は train_grpo.py のロジックをそのまま移植し、`train_grpo_hf.py` からでも
#   同じ動作になるようにしています。評価者が変わっても学習結果の比較が容易です。
# - モデル準備では AutoModelForCausalLM + PEFT を利用し、LoRA や量子化、FlashAttention を
#   自由に切り替えできます。H100 前提なので bf16・flash-attn がデフォルトです。
# - `GRPOTrainer` の設定は TRL 0.22 の API に合わせていますが、既存スクリプトの挙動と
#   ずれないようにカスタムロジック（シードや steps_per_generation の再適用など）を残しています。
# - 最後に LoRA アダプタとマージ済みモデルを保存する処理を実装しています。Unsloth の
#   FastLanguageModel とは異なり merge_and_unload を使う点に注意してください。
# --------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer


# Conversation parsing -----------------------------------------------------------------

# ここでは CSV に保存された文字列ベースのプロンプトを、ChatML 互換の辞書リストに変換します。
# 元データが `system:` や `user:` プレフィックスで連結されている前提なので、それを正規表現で
# 分解して Hugging Face の chat_template で扱えるよう整形します。
# Unsloth 版と完全互換にするため、細かいフォーマットも既存処理と同じです。
ROLE_PREFIX = re.compile(r"^(system|user|assistant):\s*", re.IGNORECASE)


def _flush_message(messages: list[dict[str, str]], role: str | None, lines: list[str]) -> None:
    # lines バッファに貯めている行を 1 メッセージとして確定させます。
    # role が None の場合はまだロールを特定できていないのでスキップします。
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
    # 元々の CSV は 1 カラム内に段落を詰め込んでおり、行頭に "role:" が出現するたびに
    # 新しいメッセージとして扱う必要があります。ここではそのルールを忠実に再現しています。
    # この形式に合わせておくことで、報酬関数やメトリクス周りが既存コードと同じ挙動になります。
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
    # PromptRecord は 1 サンプルに含まれる情報を凝縮したデータ構造です。
    # prompt: チャット形式のリスト。q_values: 4 行動に対する事前推定 Q 値。
    # optimal_action: 最適とラベル付けされたアクション。metadata: 解析用メタ情報。
    prompt: list[dict[str, str]]
    q_values: list[float]
    optimal_action: int
    metadata: dict[str, object]


def _safe_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value))


def load_prompt_records(data_root: Path) -> list[PromptRecord]:
    # data_root 直下の *_q_dataset.csv をすべて走査し、GRPO 用のサンプルリストを作ります。
    # 元のノートブックが複数時間帯の CSV をまとめて扱っていたため、その構造を尊重しています。
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
    # Hugging Face datasets は PyTorch や transformers とシームレスに連携できるため、
    # ここで Python のリストから Dataset を構築しておきます。JSON 文字列化した metadata を
    # 一緒に保持することで後段のロギングでも情報損失を避けています。
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
        # metrics.csv が指定されている場合はヘッダー行を作成します。
        # 後から pandas や可視化ツールで読みやすいよう、小数点以下 6 桁に揃える設計です。
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

        # GRPO はロールアウトのばらつきが大きいので、EMA を使って指標を平滑化します。
        # parse_rate: `[digit]` フォーマットに従った生成割合、acc_opt: 最適行動を当てた割合。
        # batch_regret/batch_reward は 1 ステップ分の平均値で、None のケースもあり得ます。
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
        # generations.csv はデバッグ用途で超大きくなる可能性があります。
        # そのため初期化時に存在確認と親ディレクトリの作成を済ませておき、
        # ラン中の I/O エラーを最小限に抑えています。

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
        # completion は文字列にならない場合（リスト/辞書）もあるため、json.dumps で統一的に保存します。
        # ensure_ascii=True で制御することで、CSV が ASCII のみで構成されダンプしやすくなります。
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

    # 生成テキストの最終行から `[数字]` を抜き出します。余計なコメントや文章があると
    # 報酬が invalid_penalty になります。学習中のモデルがフォーマット遵守できているかを
    # この関数が担保します。
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
    # 数値安定性のために最大値シフトを行いつつソフトマックスを計算します。
    # Q 値の差分が大きくてもオーバーフローしないように max subtraction を入れています。
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
        # trainer_state からグローバルステップを取り出し、EMA ロギングに利用します。
        # DDP でも rank0 だけが CSV を更新するように制御しています。
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
                # Q 値の最大値との差 (advantage) を使うので max_q を計算しておきます。
                # reward_mode=softmax の時は advantage を温度付きで確率に変換します。
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
    # `checkpoint-00010` のようなディレクトリ名を数値部分でソートし、最新を返します。
    # 途中で別ファイルが紛れ込んでも安全に動くよう int 変換エラーを避ける実装です。
    checkpoints = sorted(
        (p for p in output_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda path: int(path.name.split("-")[-1]) if path.name.split("-")[-1].isdigit() else -1,
    )
    if checkpoints:
        return checkpoints[-1]
    return None


# Trainer assembly ---------------------------------------------------------------------

# LoRA を適用する対象モジュール一覧です。公式 Cookbook の推奨に合わせ、
# デフォルトは注意機構の q/k/v/o のみに絞っています。
# MoE MLP の gate/down へ適用する場合は target_parameters で experts.* を明示します。
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]

# GPT-OSS 20B では 4 層おきに MoE experts が存在し、OpenAI Cookbook のレシピでは
# 7, 15, 23 層目の gate_up_proj / down_proj に LoRA を挿す構成が紹介されています。
# それに倣ってデフォルト値を定義し、必要なら CLI で上書きできるようにします。
DEFAULT_TARGET_PARAMETERS = [
    "7.mlp.experts.gate_up_proj",
    "7.mlp.experts.down_proj",
    "15.mlp.experts.gate_up_proj",
    "15.mlp.experts.down_proj",
    "23.mlp.experts.gate_up_proj",
    "23.mlp.experts.down_proj",
]


def _inspect_lora_targets(model, target_modules: list[str], target_parameters: Optional[list[str]]) -> tuple[dict[str, int], dict[str, int]]:
    """
    LoRA 適用対象が実際にモデル内で何個ヒットするかを集計します。
    target_modules はモジュールのサフィックス、target_parameters は完全修飾パラメータ名を想定。
    """
    module_hits: dict[str, int] = {name: 0 for name in target_modules}
    for module_name, _ in model.named_modules():
        for target in target_modules:
            if module_name.endswith(target):
                module_hits[target] += 1

    param_hits: dict[str, int] = {}
    if target_parameters:
        param_hits = {name: 0 for name in target_parameters}
        for param_name, _ in model.named_parameters():
            for target in target_parameters:
                if target in param_name:
                    param_hits[target] += 1

    return module_hits, param_hits


def infer_prompt_stats(dataset: Dataset, tokenizer, add_generation_prompt: bool = True) -> tuple[int, float]:
    # 学習前にプロンプト長を見積もって、max_seq_length とのバッファを計算します。
    # 生成トークン上限 (max_completion_length) を安全に設定するため、最大長と平均値を両方返します。
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


def _maybe_import_bitsandbytes():
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        # bitsandbytes が入っていないと 4bit/8bit 量子化のロードができないため、
        # 明示的に例外を投げて CLI 側にヒントを表示します。
        raise RuntimeError(
            "bitsandbytes is not installed. Install it or set --quantization none to disable 4/8-bit loading."
        ) from exc
    return BitsAndBytesConfig


def prepare_model_and_tokenizer(args) -> tuple[torch.nn.Module, object]:
    # H100 + PyTorch 2 系では TF32 が有効だと matmul が高速化されるため、ここで有効化します。
    # さらに `torch.set_float32_matmul_precision("high")` により推論/学習の精度と速度のバランスを確保します。
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    tokenizer_name = args.tokenizer_name or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=not args.disable_fast_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    # 一部モデルは pad_token が未定義なので EOS を流用します。左パディングはチャットテンプレートと
    # GRPO の兼ね合いで必要（生成時に新トークンが末尾に積み上がる）です。
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = args.max_seq_length

    dtype = torch.bfloat16 if args.compute_dtype == "bf16" else torch.float16
    load_kwargs: dict[str, object] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": args.trust_remote_code,
    }

    # device_map=auto をデフォルトにしておけば、複数 GPU や CPU offload にも対応しやすくなります。
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map

    # Transformers 4.36+ では attn_implementation で FlashAttention などを明示できます。
    # H100 だと flash_attention_2 が最速ですが、環境に合わせて上書き可能にしています。
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation

    if args.quantization != "none":
        BitsAndBytesConfig = _maybe_import_bitsandbytes()
        if args.quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=not args.disable_double_quant,
                bnb_4bit_quant_type=args.bnb_quant_type,
                bnb_4bit_compute_dtype=dtype,
            )
        else:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["quantization_config"] = quant_config

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)

    gc_kwargs = args.gradient_checkpointing_kwargs or {}
    # 量子化している場合は PEFT で LoRA を差し込む前に prepare_model_for_kbit_training を通します。
    # これにより LayerNorm の統合や requires_grad の設定が適切に行われます。
    if args.quantization != "none":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=False,
        )
        model.enable_input_require_grads()

    # LoRA 適用対象モジュールの指定。CLI から自由に調整できるよう文字列 → リスト化しています。
    if args.peft_target_modules:
        target_modules = [m.strip() for m in args.peft_target_modules.split(",") if m.strip()]
    else:
        target_modules = DEFAULT_TARGET_MODULES
    target_parameters = (
        [p.strip() for p in args.peft_target_parameters.split(",") if p.strip()]
        if args.peft_target_parameters and args.peft_target_parameters.lower() not in {"", "none", "null"}
        else None
    )
    if target_parameters:
        # GPT-OSS 20B の MoE 層は experts 配下にあるため、Cookbook の通りに完全修飾名で指定します。
        print(f"[info] Applying LoRA to experts via target_parameters: {target_parameters}")

    module_hits, param_hits = _inspect_lora_targets(model, target_modules, target_parameters)
    missing_modules = [name for name, count in module_hits.items() if count == 0]
    if missing_modules:
        print(f"[warn] No modules matched LoRA target suffixes: {', '.join(missing_modules)}")
    if target_parameters:
        missing_params = [name for name, count in param_hits.items() if count == 0]
        if missing_params:
            print(f"[warn] No parameters matched LoRA target_parameters: {', '.join(missing_params)}")

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha or args.lora_rank * 2,
        target_modules=target_modules,
        target_parameters=target_parameters,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # 勾配チェックポイントはメモリ節約に必須。H100 でも 20B モデルは巨大なのでデフォルト有効です。
    if not args.disable_gradient_checkpointing:
        model.gradient_checkpointing_enable(**gc_kwargs)

    # generation_config.use_cache を True, 学習中の use_cache を False にして再計算を防ぎます。
    model.config.use_cache = False
    model.generation_config.use_cache = True

    if args.torch_compile:
        compile_kwargs = {}
        if args.compile_mode:
            compile_kwargs["mode"] = args.compile_mode
        compile_kwargs["fullgraph"] = args.compile_fullgraph
        model = torch.compile(model, **compile_kwargs)

    return model, tokenizer


def configure_trainer(
    args,
    model,
    tokenizer,
    dataset: Dataset,
    reward_fn,
) -> GRPOTrainer:
    # まずプロンプト長から生成上限を算出し、max_seq_length を超えないようにします。
    # ここで例外を出しておくと学習開始前に設定ミスへ即座に気付けます。
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

    user_generation_batch = args.generation_batch_size
    if user_generation_batch is not None and user_generation_batch <= 0:
        raise ValueError("--generation-batch-size must be a positive integer when provided.")

    user_steps_per_generation = args.steps_per_generation
    if user_steps_per_generation is not None and user_steps_per_generation <= 0:
        raise ValueError("--steps-per-generation must be a positive integer when provided.")

    if user_generation_batch is not None and user_steps_per_generation is not None:
        raise ValueError("Specify at most one of --generation-batch-size or --steps-per-generation.")

    if user_steps_per_generation is not None:
        steps_per_generation = user_steps_per_generation
        generation_batch_size_arg = user_generation_batch
    elif user_generation_batch is not None:
        steps_per_generation = None
        generation_batch_size_arg = user_generation_batch
    else:
        steps_per_generation = num_generations
        generation_batch_size_arg = None

    # TRL の GRPOConfig は PPOConfig 由来で、bf16/fp16 フラグを手動で制御します。
    # 既定では bf16=True としつつ、ユーザが fp16 を明示した場合にも対応しています。
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        bf16=args.compute_dtype == "bf16",
        fp16=args.compute_dtype == "fp16",
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
        generation_batch_size=generation_batch_size_arg,
        steps_per_generation=steps_per_generation,
        temperature=args.temperature,
        top_p=args.top_p,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        report_to=args.report_to,
        remove_unused_columns=False,
        optim=args.optim,
    )

    # generation_batch_size を未指定の場合、world_size を使って暗黙的に決まる値を計算し、
    # steps_per_generation を後から trainer に戻すために一旦 _target_steps... に退避させます。
    global_batch = max(1, training_args.per_device_train_batch_size * getattr(training_args, "world_size", 1))
    effective_generation_batch = training_args.generation_batch_size or global_batch
    computed_steps = (
        training_args.steps_per_generation
        if training_args.steps_per_generation is not None
        else max(1, (effective_generation_batch or global_batch) // global_batch)
    )
    training_args.__dict__["_target_steps_per_generation"] = computed_steps
    training_args.steps_per_generation = None

    # 実行前にプロンプト長と推定マイクロステップ数をログとして吐いておくと、
    # 実験ノートに貼る際に便利です。平均長も表示してプロンプト分布を把握します。
    print(
        f"[info] max_prompt_tokens={max_prompt_length} "
        f"(avg={avg_prompt_length:.1f}), max_completion_tokens={max_completion_length}"
    )
    if training_args.num_generations and effective_generation_batch:
        print(
            f"[info] Effective generation batch size={effective_generation_batch}, "
            f"micro-steps per GRPO step≈{computed_steps}"
        )

    return GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        reward_func=reward_fn,
    )
    # トレーナーは reward_func を直接受け取る特別なコンストラクタを持っており、
    # ここで構築した関数が学習ループ内から呼び出されます。


# CLI -------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    # コマンドラインオプションは元スクリプトと互換性を保ちながら、Hugging Face 版で必要な
    # 追加パラメータ（量子化や attn 実装など）を拡張しています。日本語コメントでは
    # 主に「どの環境で何を指定すべきか」のヒントを残しています。
    parser = argparse.ArgumentParser(description="GRPO finetuning with Q-value rewards (Hugging Face stack).")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing *_q_dataset.csv files.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="unsloth/gpt-oss-20b",
        help="Base checkpoint to finetune. Defaults to the public GPT-OSS 20B release.",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=None,
        help="Optional tokenizer identifier. Defaults to --model-name when omitted.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow execution of custom modeling code (required for some community checkpoints).",
    )
    parser.add_argument(
        "--disable-fast-tokenizer",
        action="store_true",
        help="Force use of the slow tokenizer implementation.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=5000,
        help="Total max sequence length (prompt + completion). Defaults to 5000 tokens for long reasoning windows.",
    )
    parser.add_argument("--lora-rank", type=int, default=4, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha; defaults to 2 * lora_rank.")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout probability.")
    parser.add_argument(
        "--peft-target-modules",
        type=str,
        default=",".join(DEFAULT_TARGET_MODULES),
        help=(
            "Comma-separated list of module suffixes to apply LoRA to "
            "(defaults to attention projections: q_proj,k_proj,v_proj,o_proj)."
        ),
    )
    parser.add_argument(
        "--peft-target-parameters",
        type=str,
        default=",".join(DEFAULT_TARGET_PARAMETERS),
        help=(
            "Comma-separated fully-qualified parameter names for MoE experts, e.g. "
            "'7.mlp.experts.gate_up_proj,7.mlp.experts.down_proj'. "
            "Use this to follow the official GPT-OSS MoE guidance."
        ),
    )
    parser.add_argument("--num-generations", type=int, default=4, help="Number of completions sampled per prompt.")
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=None,
        help=(
            "Batch size used during response generation. Must be divisible by num-generations. "
            "Leave unset to let --steps-per-generation control the schedule."
        ),
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
    parser.add_argument("--invalid-penalty", type=float, default=-0.5, help="Reward assigned to malformed outputs.")
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
        "--skip-save-merged",
        action="store_true",
        help="Disable saving merged full weights after training.",
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
    parser.add_argument(
        "--quantization",
        choices=["4bit", "8bit", "none"],
        default="4bit",
        help="Quantization mode for model loading. Set to 'none' for full-precision weights.",
    )
    # 量子化関連の指定は H100 で 4bit + LoRA が最も省メモリ・高効率なのでデフォルトを 4bit にしています。
    # bitsandbytes の wheel が見つからない場合は `--quantization none` を指定すると FP16/BF16 の全精度ロードになります。
    parser.add_argument(
        "--compute-dtype",
        choices=["bf16", "fp16"],
        default="bf16",
        help="Computation dtype for forward/backward passes.",
    )
    parser.add_argument(
        "--bnb-quant-type",
        choices=["nf4", "fp4"],
        default="nf4",
        help="Quantization type when --quantization=4bit.",
    )
    parser.add_argument(
        "--disable-double-quant",
        action="store_true",
        help="Disable double quantization for bitsandbytes 4-bit loaders.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map passed to from_pretrained. Use 'none' to disable automatic placement.",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation hint passed to Transformers (e.g., flash_attention_2, sdpa, eager).",
    )
    parser.add_argument(
        "--disable-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing. Enabled by default for memory savings.",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile for the model (best on PyTorch 2.1+).",
    )
    # torch.compile は追加の最適化を行う一方で JIT に時間がかかるため、必要な実験だけで有効化することを想定。
    # H100 では max-autotune モードが有効なケースが多いので、--compile-mode で細かく調整できます。
    parser.add_argument(
        "--compile-mode",
        type=str,
        default=None,
        help="Optional torch.compile mode, e.g. 'max-autotune' or 'reduce-overhead'.",
    )
    parser.add_argument(
        "--compile-fullgraph",
        action="store_true",
        help="Request a full-graph compile.",
    )
    parser.add_argument(
        "--gradient-checkpointing-kwargs",
        type=json.loads,
        default=None,
        help="Optional JSON dict forwarded to gradient checkpointing APIs.",
    )
    return parser.parse_args()


def main() -> None:
    # メインフローは以下の順に進みます：
    # 1) 引数パースと乱数シード設定
    # 2) データ読み込みと Dataset 化
    # 3) メトリクス・生成ロガー初期化（オプション）
    # 4) GPU/バックエンド検出ログ
    # 5) 報酬関数の構築
    # 6) モデル + トークナイザ準備、GRPO トレーナー生成
    # 7) 学習ループ実行とチェックポイント処理
    # 8) LoRA/マージ済みモデル保存
    args = parse_args()
    set_seed(args.seed)

    records = load_prompt_records(args.data_dir)
    dataset = build_dataset(records)

    metrics_path: Optional[Path]
    if args.metrics_csv and str(args.metrics_csv).lower() not in {"", "none", "null"}:
        metrics_path = Path(args.metrics_csv)
    else:
        metrics_path = None
    # metrics.csv を指定しない場合は None にして I/O を完全に無効化します。
    # 学習用クラスタで NFS が遅い場合などに有効です。
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

    # generations.csv もオプション扱い。巨大化しやすいので必要なときだけ有効化します。
    generation_logger = GenerationLogger(generations_path) if generations_path else None
    if generations_path:
        print(f"[info] Streaming generated completions to {generations_path}.")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
        print(f"[info] Detected CUDA devices ({device_count}): {', '.join(gpu_names)}")
        if args.attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
                print("[info] flash-attn available; using FlashAttention kernels.")
            except ImportError:
                print("[warn] flash-attn package not found. Install flash-attn for FlashAttention kernels on H100.")
        # CUDA がある場合は GPU 名を一覧表示。クラスタで複数 GPU を使う際の確認に便利です。
    elif torch.backends.mps.is_available():
        print("[info] Detected Apple Metal (MPS) backend.")
    elif hasattr(torch.backends, "hip") and torch.backends.hip.is_built():
        device_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
        print(f"[info] Detected HIP devices ({device_count}): {', '.join(gpu_names)}")
    else:
        print("[warn] No GPU backend detected. Training large models without GPUs is impractical.")

    reward_fn = build_q_reward_fn(
        args.invalid_penalty,
        args.q_reward_mode,
        args.q_temperature,
        metrics_tracker,
        generation_logger,
    )
    # 報酬関数はデータセットから渡される Q 値をもとに Advantage/Softmax のどちらかでスカラー報酬を返します。
    # これにより Unsloth 版と完全に同じ学習信号を得られます。

    model, tokenizer = prepare_model_and_tokenizer(args)
    trainer = configure_trainer(args, model, tokenizer, dataset, reward_fn)
    # configure_trainer 内部で steps_per_generation をリセットしているため、
    # ここで再適用して最終的に想定どおりのマイクロバッチ動作に戻します。
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
        # `--resume-from-checkpoint latest` とすると自動で最新チェックポイントを探す仕様です。
        # それ以外のパスは存在チェックを行い、なければ警告を出して新規学習を続行します。
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
    print("[info] Training complete.")

    save_on_rank_zero = getattr(trainer, "is_world_process_zero", None)
    should_save = save_on_rank_zero() if callable(save_on_rank_zero) else True

    if should_save:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if not args.skip_save_lora:
            # LoRA アダプタは推論時に簡単に再ロードできるよう専用ディレクトリへ保存します。
            # tokenizer も同時に保存しておくと ChatML テンプレートがズレる心配がありません。
            adapter_dir = args.output_dir / "lora_adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            trainer.model.save_pretrained(adapter_dir)
            tokenizer.save_pretrained(adapter_dir)
            print(f"[info] Saved LoRA adapter and tokenizer to {adapter_dir}")

        if not args.skip_save_merged:
            # merge_and_unload は LoRA を本体に統合した状態でモデルを戻してくれます。
            # こちらも tokenizer を一緒に保存し、推論バッチでそのまま利用可能にしています。
            merged_dir = args.output_dir / "merged_model"
            merged_dir.mkdir(parents=True, exist_ok=True)
            base_model = trainer.model.merge_and_unload()
            base_model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            print(f"[info] Saved merged model to {merged_dir}")


if __name__ == "__main__":
    main()
