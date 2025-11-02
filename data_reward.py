# 必要なライブラリをインポートします。
# os: ファイルシステム操作用
# random: ランダムサンプリング用
# re: 正規表現マッチング用
# glob: ファイルパターンマッチング用
# datasets: Hugging Faceのデータセットライブラリ用
import os
import random
import re
import csv
import atexit
from collections.abc import Iterable as IterableSequence
from glob import glob
from datasets import load_dataset, Dataset
from typing import Optional, Sequence, Union

TRUNCATION_TOKEN_THRESHOLD = int(os.getenv("GRPO_TRUNCATION_THRESHOLD", 128))

# アクションを抽出するための正規表現パターン
# 形式: [0], [1], [2], [3] を文中から検出
ACTION_RE = re.compile(r"\[(\d)\]")

# 無効なアクションに対するペナルティ値
PENALTY = -0


def _is_main_process() -> bool:
    rank = os.environ.get("RANK")
    if rank not in (None, "", "0"):
        return False
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank not in (None, "", "0"):
        return False
    return True


CSV_LOG_ENABLED = os.getenv("GRPO_LOG_COMPLETIONS", "1") != "0" and _is_main_process()
CSV_PATH = os.getenv(
    "GRPO_COMPLETION_LOG_PATH",
    os.path.join("runs", "micro_step_completions.csv"),
)
_CSV_FILE = None
_CSV_WRITER = None
_MICRO_STEP_COUNTER = 0


def _init_csv_logger() -> None:
    global _CSV_FILE, _CSV_WRITER
    if not CSV_LOG_ENABLED or _CSV_WRITER is not None:
        return
    os.makedirs(os.path.dirname(CSV_PATH) or ".", exist_ok=True)
    _CSV_FILE = open(CSV_PATH, "w", newline="", encoding="utf-8")
    _CSV_WRITER = csv.writer(_CSV_FILE)
    _CSV_WRITER.writerow(["step", "prompt", "completion", "action", "reward"])


def _close_csv_logger() -> None:
    global _CSV_FILE, _CSV_WRITER
    if _CSV_FILE is not None:
        _CSV_FILE.flush()
        _CSV_FILE.close()
    _CSV_FILE = None
    _CSV_WRITER = None


atexit.register(_close_csv_logger)


# プロンプトデータセットを読み込む関数
def load_prompt_dataset(data_dir: str = "data", harmony_only: bool = True) -> Dataset:
    """Load all CSV prompts (defaults to Harmony-formatted files)."""

    def _list(pattern):
        return sorted(glob(os.path.join(data_dir, pattern)))

    files = _list("*_harmony.csv") if harmony_only else _list("*.csv")
    if not files and harmony_only:
        # Harmony変換前の環境でも動くようにフォールバック
        files = _list("*.csv")
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return load_dataset("csv", data_files=files, split="train")


# データセットからランダムにk個のサンプルを抽出する関数
# dataset: 入力データセット
# k: 抽出するサンプル数
# 戻り値: サンプリングされたDatasetオブジェクト
def sample_batch(dataset: Dataset, k: int) -> Dataset:
    # データセットのインデックスからランダムにk個を選択
    idx = random.sample(range(len(dataset)), k)
    # 選択されたインデックスに基づいてサブセットを作成
    return dataset.select(idx)


# 値を指定サイズまで拡張するヘルパー関数 (内部使用)
# values: 拡張元の値リスト
# size: 目標サイズ
# 戻り値: 拡張されたリスト (繰り返しや追加でサイズを合わせる)
def _expand(values, size):
    seq = list(values)
    if not seq or size == 0:
        return [0.0] * size
    if len(seq) == size:
        return seq
    repeat = max(1, size // len(seq))
    expanded = []
    for val in seq:
        expanded.extend([val] * repeat)
    while len(expanded) < size:
        expanded.extend(seq)
    return expanded[:size]


# 報酬計算のメイン関数
# completions: 完了したアクションのリスト (例: "[0]", "[1]" など)
# reward_action_0 ~ reward_action_3: 各アクション(0-3)に対する報酬値 (スカラまたはリスト)
# **kwargs: 追加のキーワード引数 (未使用)
# 戻り値: 各completionに対する報酬のリスト
def _to_list(value: Optional[Union[Sequence, "torch.Tensor"]]) -> list:
    """Convert tensors/sequences to a plain list without importing torch globally."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        try:
            return list(value.tolist())
        except TypeError:
            return list(value)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def reward_fn(
    completions,
    reward_action_0,
    reward_action_1,
    reward_action_2,
    reward_action_3,
    **kwargs,
):
    global _MICRO_STEP_COUNTER

    size = len(completions)
    rewards_0 = _expand(reward_action_0, size)
    rewards_1 = _expand(reward_action_1, size)
    rewards_2 = _expand(reward_action_2, size)
    rewards_3 = _expand(reward_action_3, size)

    completion_tokens_raw = kwargs.get("completion_ids")
    completion_masks_raw = kwargs.get("completion_mask")
    token_sequences = []
    if isinstance(completion_tokens_raw, IterableSequence) and not isinstance(
        completion_tokens_raw, (str, bytes)
    ):
        token_sequences = [_to_list(tokens) for tokens in completion_tokens_raw]
    mask_sequences = []
    if isinstance(completion_masks_raw, IterableSequence) and not isinstance(
        completion_masks_raw, (str, bytes)
    ):
        mask_sequences = [_to_list(mask) for mask in completion_masks_raw]
    if mask_sequences and not token_sequences:
        mask_sequences = []
    if token_sequences and not mask_sequences:
        mask_sequences = [[] for _ in token_sequences]
    max_completion_len = max((len(seq) for seq in token_sequences), default=0)

    rewards = []
    action_tokens = []
    for idx, (completion, r0, r1, r2, r3) in enumerate(
        zip(completions, rewards_0, rewards_1, rewards_2, rewards_3)
    ):
        is_truncated = False
        if token_sequences and idx < len(token_sequences):
            tokens_seq = token_sequences[idx]
            mask_seq = mask_sequences[idx] if idx < len(mask_sequences) else []
            seq_len = len(tokens_seq)
            mask_all_one = bool(mask_seq) and all(int(v) == 1 for v in mask_seq)
            if (
                seq_len >= max_completion_len
                and seq_len >= TRUNCATION_TOKEN_THRESHOLD
                and mask_all_one
            ):
                is_truncated = True
        if is_truncated:
            rewards.append(PENALTY)
            action_tokens.append("NaN")
            continue

        matches = list(ACTION_RE.finditer(completion))
        if not matches:
            rewards.append(PENALTY)
            action_tokens.append("NaN")
            continue
        action = matches[-1].group(1)
        reward_map = (r0, r1, r2, r3)
        idx_int = int(action)
        reward_val = float(reward_map[idx_int]) if 0 <= idx_int < 4 else PENALTY
        rewards.append(reward_val)
        action_tokens.append(action if 0 <= idx_int < 4 else "NaN")

    if CSV_LOG_ENABLED and size:
        _init_csv_logger()
        if _CSV_WRITER is not None:
            prompts = kwargs.get("prompt") or kwargs.get("prompts") or []
            if isinstance(prompts, str):
                prompts = [prompts]
            trainer_state = kwargs.get("trainer_state")
            global_step = getattr(trainer_state, "global_step", -1)
            step_value = global_step if (global_step is not None and global_step >= 0) else _MICRO_STEP_COUNTER
            for idx, (completion, reward_val, action_token) in enumerate(
                zip(completions, rewards, action_tokens)
            ):
                prompt_text = ""
                if prompts:
                    prompt_text = prompts[idx % len(prompts)]
                _CSV_WRITER.writerow(
                    [
                        step_value,
                        prompt_text,
                        completion,
                        action_token,
                        float(reward_val),
                    ]
                )
            if _CSV_FILE is not None:
                _CSV_FILE.flush()
        _MICRO_STEP_COUNTER += 1

    return rewards
