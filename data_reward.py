# 必要なライブラリをインポートします。
# os: ファイルシステム操作用
# random: ランダムサンプリング用
# re: 正規表現マッチング用
# glob: ファイルパターンマッチング用
# datasets: Hugging Faceのデータセットライブラリ用
import os
import random
import re
from glob import glob
from datasets import load_dataset, Dataset

# アクションを抽出するための正規表現パターン
# 形式: [0], [1], [2], [3] のようなアクションをマッチ
ACTION_RE = re.compile(r"^\[(\d)\]$")

# 無効なアクションに対するペナルティ値
PENALTY = -0.5


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
def reward_fn(
    completions,
    reward_action_0,
    reward_action_1,
    reward_action_2,
    reward_action_3,
    **kwargs,
):
    size = len(completions)
    # 各アクションの報酬をsizeまで拡張
    rewards_0 = _expand(reward_action_0, size)
    rewards_1 = _expand(reward_action_1, size)
    rewards_2 = _expand(reward_action_2, size)
    rewards_3 = _expand(reward_action_3, size)
    rewards = []
    for completion, r0, r1, r2, r3 in zip(
        completions, rewards_0, rewards_1, rewards_2, rewards_3
    ):
        # completionからアクションを正規表現で抽出
        match = ACTION_RE.match(completion.strip())
        if not match:
            # マッチしない場合ペナルティを適用
            rewards.append(PENALTY)
            continue
        action = int(match.group(1))
        # アクション0-3に対応する報酬を選択、無効ならペナルティ
        reward_map = (r0, r1, r2, r3)
        rewards.append(float(reward_map[action]) if 0 <= action < 4 else PENALTY)
    return rewards
