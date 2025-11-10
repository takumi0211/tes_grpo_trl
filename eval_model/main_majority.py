"""
LLMベースのTES制御シミュレーションのメインスクリプト（多数決版）

このスクリプトは、LLM（Large Language Model）を使用して
蓄熱システム（TES: Thermal Energy Storage）の制御を行い、
CO2排出量を最小化するシミュレーションを実行する。

主な機能：
- Hugging Face Transformersを使用したLLM制御
- 予測情報（負荷、CO2排出係数）を含むプロンプトの生成
- シミュレーション結果の可視化とログ記録
"""

import math
import os
import sys
from collections import Counter

import pandas as pd

# スクリプトのディレクトリをパスに追加（親ディレクトリのモジュールをインポートするため）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..")))

from eval_model.environment import ThermalStorageEnv
from eval_model.plot import plot_three_panels
from eval_model.llm import MODEL_ID as HF_MODEL_ID
from eval_model.llm_majority import SampledAction, get_llm_action_samples

# --- データ読み込み ---
# 3日間の時系列データ（負荷、CO2排出係数など）を読み込む
DATA_PATH = os.path.join(
    SCRIPT_DIR,
    "data_3day",
    "dynamic_co2_factor_hourly_3day_0505_to_0507.csv",
)
data = pd.read_csv(DATA_PATH)
data['datetime'] = pd.to_datetime(data['datetime'])  # datetime型に変換

# --- 環境構築 ---
# 強化学習環境（TES制御シミュレーション環境）を初期化
env = ThermalStorageEnv(external_data=data)


def _round_half_up(value: float) -> int:
    """
    四捨五入を行う（0.5の場合は切り上げ）
    
    Args:
        value: 丸める数値
    
    Returns:
        int: 丸められた整数値
    
    Note:
        Pythonの標準round()は銀行丸め（偶数丸め）を使用するため、
        一般的な四捨五入が必要な場合はこの関数を使用する
    """
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def build_forecast_prompt(env: ThermalStorageEnv, current_step: int) -> str:
    """
    予測情報をテキスト形式でフォーマットしてプロンプトを構築する
    
    Args:
        env: TES制御シミュレーション環境
        current_step: 現在のステップ番号
    
    Returns:
        str: 予測情報を含むプロンプトテキスト
    
    Note:
        以下の情報を含む：
        - 当日の残り営業時間（8:00-17:00）の負荷とCO2係数の予測
        - 翌日の計画指標（平均負荷、低CO2時間帯の平均、最小CO2係数）
        - 正規化されていない生の値を使用
    """
    if len(env.external_data) == 0:
        return "Forecast unavailable (no external data)."

    # 現在のステップに対応するデータ行を取得
    row_index = min(current_step, len(env.external_data) - 1)
    current_row = env.external_data.iloc[row_index]
    current_day = current_row["date"]  # 現在の日付
    current_hour = int(current_row["hour"])  # 現在の時刻

    # データフレームのカラム名
    load_column = "冷房需要[kW]"
    co2_column = "CO2排出係数[kg-CO2/kWh]"

    # プロンプトテキストの各行を格納するリスト
    lines = [
        "Forecast features (raw values aligned with the RL look-ahead):",
        "",
        "- Remaining business hours today (08:00-17:00, close at 18:00):",
    ]

    # 当日の営業時間の各時刻について予測情報を追加
    hour_map = env.day_hour_to_index.get(current_day, {})
    for hour in env.business_hours:
        timestamp = f"{hour:02d}:00"
        idx = hour_map.get(hour)
        
        # 過去の時刻は省略
        if hour < current_hour:
            lines.append(f"  - {timestamp}: past hour -> omitted (state uses zero)")
            continue
        
        # データが利用できない場合
        if idx is None:
            lines.append(f"  - {timestamp}: data unavailable")
            continue

        # 予測データを取得
        future_row = env.external_data.iloc[idx]
        load_value = future_row[load_column]
        co2_value = future_row[co2_column]
        load_display = _round_half_up(load_value)
        
        # 現在時刻には特別なマーカーを付ける
        if hour == current_hour:
            lines.append(
                f"  - {timestamp}: load={load_display} kW, co2={co2_value:.3f} kg-CO2/kWh  ←Now!!!"
            )
        else:
            lines.append(
                f"  - {timestamp}: load={load_display} kW, co2={co2_value:.3f} kg-CO2/kWh"
            )

    # 翌日の計画指標を取得
    next_day_load_avg = env._get_next_day_business_average(current_day, load_column)
    next_day_low_co2_avg = env._get_next_day_low_co2_average(current_day, co2_column)
    next_day_co2_min = env._get_next_day_co2_min(current_day, co2_column)

    # 翌日の計画指標をプロンプトに追加
    lines.append("")
    lines.append("- Next-day planning metrics (for terminal planning):")
    lines.append(f"  - load_mean={next_day_load_avg:.1f} kW (average cooling demand for next day)")
    lines.append(f"  - co2_low5_avg={next_day_low_co2_avg:.3f} kg-CO2/kWh (average of lowest 5 hours of CO2 factor for next day)")
    lines.append(f"  - co2_min={next_day_co2_min:.3f} kg-CO2/kWh (minimum CO2 factor for next day)")

    # 全ての行を改行で結合して返す
    return "\n".join(lines)

# --- モデル設定 ---
model_name = HF_MODEL_ID  # Hugging Faceで公開されているGRPOモデルID
SAMPLES_PER_PROMPT = 10   # 1プロンプトあたりの同時生成数

# LLMエラー時の再試行設定
MAX_LLM_RETRIES = 5  # エラー時の最大リトライ回数
LLM_ERROR_PATTERNS = ("Error occurred", "Randomly selected an action")  # エラーパターン


def _contains_error_pattern(text: str) -> bool:
    """Check whether the thought includes known error phrases or is empty."""

    thought_text = (text or "").strip()
    if not thought_text:
        return True
    return any(pattern in thought_text for pattern in LLM_ERROR_PATTERNS)


def _is_valid_sample(sample: SampledAction) -> bool:
    """A sample is valid only when parsing succeeded and no error pattern is present."""

    if not sample.parsed_successfully:
        return False
    return not _contains_error_pattern(sample.thought)


def summarize_majority(samples: list[SampledAction]) -> tuple[str, int, bool]:
    """Summarise batch outputs and return (summary_text, majority_action, has_valid_votes)."""

    if not samples:
        raise ValueError("No samples provided for majority voting.")

    valid_samples = [sample for sample in samples if _is_valid_sample(sample)]
    used_pool = valid_samples if valid_samples else samples

    counts = Counter(sample.action for sample in used_pool)
    best_action, best_support = max(counts.items(), key=lambda item: (item[1], -item[0]))

    lines = [
        (
            f"Majority vote selected action {best_action} "
            f"(support {best_support}/{len(used_pool)}; "
            f"valid_votes={len(valid_samples)}/{len(samples)})."
        )
    ]
    if not valid_samples:
        lines.append(
            "All samples were marked invalid (format errors or fallbacks); using the batch as-is."
        )

    lines.append("Sample breakdown:")
    for sample in samples:
        status_tokens: list[str] = []
        if _is_valid_sample(sample):
            status_tokens.append("valid")
        else:
            if not sample.parsed_successfully:
                status_tokens.append("parse_fallback")
            if _contains_error_pattern(sample.thought):
                status_tokens.append("error_text")
            if not status_tokens:
                status_tokens.append("invalid")
        status = ",".join(status_tokens)
        lines.append(
            f"- Sample {sample.sample_id}: action={sample.action}, status={status}, thought={sample.thought}"
        )

    return "\n".join(lines), best_action, bool(valid_samples)

# --- 出力ファイル名の生成 ---
# モデル名に含まれるコロンをアンダースコアに変換（ファイル名として使用可能にする）
model_name_name = model_name.replace(':', '_')

# --- 出力テキストファイル名とログファイル名 ---
text_name = f"results/{model_name_name}_majority_process.txt"
csv_log_name = f"results/{model_name_name}_majority_llm_logs.csv"

# --- 書き出しグラフ名 ---
output_name = f"results/{model_name_name}_majority_results"

# --- 出力先ディレクトリの作成 ---
os.makedirs(os.path.dirname(text_name), exist_ok=True)
os.makedirs(os.path.dirname(output_name), exist_ok=True)

# --- シミュレーション開始 ---
initial_tes_energy = 0.5  # 初期TES蓄熱量（正規化された値: 0-1の範囲）
state = env.reset(initial_tes_energy=initial_tes_energy)

# --- シミュレーション用の結果格納リスト ---
# 初期TESエネルギーを含めておく（時間軸の先行ズレ防止）
tes_energy_history = [env.tes_energy]  # TES蓄熱量の履歴
cooling_load_history = []               # 冷房需要の履歴
chiller_load_history = []               # ASHPの実際の出力の履歴
electric_consumption_history = []       # 電力消費量の履歴
cop_history = []                        # COP（成績係数）の履歴
reward_history = []                     # 報酬の履歴
carbon = 0.0                            # 累積CO2排出量
carbon_offset_total = 0.0               # 累積CO2オフセット量（TESへの蓄熱時）

# シミュレーションループ（プロセスログをファイルに記録しながら実行）
with open(text_name, 'w', encoding='utf-8') as f:
    while not env.done:
        # 現在の状態を取得
        current_step = env.current_step
        current_row_idx = min(current_step, len(env.external_data) - 1)
        current_time = env.external_data.iloc[current_row_idx]["datetime"]
        forecast_text = build_forecast_prompt(env, current_step)  # 予測情報のテキストを生成

        current_storage = env.tes_energy  # 現在のTES蓄熱量
        
        print('======================')
        print("Current Time:", current_time)
        
        # 時刻の取得と営業時間かどうかの判定
        hour = pd.to_datetime(current_time).hour
        is_control_time = env.is_business_hour(hour)  # 営業時間（8-17時）かどうか
        
        # 制御時間内の場合のみLLMによる制御を実行
        if is_control_time:
            attempt = 0
            thought = ""
            action_index = 0
            while True:
                attempt += 1
                samples = get_llm_action_samples(
                    current_time,
                    forecast_text,
                    current_storage,
                    model_name=model_name,
                    num_samples=SAMPLES_PER_PROMPT,
                )
                thought, action_index, has_valid_votes = summarize_majority(samples)

                # アクションインデックスを有効な範囲（0-3）にクリップ
                action_index = int(max(0, min(action_index, len(env.action_set) - 1)))

                if has_valid_votes:
                    break

                if attempt >= MAX_LLM_RETRIES:
                    print(
                        f"Majority voting could not obtain valid outputs after {attempt} attempts. Proceeding with fallback action {action_index}."
                    )
                    break

                print(
                    f"Majority voting produced only invalid samples (attempt {attempt}); retrying batch generation..."
                )

        else:
            # 制御時間外（営業時間外）の場合はASHPを停止
            thought = "制御時間外（8-17時以外・18時閉店）のため、ASHPを停止"
            action_index = 0  # アクション0 = ASHP停止

        # コンソールに出力
        print('====')
        print("Thought Process:", thought)
        print('====')
        print("Action Index:", action_index)
        print('======================')
        
        # ファイルに記録
        f.write('======================\n')
        f.write(f"Current Time: {current_time}\n")
        f.write(f"Current TES energy [kWh]: {current_storage}\n")
        f.write('====\n')
        f.write(f"Thought Process: {thought}\n")
        f.write('====\n')
        f.write(f"Action Index: {action_index}\n")
        f.write('======================\n')
        
        # 環境のステップを実行（アクションを適用）
        state, reward, _learning_done, cooling_load, cop, carbon_discharge = env.step(action_index)

        # 電力消費量を計算（冷房出力 / COP）
        electric_consumption = cooling_load / cop

        # 各種データを履歴に記録
        tes_energy_history.append(env.tes_energy)
        cooling_load_history.append(
            data.iloc[min(env.current_step - 1, len(data) - 1)]['冷房需要[kW]']
        )
        chiller_load_history.append(cooling_load)
        electric_consumption_history.append(electric_consumption)
        cop_history.append(cop)
        reward_history.append(reward)

        # CO2排出量を累積
        carbon += carbon_discharge
        carbon_offset_total += env.last_carbon_offset

        # シミュレーション終了時の処理
        if env.done:
            f.write('======================\n')
            net_carbon = carbon - carbon_offset_total  # 正味のCO2排出量
            f.write('======================\n')
            f.write(f'総CO2排出量 (報酬換算): {net_carbon}kg-CO2\n')

# --- 結果まとめ ---
# シミュレーション結果を辞書にまとめる
results = {
    'tes_energy_history': tes_energy_history,                   # TES蓄熱量の履歴
    'cooling_load_history': cooling_load_history,               # 冷房需要の履歴
    'chiller_load_history': chiller_load_history,               # ASHP出力の履歴
    'electric_consumption_history': electric_consumption_history, # 電力消費量の履歴
    'reward_history': reward_history,                           # 報酬の履歴
    'cop_history': cop_history                                  # COPの履歴
}

# 正味のCO2排出量を計算して表示
net_carbon = carbon - carbon_offset_total
print(net_carbon)

# --- プロット ---
# 結果を可視化してPNGファイルに保存
plot_three_panels(data, results, output_name=output_name)
