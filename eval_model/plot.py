# plot.py
# シミュレーション結果を可視化するためのプロット関数モジュール
# 3つのパネル（冷房負荷、TES蓄熱量、消費電力とCO2排出係数）を含むグラフを生成

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def _ensure_parent_dir(path: str) -> None:
    """
    ファイルパスの親ディレクトリが存在することを確認し、
    存在しない場合は作成する
    
    Args:
        path (str): ファイルパス
    """
    directory = Path(path).resolve().parent
    directory.mkdir(parents=True, exist_ok=True)


def _clip_or_pad(series: np.ndarray, target_length: int) -> np.ndarray:
    """
    配列を目標の長さに調整する（切り詰めまたはパディング）
    
    Args:
        series (np.ndarray): 入力配列
        target_length (int): 目標の長さ
    
    Returns:
        np.ndarray: 調整後の配列
    """
    # 配列が目標の長さ以上の場合は切り詰め
    if len(series) >= target_length:
        return series[:target_length]
    # 配列が空の場合はゼロで埋める
    if len(series) == 0:
        return np.zeros(target_length)
    # 配列が短い場合は最後の値でパディング
    pad_count = target_length - len(series)
    return np.concatenate([series, np.repeat(series[-1], pad_count)])


def plot_three_panels(data, results, output_name='three_panels.png'):
    """
    シミュレーション結果を3つのパネルで可視化
    
    パネル構成:
    1. 冷房負荷（Cooling Load）とチラー負荷（Chiller Load）
    2. TES蓄熱量（TES Energy）
    3. 消費電力（Electric Consumption）とCO2排出係数（CO2 Intensity）
    
    Args:
        data (pd.DataFrame): 外部データ（冷房需要、CO2排出係数など）
        results (dict): シミュレーション結果の辞書
        output_name (str): 出力ファイル名（デフォルト: 'three_panels.png'）
    """
    # =========================
    # 時間軸の準備
    # =========================
    # datetime列から時間軸を作成（なければデフォルトで生成）
    if 'datetime' in data.columns:
        datetime_series = pd.to_datetime(data['datetime'])
    else:
        # datetime列がない場合、2024年1月1日から1時間ごとのデータと仮定
        datetime_series = pd.date_range(start='2024-01-01', periods=len(data), freq='h')
    
    # シミュレーション結果用の時間軸を準備
    # TES Energyは初期値を含むため、他の履歴より1つ多い
    steps_tes = datetime_series[:len(results['tes_energy_history'])]
    steps = datetime_series[:len(results['electric_consumption_history'])]
    electric_consumption = np.asarray(results['electric_consumption_history'])
    
    # プロットを4日分に延長（96時間 = 4日間）
    # シミュレーションは3日分だが、視覚的に次の日の予測を表示するため
    extended_length = min(96, len(data))
    steps_extended = datetime_series[:extended_length]
    
    # 4日分の冷房需要データを取得
    cooling_load_extended = data['冷房需要[kW]'].to_numpy()[:extended_length]
    
    # 4日分のCO2排出係数データを取得
    co2_series_extended = data['CO2排出係数[kg-CO2/kWh]'].to_numpy()[:extended_length]

    # =========================
    # 図の作成（3つのパネル）
    # =========================
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    # =========================
    # 営業時間帯の背景色を設定
    # =========================
    # 営業時間（8時から18時）の背景を薄い緑色で塗る
    business_start_hour = 8
    business_end_hour = 18
    
    # 全てのパネルに4日分の営業時間帯の背景を追加
    # 4日目は薄い緑で差別化（シミュレーション期間外の参考情報として）
    unique_dates = steps_extended.dt.date.unique()
    for ax in axes:
        for idx, date in enumerate(unique_dates):
            # 各日の営業開始・終了時刻を計算
            day_start = pd.Timestamp(date) + pd.Timedelta(hours=business_start_hour)
            day_end = pd.Timestamp(date) + pd.Timedelta(hours=business_end_hour)
            # 4日目（最後の日）はより薄い緑で表示
            if idx == len(unique_dates) - 1:
                ax.axvspan(day_start, day_end, alpha=0.05, color='green', zorder=0)
            else:
                ax.axvspan(day_start, day_end, alpha=0.15, color='green', zorder=0)

    # =========================
    # パネル1: 冷房負荷（Cooling Load）とチラー負荷（Chiller Load）
    # =========================
    # 冷房需要は4日分のデータを表示（参考情報として）
    axes[0].step(steps_extended, cooling_load_extended, where='post',
                 label='Cooling Load [kW]', color='steelblue', alpha=0.8, linewidth=2.5)
    # チラー負荷は3日分のシミュレーション結果を表示（実際の運転結果）
    axes[0].step(steps, results['chiller_load_history'], where='post',
                 label='Chiller Load [kW]', color='coral', alpha=0.8, linewidth=2.5)
    
    # 軸ラベルと範囲の設定
    axes[0].set_ylabel('Load [kW]')
    axes[0].set_xlabel('Date')
    axes[0].set_ylim(0, 120)  # 冷房負荷の縦軸を0-120kWに固定（余裕を持たせる）
    axes[0].grid(True)  # グリッドを表示
    axes[0].legend(loc='upper right')  # 凡例を右上に配置
    
    # x軸の日時フォーマット設定
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    axes[0].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 8, 18]))  # 0時、8時、18時にマーカー
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')  # ラベルを45度回転
    
    # 横軸を4日分に拡張
    axes[0].set_xlim(steps_extended.min(), steps_extended.max())

    # =========================
    # パネル2: TES蓄熱量（TES Energy）
    # =========================
    # TES蓄熱量の時間変化をプロット
    axes[1].plot(steps_tes, results['tes_energy_history'], label='TES Energy [kWh]', 
                 color='darkorange', linewidth=1.5)
    
    # 軸ラベルと範囲の設定
    axes[1].set_ylabel('TES Energy [kWh]')
    axes[1].set_xlabel('Date')
    axes[1].set_ylim(0, 310)  # 蓄熱量の縦軸を0-310kWhに固定（容量300kWhに余裕を持たせる）
    axes[1].grid(True)  # グリッドを表示
    axes[1].legend(loc='upper right')  # 凡例を右上に配置
    
    # x軸の日時フォーマット設定
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    axes[1].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 8, 18]))  # 0時、8時、18時にマーカー
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')  # ラベルを45度回転
    
    # 横軸を4日分に拡張
    axes[1].set_xlim(steps_extended.min(), steps_extended.max())

    # =========================
    # パネル3: 消費電力（Electric Consumption）とCO2排出係数（CO2 Intensity）
    # =========================
    # 2つのy軸を持つパネルを作成（左: 消費電力、右: CO2排出係数）
    ax3 = axes[2]
    ax3_twin = ax3.twinx()  # 右側に第2のy軸を追加
    
    # 消費電力（左軸）をプロット
    line1, = ax3.step(steps, electric_consumption, where='post',
                      label='Electric Consumption [kW]', color='black', linewidth=1.5)
    ax3.set_ylabel('Electric Power [kW]')
    ax3.set_xlabel('Date')
    ax3.set_ylim(0, 28)  # 消費電力の縦軸を0-28kWに固定（余裕を持たせる）
    ax3.grid(True)  # グリッドを表示
    
    # CO2排出係数（右軸）を4日分まで延長して表示
    line2, = ax3_twin.step(steps_extended, co2_series_extended, where='post',
                           label='CO2 Intensity [kg-CO2/kWh]', color='red', alpha=0.7, linewidth=1.5)
    ax3_twin.set_ylabel('CO2 Intensity [kg-CO2/kWh]')
    ax3_twin.set_ylim(0, 0.62)  # CO2係数の縦軸を0-0.62 kg-CO2/kWhに固定（余裕を持たせる）
    
    # 両方の線を1つの凡例にまとめる
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper right')  # 凡例を右上に配置
    
    # x軸の日時フォーマット設定
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax3.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 8, 18]))  # 0時、8時、18時にマーカー
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')  # ラベルを45度回転
    
    # パネル3の横軸を4日分に拡張
    ax3.set_xlim(steps_extended.min(), steps_extended.max())

    # =========================
    # プロットの保存
    # =========================
    plt.tight_layout()  # レイアウトを自動調整（パネル間の重なりを防ぐ）
    
    # 出力ファイルパスの設定（拡張子がなければ.pngを追加）
    output_path = output_name if output_name.endswith('.png') else f'{output_name}.png'
    _ensure_parent_dir(output_path)  # 出力ディレクトリが存在することを確認
    
    # 画像ファイルとして保存
    plt.savefig(output_path, dpi=150)
    plt.close(fig)  # メモリ解放のため図を閉じる
