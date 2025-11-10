# ============================================
# 蓄熱システムの環境クラス（SAC用）
# ============================================
# 空調用蓄熱システムのシミュレーション環境
# 冷房需要とCO2排出係数に基づいて蓄熱運用を最適化

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from config import params


class ThermalStorageEnv:
    """
    蓄熱システムの強化学習環境
    冷房需要を満たしつつ、CO2排出量を最小化する蓄熱運用を学習
    """
    def __init__(self, external_data):
        """
        環境の初期化
        
        Args:
            external_data: 外部データ（冷房需要、CO2排出係数、時刻情報を含むDataFrame）
        """
        self.external_data = external_data.copy()
        
        # 営業時間の設定（8時～17時、18時閉店のため制御は17時台まで）
        self.business_hours = tuple(range(8, 18))
        # 低炭素時間帯の時間数（深夜電力利用など）
        self.low_carbon_hours = 5
        # シミュレーション日数
        self.simulation_days = 3
        # 正規化用の定数
        self.load_normalizer = 100.0  # 冷房需要の正規化係数
        self.co2_normalizer = 0.6     # CO2排出係数の正規化係数

        # データの前処理
        self._prepare_data()

        # シミュレーション状態の初期化
        self.current_step = 0  # 現在のステップ数
        self.simulation_steps = self.simulation_days * 24  # 総ステップ数
        self.max_steps = self.simulation_steps
        # 学習終了ステップ（最終日の営業時間終了時）
        self.learning_terminal_step = ((self.simulation_days - 1) * 24) + (self.business_hours[-1] + 1)
        
        # 蓄熱システムのパラメータ
        self.tes_capacity = params["capacity_tes"]  # 蓄熱容量 [kWh]
        self.tes_energy = 0 * self.tes_capacity     # 現在の蓄熱量 [kWh]
        
        # 終了フラグ
        self.done = False               # エピソード終了フラグ
        self.learning_done = False      # 学習ウィンドウ終了フラグ
        self.terminal_reward_given = False  # 終端報酬付与済みフラグ

        # チラー（冷凍機）のパラメータ
        self.chiller_capacity = params["capacity_ashp"]  # チラー容量 [kW]
        self.cop = params["cop_ashp"]                    # 成績係数（COP）

        # 行動空間の定義（ASHP出力比: 0%, 33%, 67%, 100%）
        self.action_set = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
        # 未充足需要に対するペナルティ係数
        self.unserved_penalty = params.get("unserved_load_penalty", 10.0)
        # 最終ステップのカーボンオフセット量
        self.last_carbon_offset = 0.0

        # 直前ステップの動作記録用
        self.last_tes_charge = 0.0      # 充熱量 [kWh]
        self.last_tes_discharge = 0.0   # 放熱量 [kWh]
        self.last_unserved_load = 0.0   # 未充足需要 [kW]

        # 連続放熱モードのカウンター（使用されていない可能性あり）
        self.consecutive_discharge_count = 0
        # 直前に選択したアクションのインデックス
        self.last_action_index = 1

    def reset(self, initial_tes_energy=0.0):
        """
        環境をリセットして初期状態に戻す
        
        Args:
            initial_tes_energy: 初期蓄熱量（蓄熱容量に対する比率 0.0～1.0）
        
        Returns:
            初期状態の観測値
        """
        self.current_step = 0
        self.tes_energy = initial_tes_energy * self.tes_capacity  # 初期蓄熱量を設定
        self.done = False
        self.learning_done = False
        self.terminal_reward_given = False
        # 履歴変数をリセット
        self.last_tes_charge = 0.0
        self.last_tes_discharge = 0.0
        self.last_unserved_load = 0.0
        self.last_carbon_offset = 0.0
        self.consecutive_discharge_count = 0
        self.last_action_index = 1  # デフォルトアクション
        return self._get_state()

    def step(self, action_index):
        """
        1ステップ実行：行動を実行して次の状態と報酬を返す
        
        Args:
            action_index: 選択された行動のインデックス（0～3）
        
        Returns:
            next_state: 次の状態
            reward: 報酬
            learning_done: 学習ウィンドウが終了したかどうか
            cooling_load: チラーが出力した冷却量 [kW]
            cop: 成績係数
            carbon_discharge: CO2排出量 [kg-CO2]
        """
        # 行動インデックスから実際の出力比率を取得
        action_ratio = self.action_set[action_index]
        self.last_action_index = action_index
        self.last_carbon_offset = 0.0

        # 最大ステップ数に達した場合は終了
        if self.current_step >= self.max_steps:
            self.done = True
            self.learning_done = True
            return self._get_state(), 0.0, self.learning_done, 0.0, self.cop, 0.0

        # 現在のステップのデータを取得
        row = self.external_data.iloc[self.current_step]
        load = row['冷房需要[kW]']              # 冷房需要
        co2_factor = row['CO2排出係数[kg-CO2/kWh]']  # CO2排出係数

        # チラーのスケジュール冷却出力を計算
        scheduled_cooling = action_ratio * self.chiller_capacity

        # ========== エネルギー収支の計算 ==========
        # 1. チラーで直接供給できる冷房量
        load_served_by_chiller = min(load, scheduled_cooling)
        remaining_load = load - load_served_by_chiller

        # 2. 蓄熱槽から放熱して供給
        tes_discharge = min(remaining_load, self.tes_energy)
        self.tes_energy -= tes_discharge  # 蓄熱量を減らす
        remaining_load -= tes_discharge
        unserved_load = max(0.0, remaining_load)  # 未充足需要

        # 3. 余剰冷房量を蓄熱槽に充熱
        surplus_cooling = max(0.0, scheduled_cooling - load_served_by_chiller)
        available_tes_space = self.tes_capacity - self.tes_energy  # 蓄熱可能な残容量
        tes_charge = min(surplus_cooling, available_tes_space)
        self.tes_energy += tes_charge  # 蓄熱量を増やす

        # 履歴変数を更新
        self.last_tes_charge = tes_charge
        self.last_tes_discharge = tes_discharge
        self.last_unserved_load = unserved_load

        # ========== CO2排出量と報酬の計算 ==========
        cooling_load = scheduled_cooling  # チラーの実際の出力

        # 電力消費量を計算（COPで割る）
        electric_consumption = cooling_load / self.cop
        # CO2排出量を計算
        carbon_discharge = electric_consumption * co2_factor

        # 報酬の計算：CO2排出量と未充足需要のペナルティ
        penalty = self.unserved_penalty * unserved_load
        reward = -1.0 * ((carbon_discharge + penalty) / 6.0)  # 正規化のため6.0で割る

        # ステップを進める
        self.current_step += 1

        # 学習ウィンドウ終了の判定（最終日の営業時間終了時）
        learning_window_reached = self.current_step >= self.learning_terminal_step

        # ========== 終端報酬の計算 ==========
        # 学習ウィンドウ終了時に、残っている蓄熱量を評価
        if not self.terminal_reward_given and learning_window_reached:
            # 翌日の低CO2時間帯の平均排出係数を取得
            next_day_low_co2_avg = self._get_next_day_low_co2_average(row['date'], 'CO2排出係数[kg-CO2/kWh]')
            # 蓄熱量を翌日に使用した場合のCO2削減効果を計算
            carbon_offset = next_day_low_co2_avg * (self.tes_energy / self.cop)
            # 終端ボーナスとして報酬に加算
            terminal_bonus = carbon_offset / 6.0
            reward += terminal_bonus
            self.terminal_reward_given = True
            self.last_carbon_offset = carbon_offset
        
        # 学習終了フラグを更新
        self.learning_done = learning_window_reached

        # 次の状態を取得
        next_state = self._get_state()

        # エピソード終了の判定
        if self.current_step >= self.max_steps:
            self.done = True

        return next_state, reward, self.learning_done, cooling_load, self.cop, carbon_discharge

    def _get_state(self):
        """
        現在の観測状態を取得
        
        Returns:
            状態ベクトル（numpy配列）：
            - 営業時間内の正規化位置
            - 営業時間フラグ
            - 正規化蓄熱量
            - 前回の行動インデックス
            - 正規化冷房需要
            - 正規化CO2排出係数
            - 冷房需要予測（営業時間内の各時間）
            - CO2排出係数予測（営業時間内の各時間）
            - 翌日の平均冷房需要
            - 翌日の低CO2時間帯平均排出係数
            - 翌日のCO2排出係数最小値
        """
        # 現在のステップのインデックスを取得（範囲外にならないように制限）
        current_step_idx = min(self.current_step, len(self.external_data) - 1, self.max_steps - 1)
        current_row = self.external_data.iloc[current_step_idx]

        hour = int(current_row['hour'])
        
        # 営業時間内の正規化位置: 8時=0.0, 17時=1.0（18時閉店）
        if hour in self.business_hours:
            start_hour = self.business_hours[0]
            end_hour = self.business_hours[-1]
            span = max(end_hour - start_hour, 1)
            normalized_business_time = (hour - start_hour) / span  # 0.0 ~ 1.0
            is_business_hour = 1.0
        else:
            # 営業時間外は-1.0で表現
            normalized_business_time = -1.0
            is_business_hour = 0.0

        # 現在の状態を正規化
        normalized_tes_energy = self.tes_energy / self.tes_capacity  # 蓄熱率
        normalized_load = current_row['冷房需要[kW]'] / self.load_normalizer  # 冷房需要
        normalized_co2 = current_row['CO2排出係数[kg-CO2/kWh]'] / self.co2_normalizer  # CO2排出係数

        current_day = current_row['date']

        # 営業時間内の冷房需要とCO2排出係数の予測を取得
        load_forecast = self._build_business_hour_forecast(current_day, hour, '冷房需要[kW]', self.load_normalizer)
        co2_forecast = self._build_business_hour_forecast(current_day, hour, 'CO2排出係数[kg-CO2/kWh]', self.co2_normalizer)

        # 翌日の情報を取得
        next_day_load_avg = self._get_next_day_business_average(current_day, '冷房需要[kW]') / self.load_normalizer
        next_day_low_co2_avg = self._get_next_day_low_co2_average(current_day, 'CO2排出係数[kg-CO2/kWh]') / self.co2_normalizer
        next_day_co2_min = self._get_next_day_co2_min(current_day, 'CO2排出係数[kg-CO2/kWh]') / self.co2_normalizer

        # 状態ベクトルを構築
        return np.array([
            normalized_business_time,    # 営業時間内の位置
            is_business_hour,             # 営業時間フラグ
            normalized_tes_energy,        # 蓄熱率
            float(self.last_action_index),  # 前回の行動
            normalized_load,              # 現在の冷房需要
            normalized_co2,               # 現在のCO2排出係数
            *load_forecast,               # 冷房需要予測（営業時間内）
            *co2_forecast,                # CO2排出係数予測（営業時間内）
            next_day_load_avg,            # 翌日の平均冷房需要
            next_day_low_co2_avg,         # 翌日の低CO2時間帯平均排出係数
            next_day_co2_min              # 翌日のCO2排出係数最小値
        ])

    def render(self):
        """現在の状態を画面に表示（デバッグ用）"""
        print(f"Step: {self.current_step}, TES Energy: {self.tes_energy:.2f} kWh")

    def get_current_hour(self):
        """現在のステップの時刻を取得"""
        idx = min(self.current_step, self.max_steps - 1)
        return int(self.external_data.iloc[idx]['hour'])

    def is_business_hour(self, hour):
        """指定された時刻が営業時間内かどうかを判定"""
        return hour in self.business_hours

    def is_learning_done(self):
        """学習ウィンドウが終了したかどうかを返す"""
        return self.learning_done

    def _prepare_data(self):
        """
        外部データの前処理と統計情報の計算
        日付・時刻の解析、営業時間外の需要ゼロ化、各種集計値の事前計算を行う
        """
        # datetime列の存在確認
        if 'datetime' not in self.external_data.columns:
            raise ValueError("external_data must include a 'datetime' column")

        # 日時データの処理
        self.external_data['datetime'] = pd.to_datetime(self.external_data['datetime'])
        self.external_data.sort_values('datetime', inplace=True)
        self.external_data.reset_index(drop=True, inplace=True)
        self.external_data['hour'] = self.external_data['datetime'].dt.hour
        
        # 18-19時は営業外のため冷房需要をゼロ化
        self.external_data.loc[self.external_data['hour'].between(18, 19), '冷房需要[kW]'] = 0.0
        self.external_data['date'] = self.external_data['datetime'].dt.date

        # シミュレーション期間のデータのみを保持
        # simulation_days + 1日分必要（予測用に翌日データも使用）
        unique_dates = list(dict.fromkeys(self.external_data['date']))
        if len(unique_dates) < self.simulation_days + 1:
            raise ValueError("external_data must cover at least simulation_days + 1 days")

        reference_dates = unique_dates[: self.simulation_days + 1]
        self.external_data = self.external_data[self.external_data['date'].isin(reference_dates)].reset_index(drop=True)

        # 日付関連の情報を保存
        self.reference_dates = reference_dates  # 参照用の全日付
        self.simulation_dates = reference_dates[: self.simulation_days]  # シミュレーション対象日
        self.date_to_position = {date: idx for idx, date in enumerate(self.reference_dates)}  # 日付→位置マッピング

        # 日付と時刻からデータインデックスを取得するための辞書を構築
        self.day_hour_to_index = {}
        self.day_to_business_indices = {date: [] for date in self.reference_dates}
        for idx, row in self.external_data.iterrows():
            day = row['date']
            hour = int(row['hour'])
            self.day_hour_to_index.setdefault(day, {})[hour] = idx
            # 営業時間のインデックスを記録
            if hour in self.business_hours:
                self.day_to_business_indices.setdefault(day, []).append(idx)

        # 営業時間内の平均値を計算するヘルパー関数
        def compute_business_average(day, column):
            """指定日の営業時間内の平均値を計算"""
            indices = self.day_to_business_indices.get(day, [])
            if not indices:
                return 0.0
            return float(self.external_data.loc[indices, column].mean())

        # 営業時間内の最小値N時間の平均を計算するヘルパー関数
        def compute_lowest_business_average(day, column, hours):
            """指定日の営業時間内で値が最も小さいN時間の平均値を計算"""
            indices = self.day_to_business_indices.get(day, [])
            if not indices:
                return 0.0
            series = self.external_data.loc[indices, column]
            if series.empty:
                return 0.0
            count = min(int(hours), len(series))
            lowest_values = series.nsmallest(count)  # 最小値N個を抽出
            if lowest_values.empty:
                return 0.0
            return float(lowest_values.mean())

        # 各日の営業時間内の平均値を事前計算
        self.day_to_business_avg = {
            day: {
                '冷房需要[kW]': compute_business_average(day, '冷房需要[kW]'),
                'CO2排出係数[kg-CO2/kWh]': compute_business_average(day, 'CO2排出係数[kg-CO2/kWh]')
            }
            for day in self.reference_dates
        }

        # 各日の低CO2時間帯（最小N時間）の平均CO2排出係数を事前計算
        self.day_to_lowest_co2_avg = {
            day: compute_lowest_business_average(day, 'CO2排出係数[kg-CO2/kWh]', self.low_carbon_hours)
            for day in self.reference_dates
        }

        # 営業時間内の最小値を計算するヘルパー関数
        def compute_business_min(day, column):
            """指定日の営業時間内の最小値を計算"""
            indices = self.day_to_business_indices.get(day, [])
            if not indices:
                return 0.0
            return float(self.external_data.loc[indices, column].min())

        # 各日のCO2排出係数の最小値を事前計算
        self.day_to_co2_min = {
            day: compute_business_min(day, 'CO2排出係数[kg-CO2/kWh]')
            for day in self.reference_dates
        }

    def _build_business_hour_forecast(self, day, current_hour, column, normalizer):
        """
        当日の営業時間内の予測値を構築
        現在時刻より前の時間は0、以降の時間は実際の値を返す
        
        Args:
            day: 対象日
            current_hour: 現在の時刻
            column: データ列名
            normalizer: 正規化係数
        
        Returns:
            営業時間内の予測値リスト（正規化済み）
        """
        forecast = []
        hour_map = self.day_hour_to_index.get(day, {})
        for hour in self.business_hours:
            if hour <= current_hour:
                # 過去の時間は0で埋める（既に経過）
                forecast.append(0.0)
            else:
                # 未来の時間は実際の値を使用
                idx = hour_map.get(hour)
                if idx is None:
                    forecast.append(0.0)
                else:
                    value = self.external_data.iloc[idx][column]
                    forecast.append(value / normalizer)
        return forecast

    def _get_next_day_business_average(self, current_day, column):
        """
        翌日の営業時間内平均値を取得
        
        Args:
            current_day: 現在の日付
            column: データ列名
        
        Returns:
            翌日の営業時間内平均値
        """
        position = self.date_to_position.get(current_day)
        if position is None:
            return 0.0
        next_position = position + 1
        if next_position >= len(self.reference_dates):
            return 0.0
        next_day = self.reference_dates[next_position]
        return self.day_to_business_avg.get(next_day, {}).get(column, 0.0)

    def _get_next_day_low_co2_average(self, current_day, column='CO2排出係数[kg-CO2/kWh]'):
        """
        翌日の低CO2時間帯の平均排出係数を取得
        
        Args:
            current_day: 現在の日付
            column: データ列名（デフォルトはCO2排出係数）
        
        Returns:
            翌日の低CO2時間帯（最小N時間）の平均排出係数
        """
        position = self.date_to_position.get(current_day)
        if position is None:
            return 0.0
        next_position = position + 1
        if next_position >= len(self.reference_dates):
            return 0.0
        next_day = self.reference_dates[next_position]
        return float(self.day_to_lowest_co2_avg.get(next_day, 0.0))

    def _get_next_day_co2_min(self, current_day, column='CO2排出係数[kg-CO2/kWh]'):
        """
        翌日のCO2排出係数最小値を取得
        
        Args:
            current_day: 現在の日付
            column: データ列名（デフォルトはCO2排出係数）
        
        Returns:
            翌日の営業時間内のCO2排出係数最小値
        """
        position = self.date_to_position.get(current_day)
        if position is None:
            return 0.0
        next_position = position + 1
        if next_position >= len(self.reference_dates):
            return 0.0
        next_day = self.reference_dates[next_position]
        return float(self.day_to_co2_min.get(next_day, 0.0))
